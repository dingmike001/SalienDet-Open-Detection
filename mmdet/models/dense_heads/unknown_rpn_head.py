import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.core.bbox import bbox_overlaps
from ..builder import HEADS, build_loss
from .rpn_head import RPNHead


@HEADS.register_module()
class uknrpnhead(RPNHead):

    def __init__(self, loss_unknown, unknown_type='Centerness', **kwargs):
        super(uknrpnhead, self).__init__(**kwargs)
        # Objectness loss
        self.loss_unknown = build_loss(loss_unknown)
        self.unknown_type = unknown_type
        self.with_class_score = self.loss_cls.loss_weight > 0.0
        self.with_objectness_score = self.loss_unknown.loss_weight > 0.0

        # Define objectness assigner and sampler
        if self.train_cfg:
            self.unknown_assigner = build_assigner(
                self.train_cfg.unknown_assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'unknown_sampler'):
                unknown_sampler_cfg = self.train_cfg.unknown_sampler
            else:
                unknown_sampler_cfg = dict(type='PseudoSampler')
            self.unknown_sampler = build_sampler(
                unknown_sampler_cfg, context=self)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

        assert self.num_base_priors == 1 and self.cls_out_channels == 1
        self.rpn_ukn = nn.Conv2d(self.feat_channels, self.num_anchors, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)
        normal_init(self.rpn_ukn, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        x = F.normalize(x, p=2, dim=1)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        rpn_ukn_pred = self.rpn_ukn(x)
        return rpn_cls_score, rpn_bbox_pred, rpn_ukn_pred

    def loss_single(self, cls_score, bbox_pred, unknown_score, anchors,
                    labels, label_weights, bbox_targets, bbox_weights,
                    unknown_targets, unknown_weights, num_total_samples):

        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        # unknown loss
        unknown_targets = unknown_targets.reshape(-1)
        unknown_weights = unknown_weights.reshape(-1)
        assert self.cls_out_channels == 1, (
            'cls_out_channels must be 1 for objectness learning.')
        unknown_score = unknown_score.permute(0, 2, 3, 1).reshape(-1)

        loss_unknown = self.loss_unknown(
            unknown_score.sigmoid(),
            unknown_targets,
            unknown_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox, loss_unknown

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'unknown_scores'))
    def loss(self,
             cls_scores,
             bbox_preds,
             unknown_scores,
             gt_bboxes,
             # gt_labels,  # gt_labels is not used since we sample the GTs.
             img_metas,
             gt_bboxes_ignore=None,
             gt_labels=None):  # gt_labels is not used since we sample the GTs.

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_unknown_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_unknown_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, unknown_targets_list,
         unknown_weights_list) = cls_reg_unknown_targets

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_unknown = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            unknown_scores,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            unknown_targets_list,
            unknown_weights_list,
            num_total_samples=num_total_samples)

        return dict(
            loss_rpn_cls=losses_cls,
            loss_rpn_bbox=losses_bbox,
            loss_rpn_obj=losses_unknown, )

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        # Assign unknown gt and sample anchors
        unknown_assign_result = self.unknown_assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore, None)
        unknown_sampling_result = self.unknown_sampler.sample(
            unknown_assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes

            # Sanlity check: left, right, top, bottom distances must be greater
            # than 0.
            valid_targets = torch.min(pos_bbox_targets, -1)[0] > 0
            bbox_targets[pos_inds[valid_targets], :] = (
                pos_bbox_targets[valid_targets])
            bbox_weights[pos_inds[valid_targets], :] = 1.0

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        unknown_targets = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)
        unknown_weights = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)
        unknown_pos_inds = unknown_sampling_result.pos_inds
        unknown_neg_inds = unknown_sampling_result.neg_inds
        unknown_pos_neg_inds = torch.cat(
            [unknown_pos_inds, unknown_neg_inds])

        if len(unknown_pos_inds) > 0:
            # Centerness as tartet -- Default
            if self.unknown_type == 'Centerness':
                pos_unknown_bbox_targets = self.bbox_coder.encode(
                    unknown_sampling_result.pos_bboxes,
                    unknown_sampling_result.pos_gt_bboxes)
                valid_targets = torch.min(pos_unknown_bbox_targets, -1)[0] > 0
                pos_unknown_bbox_targets[valid_targets == False, :] = 0
                top_bottom = pos_unknown_bbox_targets[:, 0:2]
                left_right = pos_unknown_bbox_targets[:, 2:4]
                pos_unknown_targets = torch.sqrt(
                    (torch.min(top_bottom, -1)[0] /
                     (torch.max(top_bottom, -1)[0] + 1e-12)) *
                    (torch.min(left_right, -1)[0] /
                     (torch.max(left_right, -1)[0] + 1e-12)))
            elif self.unknown_type == 'BoxIoU':
                pos_unknown_targets = bbox_overlaps(
                    unknown_sampling_result.pos_bboxes,
                    unknown_sampling_result.pos_gt_bboxes,
                    is_aligned=True)
            else:
                raise ValueError(
                    'unknown_type must be either "Centerness" (Default) or '
                    '"BoxIoU".')

            unknown_targets[unknown_pos_inds] = pos_unknown_targets
            unknown_weights[unknown_pos_inds] = 1.0

        if len(unknown_neg_inds) > 0:
            unknown_targets[unknown_neg_inds] = 0.0
            unknown_weights[unknown_neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

            # unknown targets
            unknown_targets = unmap(
                unknown_targets, num_total_anchors, inside_flags)
            unknown_weights = unmap(
                unknown_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result,
                unknown_targets, unknown_weights,
                unknown_pos_inds, unknown_neg_inds, unknown_pos_neg_inds,
                unknown_sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):

        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list,
         all_unknown_targets, all_unknown_weights,
         unknown_pos_inds_list, unknown_neg_inds_list,
         unknown_pos_neg_inds_list, unknown_sampling_results_list
         ) = results[:13]

        rest_results = list(results[13:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        unknown_targets_list = images_to_levels(all_unknown_targets,
                                                num_level_anchors)
        unknown_weights_list = images_to_levels(all_unknown_weights,
                                                num_level_anchors)

        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg,
               unknown_targets_list, unknown_weights_list,)

        if return_sampling_results:
            res = res + (sampling_results_list,)
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'unknown_scores'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   unknown_scores,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            unknown_scores (list[Tensor]): Box unknown scorees for each anchor
                point with shape (N, num_anchors, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.

        """
        assert len(cls_scores) == len(bbox_preds) and (
                len(cls_scores) == len(unknown_scores))
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            unknown_score_list = [
                unknown_scores[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    unknown_score_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    unknown_score_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           unknown_scores,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):

        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            # <
            rpn_unknown_score = unknown_scores[idx]
            # >

            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)

            assert self.use_sigmoid_cls, 'use_sigmoid_cls must be True.'
            rpn_cls_score = rpn_cls_score.reshape(-1)
            rpn_cls_scores = rpn_cls_score.sigmoid()

            rpn_unknown_score = rpn_unknown_score.permute(
                1, 2, 0).reshape(-1)
            rpn_unknown_scores = rpn_unknown_score.sigmoid()

            # We use the predicted unknown score (i.e., localization quality)
            # as the final RPN score output.
            scores = rpn_unknown_scores

            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0),), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # nms_cfg = dict(type=cfg.nms.type, iou_threshold=cfg.nms.iou_threshold)

        # No NMS:
        # need to plus NMS

        dets = torch.cat([proposals, scores.unsqueeze(1)], 1)

        return dets
