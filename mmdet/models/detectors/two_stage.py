# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import json

from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_roi_extractor, HEADS, build_loss
from .base import BaseDetector
from ...core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, multi_apply, multiclass_nms
import cv2





@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.used = json.load(open(test_cfg['external_proposal_path']))

        self.saliency_transformer = cv2.saliency.StaticSaliencyFineGrained_create()
        self.sal_linear1 = nn.Conv2d(3, 3, 1, bias=True)
        self.sal_linear2 = nn.Conv2d(3, 3, 1, bias=True)

        if neck is not None:
            self.neck = build_neck(neck)


        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs




    def saliency_to_img(self,img_metas,imgs,json_path,mode,bias):
        if mode < 2:
            all_data = json_path
            min_ratio = self.test_cfg['external_proposal_min_box_area']
            max_ratio = self.test_cfg['external_proposal_max_box_area']
            rpn_score = self.test_cfg['rpn_score_thr']
        img_num = len(img_metas)
        device = imgs.device
        imgs_out = torch.ones_like(imgs)

        for i in range(img_num):
            img_meta = img_metas[i]
            filepath = img_meta['filename']
            img_fast = imgs[i]
            img = cv2.imread(filepath)
            img_ori = img.copy()

            img = img.astype('int16')
            ori_w = img_meta['ori_shape'][1]
            ori_h = img_meta['ori_shape'][0]
            new_w = img_meta['img_shape'][1]
            new_h = img_meta['img_shape'][0]
            w_ratio = ori_w / new_w
            h_ratio = ori_h / new_h
            pic_name = img_meta['ori_filename']
            if mode < 2:
                img_info = all_data[pic_name]
                bboxs_info = img_info['bbox']
                bboxs_info = np.array(bboxs_info)

                if len(bboxs_info)>0:

                    bb_scores = bboxs_info[:,-1]

                    idx = bb_scores>rpn_score
                    bboxs_info = bboxs_info[idx,:]
                    chosen_arear_max = img_info['w']*img_info['h']*max_ratio
                    chosen_arear_min = img_info['w']*img_info['h']*min_ratio
                    bbox_area = abs((bboxs_info[:,0]-bboxs_info[:,2])*(bboxs_info[:,1]-bboxs_info[:,3]))
                    idx = bbox_area<chosen_arear_max
                    bboxs_info = bboxs_info[idx,:-1]
                    bboxs_info = bboxs_info.tolist()
                else:
                    bboxs_info=[]


            else:
                bboxs_info = json_path[i]
            for bbox_info in bboxs_info:
                if mode < 2:
                    bbox = bbox_info
                    start_point = (int(bbox[0]), int(bbox[1]))
                    end_point = (int(bbox[2]), int(bbox[3]))
                else:
                    bbox = bbox_info
                    start_point = (int(bbox[0] * w_ratio), int(bbox[1] * h_ratio))
                    end_point = (int(bbox[2] * w_ratio), int(bbox[3] * h_ratio))

                img[start_point[1]:end_point[1], start_point[0]:end_point[0], :] = 1000

            index = img < 800
            img[index] = 0
            img = np.minimum(img_ori, img)
            img = img.astype('uint8')
            (suc, img_sal) = self.saliency_transformer.computeSaliency(img)
            if suc:
                img_sal = (img_sal * 255).astype('uint8')
                img_sal = cv2.threshold(img_sal, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                img_sal = cv2.resize(img_sal,(new_w,new_h),interpolation = cv2.INTER_LINEAR)
                img_sal = img_sal/255*bias
                img_idx = img_sal<bias*0.3
                img_sal[img_idx]=0

                img_sal = torch.tensor(img_sal,dtype=torch.float,device=device)


                img_fast_new = torch.ones_like(img_fast)
                for kkk in range(len(img_fast)):
                    img_fast_new[kkk,:,:]=img_sal
                img_fast_new = self.sal_linear1(img_fast_new)
                img_fast_new = self.sal_linear2(img_fast_new)
                img_fast_new = img_fast+img_fast_new
                imgs_out[i,:,:,:]=img_fast_new
            else:
                imgs_out[i,:,:,:]=img_fast
        return imgs_out




    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.train_cfg['external_feature_bias']>0.05:
            img = self.saliency_to_img(img_metas,img,gt_bboxes, 5,self.train_cfg['external_feature_bias'])


        x = self.extract_feat(img)


        losses = dict()


        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            # enforce proposal from saliency map into gt_boxes
            rpn_losses, proposal_list_ext = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list_ext = proposals



        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list_ext,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)


        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'

        if self.test_cfg['external_feature_bias'] > 0.05:
            img = self.saliency_to_img(img_metas, img, self.used, 0.5, self.test_cfg['external_feature_bias'])

        x = self.extract_feat(img)


        if proposals is None:
            proposal_list_ext = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list_ext = proposals

        return self.roi_head.simple_test(
            x, proposal_list_ext, img_metas, rescale=rescale)


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'
                # noqa E501
            )
