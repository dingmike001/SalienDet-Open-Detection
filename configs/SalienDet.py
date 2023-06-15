_base_ = [
    '_base_/datasets/coco_detection.py',
    '_base_/default_runtime.py'
]

model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        # init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')
    # ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='uknrpnhead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            # ratios=[0.5, 1.0, 2.0],
            ratios=[1.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='TBLRBBoxCoder',
            normalizer=1.0,),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='IoULoss', mode='linear', loss_weight=10.0),
        unknown_type='Centerness',
        loss_unknown=dict(type='L1Loss', loss_weight=1.0),
        ),
    roi_head=dict(
        type='unkRoihead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxScoreHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            bbox_score_type='BoxIoU',
            loss_bbox_score=dict(type='L1Loss', loss_weight=1.0),
            # loss_bbox_obj_pred=dict(type='L1Loss', loss_weight=1.0),
        )),
    # model training and testing settings
    train_cfg=dict(
        ex_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
        ex_sampler=dict(
                type='RandomSampler',
                num=4096,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
        proposal_thrain_thr=0.8,
        proposal_samples = 4096,
        proposal_new_samples = 2000,
        proposal_bbox_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=1,
            featmap_strides=[4, 8, 16, 32]),
        external_loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        external_requrie_grad=True,
        external_proposal_path='/home/mike/Documents/dataset/neu_scene/saliency_map/All_Proposal',
        external_proposal_num=11999,
        external_proposal_min_box_area=1000,
        external_proposal_max_box_area=250000,
        proposal_to_gtbox=True,
        external_feature_bias=0.5,
        external_feature_used=True,
        rpn=dict(
            unknown_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.3,
                neg_iou_thr=0.1,
                min_pos_iou=0.3,
                # match_low_quality=True,
                ignore_iof_thr=-1
            ),
            unknown_sampler=dict(
                type='RandomSampler',
                num=256,
                # Ratio 0 for negative samples.
                pos_fraction=1.,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=1,
                # neg_pos_ub=500,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            # allowed_border=-1,
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        proposal_bbox_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=1,
            featmap_strides=[4, 8, 16, 32]),
        external_feature_bias=0.5,
        external_feature_used=True,
        external_proposal_path='/dataset/All_Proposal.json',
        external_proposal_num=11999,
        external_proposal_min_box_area=1000,
        external_proposal_max_box_area=250000,
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            max_per_img=2000,
            # nms=dict(type='nms', iou_threshold=0.7),
            nms=dict(type='nms', iou_threshold=0),
            min_bbox_size=0,
            ),
        rcnn=dict(
            score_thr=0.6,
            obj_score_thr=0.5,
            nms=dict(type='nms', iou_threshold=0.7),
            obj_class_nms=dict(type='nms', iou_threshold=0.5),


            max_per_img=1500,
            class_score_thr=0.000002,
            class_nms=dict(type='nms', iou_threshold=0.5),
            class_max_per_img=1500
        )
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

dataset_type = 'CocoDataset'
classes = ('object',)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=False,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=(1333, 800),
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='Pad', size=(750, 1333), pad_val=0),
    # dict(type='Pad', size=(416, 1344), pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='Pad', size=(750, 1333), pad_val=0),
            # dict(type='Pad', size=(416, 1344), pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        img_prefix='/dataset/train',
        classes=classes,
        ann_file='/dataset/ex',
        type=dataset_type,
        pipeline=train_pipeline
    ),
    val=dict(
        img_prefix='/dataset/val',
        classes=classes,
        ann_file='/dataset/train_demo_object.json',
        type=dataset_type,
        pipeline=test_pipeline),
    test=dict(
        img_prefix='/dataset/val',
        classes=classes,
        ann_file='/dataset/train_demo_object.json',
        type=dataset_type,
        pipeline=test_pipeline))


evaluation = dict(interval=5)
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(
    policy='fixed',
)
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
auto_scale_lr = dict(enable=False, base_batch_size=32)

