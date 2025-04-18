_base_ = './mask_rcnn_r50_fpn_normed_mask_mstrain_1x_lvis_v1.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='BCE', use_sigmoid=True, loss_weight=1.0, num_classes=1203))),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_post=2000,
            max_num=2000)))

dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis_v1/'
image_size = (1280, 1280)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.8, 1.25),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size=image_size),
]
train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/lvis_v1_train.json',
            img_prefix=data_root,
            pipeline=load_pipeline),
        pipeline=train_pipeline))
optimizer =dict(weight_decay=0.00005)