_base_ = './faster_rcnn_r50_fpn_mstrain_1x_lvis_v0.5.py'

model = dict(
    roi_head=dict(
        type='FhmRoIHead',
        bbox_head=dict(
            type='FhmShared2FCBBoxHead',
            loss_cls=dict(
                type='FCBL', use_sigmoid=True, loss_weight=1.0, num_classes=1230, alpha=0.85, prob_thr=0.7),
            fhm_cfg=dict(decay_ratio=0.1,
                         sampled_num_classes=8,
                         sampled_num_features=12))),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_post=2000,
            max_num=2000)))

# custom hooks: ReweightHook is defined in mmdet/core/utils/reweight_hook.py
custom_hooks = [
    dict(
        type="ReweightHook",
        step=1)
]

load_from = './work_dirs/bacl_representation_faster_rcnn_r50_fpn_1x_lvis_v0.5/epoch_12.pth'

# Train which part, 0 for all, 1 for fc_cls, fc_reg, rpn and mask_head(if exists)
selectp = 1