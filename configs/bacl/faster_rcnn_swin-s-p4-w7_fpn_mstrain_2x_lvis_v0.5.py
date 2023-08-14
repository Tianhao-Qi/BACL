_base_ = './faster_rcnn_swin-t-p4-w7_fpn_mstrain_2x_lvis_v0.5.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)