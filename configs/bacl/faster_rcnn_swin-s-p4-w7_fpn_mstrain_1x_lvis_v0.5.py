_base_ = './faster_rcnn_swin-s-p4-w7_fpn_mstrain_2x_lvis_v0.5.py'

lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)