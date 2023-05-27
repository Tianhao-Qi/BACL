_base_ = './faster_rcnn_r50_fpn_mstrain_2x_lvis_v1.py'

model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(depth=101))