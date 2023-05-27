_base_ = './bacl_representation_faster_rcnn_r50_fpn_1x_lvis_v0.5.py'

model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(depth=101))