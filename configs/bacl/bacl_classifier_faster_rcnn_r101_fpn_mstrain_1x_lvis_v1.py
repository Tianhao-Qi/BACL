_base_ = './bacl_classifier_faster_rcnn_r50_fpn_mstrain_1x_lvis_v1.py'

model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(depth=101))

load_from = './work_dirs/bacl_representation_faster_rcnn_r101_fpn_1x_lvis_v1/epoch_12.pth'