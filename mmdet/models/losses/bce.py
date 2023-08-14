import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmdet.models.losses import accuracy
from mmdet.utils import get_root_logger
from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class BCE(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 loss_weight=1.0,
                 num_classes=1230):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        
        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True

        # initial variables
        logger = get_root_logger()
        logger.info(f"build BCE loss")

    def forward(self,
                cls_score,
                label,
                label_weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label

        def expand_label(pred):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), self.gt_classes] = 1
            return target

        target = expand_label(cls_score)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')
        # cls_loss = torch.sum(cls_loss) / self.n_i
        label_weight = label_weight.view(-1, 1).expand(
            label_weight.size(0), self.n_c)
        cls_loss = weight_reduce_loss(
            cls_loss, label_weight, reduction=reduction, avg_factor=avg_factor)
        return self.loss_weight * cls_loss

    def get_cls_channels(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        cls_score[:, :-1] *= (1 - bg_score)
        return cls_score
    
    def get_accuracy(self, cls_score, labels):
        acc = dict()
        acc['acc'] = accuracy(cls_score, labels)
        return acc