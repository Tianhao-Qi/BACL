import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import json

from mmdet.utils import get_root_logger
from ..builder import LOSSES


# ABL stands for Accuracy_BaLance loss
# use redefined reweight, collect correct_cls_samples from different gpus
@LOSSES.register_module()
class FCBL(nn.Module):
    def __init__(self,
                 prob_thr=0.7,
                 use_sigmoid=True,
                 reduction='mean',
                 loss_weight=1.0,
                 num_classes=1230,
                 alpha=1.0):
        super(FCBL, self).__init__()
        self.prob_thr = prob_thr
        assert self.prob_thr > 0 and self.prob_thr < 1

        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.alpha = alpha
        self.reweight = False

        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True

        # initial variables
        logger = get_root_logger()
        logger.info(f"build FCBL loss, alpha: {alpha}, prob_thr: {prob_thr}")

        # cumulative samples for each category
        self.register_buffer(
            'cm_numerator',
            torch.zeros((self.num_classes, self.num_classes), dtype=torch.float))
        self.register_buffer(
            'num_gt',
            torch.zeros(self.num_classes, dtype=torch.float))

    def get_cm(self):
        return self.cm_numerator / self.num_gt[:, None].clamp(min=1.0)

    def get_correct_cls_ratio(self):
        cm = self.get_cm()
        return torch.diag(cm)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        pos_inds = label < self.num_classes
        neg_inds = label == self.num_classes

        def expand_label(pred):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), self.gt_classes] = 1
            return target
        
        target = expand_label(cls_score)

        # calculate margin
        margin = cls_score.new_zeros(self.n_i, self.n_c)
        cm = self.get_cm()
        numerator = cm[label[pos_inds], :].clamp(min=1e-3)
        denominator = cm[:, label[pos_inds]].clamp(min=1e-3)
        margin[pos_inds, :self.num_classes] = (numerator / denominator.t()).log() * self.alpha

        # calculate weight
        if self.reweight:
            weight = self.get_weight(cls_score.detach(), pos_inds)
        else:
            weight = cls_score.new_ones(self.n_i, self.n_c)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score + margin, target,
                                                      reduction='none')
        cls_loss = torch.sum(weight * cls_loss) / self.n_i

        if pos_inds.sum() > 0:
            self.update_cm(cls_score.detach(), target, pos_inds)

        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def get_weight(self, cls_score, pos_inds):
        # device = cls_score.device
        prob = self.get_activation(cls_score[pos_inds])
        weight = cls_score.new_ones(self.n_i, self.n_c)
        n_i = cls_score[pos_inds].size(0)
        gt_prob_mat = prob[torch.arange(n_i), self.gt_classes[pos_inds]].unsqueeze(1).expand(n_i, self.n_c)
        weight_fg = cls_score.new_zeros(n_i, self.n_c)
        weight_fg[prob >= gt_prob_mat] = 1
        weight_fg[prob >= self.prob_thr] = 1
        weight_fg[:, self.num_classes] = 1
        weight[pos_inds] = weight_fg
        return weight

    def update_cm(self, cls_score, target, pos_inds):
        pred_fg_distri = F.softmax(cls_score[pos_inds, :self.num_classes], dim=1)
        M = torch.mm(target[pos_inds, :self.num_classes].t(), pred_fg_distri)
        I = torch.zeros_like(self.num_gt).scatter_add_(0, self.gt_classes[pos_inds], torch.ones(pos_inds.sum()).to(cls_score.device))
        dist.all_reduce(M)
        dist.all_reduce(I)
        self.cm_numerator += M
        self.num_gt += I