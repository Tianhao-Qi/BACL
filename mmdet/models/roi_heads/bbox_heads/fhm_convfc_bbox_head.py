# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead
from mmdet.models.losses import accuracy


@HEADS.register_module()
class FhmConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 fhm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(FhmConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.decay_ratio = fhm_cfg.get('decay_ratio', 0.1)
        self.sampled_num_classes = fhm_cfg.get('sampled_num_classes', 8)
        self.sampled_num_features = fhm_cfg.get('sampled_num_features', 4)
        self.start_epoch = fhm_cfg.get('start_epoch', 0)
        self._epoch = 0
        self.flag = False

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)
            
        self.feature_prototype = nn.Parameter(
            torch.zeros(self.num_classes, self.cls_last_dim),
            requires_grad=False)
        self.feature_variance = nn.Parameter(
            torch.zeros(self.num_classes, self.cls_last_dim),
            requires_grad=False)
        self.feature_used = nn.Parameter(
            torch.zeros(self.num_classes), requires_grad=False)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
            
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, x_cls
    
    def get_targets_for_stat(self,
                    pos_bboxes_list,
                    neg_bboxes_list,
                    pos_gt_bboxes_list,
                    pos_gt_labels_list,
                    rcnn_train_cfg,
                    concat=True):
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def online_feature_distribution_collect(self, embedding, labels):
        uniq_c = torch.unique(labels)
        for c in uniq_c:
            c = int(c)
            select_index = torch.nonzero(
                labels == c, as_tuple=False).squeeze(1)

            embedding_temp = embedding[select_index]

            self.per_cls_feature_distribution_collect(embedding_temp, c)
        return

    def per_cls_feature_distribution_collect(self, embedding, labels):
        mean = embedding.mean(dim=0)
        var = embedding.var(dim=0, unbiased=False)
        n = embedding.numel() / embedding.size(1)
        if n > 1:
            var = var * n / (n - 1)
        else:
            var = var
        if self.feature_used[labels] > 0:
            with torch.no_grad():
                self.feature_prototype[labels] = self.decay_ratio * mean + (
                        1 - self.decay_ratio) * self.feature_prototype[labels]
                self.feature_variance[labels] = self.decay_ratio * var + (
                        1 - self.decay_ratio) * self.feature_variance[labels]
        else:
            self.feature_prototype[labels] = mean
            self.feature_variance[labels] = var
            self.feature_used[labels] += 1

    def sampling_classes(self, sampled_num_classes, correct_cls_ratio):
        correct_cls_ratio_fg = correct_cls_ratio[:self.loss_cls.num_classes]
        correct_cls_ratio_revert = 1.0 - correct_cls_ratio_fg
        sample_probability = correct_cls_ratio_revert / correct_cls_ratio_revert.sum()

        randnum = np.sort(np.random.rand(sampled_num_classes))
        prob_point = 0
        randnum_index = 0
        select_classes = []
        for index, prob in enumerate(sample_probability):
            prob_point += prob
            if randnum[randnum_index] <= prob_point:
                while randnum[randnum_index] <= prob_point:
                    select_classes.append(index)
                    randnum_index += 1
                    if randnum_index >= len(randnum):
                        return torch.LongTensor(select_classes).cuda()

        # for sum(sample_probability) != 1 in case
        return torch.LongTensor(select_classes).cuda()

    def hallucinated_feature_generate(self):
        correct_cls_ratio = self.loss_cls.get_correct_cls_ratio()
        selected_classes = self.sampling_classes(self.sampled_num_classes, correct_cls_ratio)

        embedding_list = []
        label_list = []
        for label in selected_classes:
            label = int(label)
            if self.feature_used[label] == 0:
                continue
            for i in range(self.sampled_num_features):
                std = torch.sqrt(self.feature_variance[label])
                new_sample = self.feature_prototype[label] + std * torch.normal(
                    0, 1, size=std.shape).cuda()
                embedding_list.append(new_sample.unsqueeze(0))
                label_list.append(label)

        if len(embedding_list) != 0:
            embedding_list = torch.cat(embedding_list, 0)
            label_list = torch.tensor(label_list, device=embedding_list.device)
        return embedding_list, label_list

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             cls_embeds_for_stat,
             bbox_labels_for_stat,
             reduction_override=None):
        losses = dict()
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # not use bbox sampling
            # pos_inds[1024:] = False
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if self.flag:
            if len(bbox_labels_for_stat) > 0:
                self.online_feature_distribution_collect(cls_embeds_for_stat, bbox_labels_for_stat)

            hallucinated_embedding, hallucinated_labels = self.hallucinated_feature_generate()
            if len(hallucinated_embedding) > 0:
                hallucinated_cls_score = self.fc_cls(
                    hallucinated_embedding) if self.with_cls else None
                cls_score = torch.cat((cls_score, hallucinated_cls_score), dim=0)
                labels = torch.cat((labels, hallucinated_labels), dim=0)
                hallucinated_label_weights = torch.ones(
                    len(hallucinated_labels), device=hallucinated_embedding.device)
                label_weights = torch.cat((label_weights, hallucinated_label_weights), dim=0)

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_accuracy:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        return losses


@HEADS.register_module()
class FhmShared2FCBBoxHead(FhmConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(FhmShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class FhmShared4Conv1FCBBoxHead(FhmConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(FhmShared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
