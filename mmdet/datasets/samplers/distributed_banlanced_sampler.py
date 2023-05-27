import math

import torch
from collections import defaultdict
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedBalancedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, num_samples=5, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.CLASSES = dataset.CLASSES
        self.num_samples = num_samples
        self.total_size = num_samples * num_replicas
        self.category_image_indices = defaultdict(list)
        self._get_category_image_indices(dataset)

    def __iter__(self):
        indices_all_classes = []
        for cat_id in self.category_image_indices:
            indices = self.category_image_indices[cat_id]

            # add extra samples to make it evenly divisible
            # in case that indices is shorter than half of total_size
            indices = (indices *
                       math.ceil(self.total_size / len(indices)))[:self.total_size]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples

            indices_all_classes += indices

        return iter(indices_all_classes)

    def _get_category_image_indices(self, dataset):
        num_images = len(dataset) # lvisv1: 99388
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0:
                cat_ids = set([len(self.CLASSES)])
            for cat_id in cat_ids:
                self.category_image_indices[cat_id].append(idx)

    def __len__(self):
        return self.num_samples * len(self.category_image_indices)
