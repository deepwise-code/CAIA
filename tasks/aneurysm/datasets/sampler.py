import math
import random

import torch
from torch.utils.data import Sampler # 一般模式
from torch.utils.data import DistributedSampler as _DistributedSampler # 分布式

from utils.tools.logger import Logger as logger

IS_Distributed = False
BASE_SAMPLER = [Sampler, _DistributedSampler][IS_Distributed]

class DefaultSampler(BASE_SAMPLER):

    def __init__(self, dataset, num_replicas=None, shuffle=True, round_up=True, seed=42):
        self.dataset = dataset
        self.shuffle = shuffle
        self.round_up = round_up
        self.num_replicas = num_replicas if num_replicas is not None else 1
        if self.round_up:
            # 计算数据集的大小，使其可以被 num_replicas 整除
            self.total_size = math.ceil(len(self.dataset) / float(self.num_replicas)) * self.num_replicas
        else:
            self.total_size = len(self.dataset)

        self.seed = seed

        logger.info(f'init default dataloader sampler: {self.__class__.__name__}')
        logger.info(f'length of sid samples are: {len(self.dataset)} => {self.total_size}')

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        # 根据需要, 可能在原来索引清单基础上再复制一份在后面,之后在取前self.total_size
        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[:self.total_size]
        assert len(indices) == self.total_size

        # subsample: DataParallel不需要考虑每个GPU上数据划分问题
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        # if self.round_up:
        #     assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.total_size

class CTSampler(BASE_SAMPLER):
    """需要注意的是，一个epoch的iter数还是决定于dataset的长度；目前这个在dataset长度的数据被遍历完之前，sampler会被多次调用
    按<PID+STUDY_UID+SERIES_UID 三级路径>控制采样
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        shuffle=True,
        round_up=True,
        seed=42,
        num_per_ct=64,
        pos_fraction=0.5,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.shuffle = shuffle
        self.round_up = round_up
        data_infos = dataset.data_infos
        self.num_per_ct = num_per_ct
        self.pos_fraction = pos_fraction
        self.ct_dict = {}
        for idx, data_info in enumerate(data_infos):
            sub_dir = data_info['img_info']['filename'].split('/')[0]
            if sub_dir not in self.ct_dict.keys():
                self.ct_dict[sub_dir] = [(data_info['img_info']['filename'] + ' %d' % data_info['gt_label'], idx)]
            else:
                self.ct_dict[sub_dir].append((data_info['img_info']['filename'] + ' %d' % data_info['gt_label'], idx))

        if self.round_up:
            self.total_size = math.ceil(len(self.ct_dict) * num_per_ct / self.num_replicas) * self.num_replicas
        else:
            self.total_size = len(self.ct_dict) * num_per_ct

        self.seed = seed

        logger.info(f'origin dataset contain series(ct): {len(self.ct_dict)}, patch={len(self.dataset)}')
        logger.info(f'init customize dataloader sampler: {self.__class__.__name__}')
        logger.info(f'total patchs={self.total_size}, ct={len(self.ct_dict)}, num_per_ct={num_per_ct}, pos_fraction={pos_fraction:.2f}')

    def __iter__(self):
        # deterministically shuffle based on epoch
        self.pos_num = 0
        self.neg_num = 0
        self.lower = 0
        self.zero = 0

        indices = []
        ct_sub_dirs = list(self.ct_dict.keys())

        for sub_dir in ct_sub_dirs:
            pos_list, neg_list = self._split_lines(self.ct_dict[sub_dir])
            sample_list = self._sample_patch(pos_list, neg_list, self.num_per_ct, self.pos_fraction)
            indices.extend(self._get_index(sample_list))

        logger.info(f"pos sample num is: {self.pos_num}, neg sample num is: {self.neg_num}(approximately)")
        logger.info(f"ct with lower sampled pos num: {self.lower}, ct with zero pos num: {self.zero}")
        if self.round_up:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(indices)

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        # if self.round_up:
        #     assert len(indices) == math.ceil(len(self.ct_dict) * self.num_per_ct / self.num_replicas)

        return iter(indices)

    def _split_lines(self, line_idx_list):
        pos_list = []
        neg_list = []
        for line_idx in line_idx_list:
            label = int(line_idx[0].split(' ')[1])
            if label == 0:
                neg_list.append(line_idx)
            else:
                pos_list.append(line_idx)

        return pos_list, neg_list

    def _sample_patch(self, pos_list, neg_list, sample_num, pos_fraction=0.5):
        pos_per_image = int(round((sample_num * pos_fraction)))
        pos_per_this_image = min(pos_per_image, len(pos_list))
        # Without deterministic SEED
        sampled_pos = random.sample(pos_list, pos_per_this_image)

        if pos_per_this_image < sample_num * pos_fraction:
            self.lower += 1
        if pos_per_this_image == 0:
            self.zero += 1
        self.pos_num += pos_per_this_image
        neg_per_this_image = sample_num - pos_per_this_image
        neg_per_this_image = min(neg_per_this_image, len(neg_list))
        sampled_neg = random.sample(neg_list, neg_per_this_image)
        self.neg_num += neg_per_this_image

        sample_list = sampled_pos + sampled_neg
        random.shuffle(sample_list)

        while len(sample_list) < sample_num:
            end_idx = min((sample_num - len(sample_list)), len(sample_list))
            sample_list = sample_list + sample_list[:end_idx]
        assert len(sample_list) == sample_num

        return sample_list

    @staticmethod
    def _get_index(sample_list):
        return [line[1] for line in sample_list]

    def __len__(self):
        return self.total_size