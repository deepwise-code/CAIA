# -*- coding=utf-8 -*-
"""
    Author: wanghao
    date: 2019/10/26
    modify date: 2020/05/04
"""
from concurrent import futures
from functools import wraps

import numpy as np
import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    """recommend replace for default pytorch dataloder to speed up data load everywhere
    install command: pip install prefetch_generator
    refer:
    * https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#issuecomment-495090086
    * https://github.com/justheuristic/prefetch_generator.git
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=8)


class Data_Prefetcher(object):

    def __init__(self, loader, device=None, transform_func=None):
        '''accelerate data copy speed from RAM to VRAM '''
        self.loader = loader
        self.transform_func = transform_func
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        # self.device = torch.device("cuda:%d" % (torch.cuda.device_count() - 1,) if torch.cuda.is_available() else "cpu")
        self.batch = []
        self.stream = torch.cuda.Stream()
        self.preload_generator = self.preload()
        self.get_next_batch_data()

    def preload(self):
        for batch in self.loader:
            with torch.cuda.stream(self.stream):  # async copy data to gpu
                if self.transform_func is None:
                    self.batch = [item.to(self.device, non_blocking=True) for item in batch]
                else:
                    self.batch = self.transform_func(batch)
            yield

    def get_next_batch_data(self):
        try:
            next(self.preload_generator)
        except StopIteration:
            self.batch = None

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.get_next_batch_data()
        return batch

    next = __next__  # Python 2 compatibility

    def __len__(self):
        return len(self.loader)


