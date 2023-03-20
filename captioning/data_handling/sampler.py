#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


from typing import List
from scipy.stats import lognorm
import torch
import numpy as np
from torch.utils.data import Sampler, BatchSampler
from random import shuffle


class BySequenceLengthSampler(Sampler):
    """
    lengths : int
    audio lengths, seconds
    bucket_boundaries: tuple
    (bucket_left, bucket_right, num_buckets)
    """
    def __init__(self, lengths,
                 bucket_boundaries, batch_size=64, drop_last=True, seed=20):
        ind_n_len = []
        for i, length in enumerate(lengths):
            ind_n_len.append((i, length))
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = np.linspace(*bucket_boundaries)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            self.random_state.shuffle(data_buckets[k])
            if len(data_buckets[k]) % self.batch_size == 0:
                iter_list += (np.split(data_buckets[k], int(data_buckets[k].shape[0] / self.batch_size)))
            else:
                drop_num = len(data_buckets[k]) % self.batch_size
                iter_list += (np.array_split(data_buckets[k][:-drop_num]
                                                 , int(data_buckets[k][:-drop_num].shape[0] / self.batch_size)))
                if not self.drop_last:
                    iter_list += data_buckets[k][-drop_num:]
        self.random_state.shuffle(iter_list)  # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list:
            yield i.tolist()  # as it was stored in an array

    def __len__(self):
        return len(self.ind_n_len)

    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
            np.less_equal(buckets_min, seq_length),
            np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id


class BySequenceBatchSampler(BatchSampler):

    def __init__(self, sampler, batch_size, drop_last=True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self):
        batch_num = len(self.sampler) // self.batch_size
        return batch_num if self.drop_last else batch_num + 1

    def __iter__(self):
        for batch in self.sampler:
            yield batch
