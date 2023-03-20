#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


"""
Dynamic Batch Sampler from speechbrain
https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/dataio/sampler.html#DynamicBatchSampler
"""
from typing import List
from scipy.stats import lognorm
import torch
import numpy as np
from torch.utils.data import Sampler, BatchSampler
from random import shuffle


class DynamicBatchSampler(Sampler):
    """This BatchSampler batches examples together by grouping them by their length.

    Arguments
    ---------
    dataset : torch.utils.data.Dataset
        PyTorch Dataset from which elements will be sampled.
    max_batch_length : int
        Upper limit for the sum of the length of examples in a batch.
        Should be chosen based on your GPU memory.
    num_buckets : int
        Number of discrete buckets used to group examples together.
        If num_buckets == 1, all examples can be batched together. As the number of buckets grows only examples with similar
        length can be grouped together. This trades-off speed with randomization.
        Low number -> better randomization, High number -> faster training.
        However if set too high the training speed will decrease. If num_buckets -> number of examples in the dataset the batch size
        will be small impacting training speed and possibly performance.
        NOTE: you have either to specify manually the bucket_boundaries or the number of buckets.
    shuffle : bool
        Whether or not shuffle examples between each epoch.
    batch_ordering : string
        If ``random``, batches are randomly permuted; otherwise ``ascending`` or ``descending`` sorted by length.
    max_batch_ex: int
        If set, it limits the maximum number of examples that can be in a batch superseeding max_batch_length
        in instances where the amount of examples will exceeed the value specified here.
        E.g. you have a lot of short examples and the batch size for those will be too high, you can use this argument
        to limit the batch size for these short examples.
    lengths_list: list
        Overrides length_func by passing a list containing the length of each example
        in the dataset. This argument must be set when the dataset is a plain
        Pytorch Dataset object and not a DynamicItemDataset object as length_func
        cannot be used on Pytorch Datasets.
    epoch : int
        The epoch to start at.
    drop_last : bool
         If ``True``, the sampler will drop the last examples which
         have not been grouped.
    """

    def __init__(self,
                 dataset,
                 max_batch_length,
                 num_buckets,
                 shuffle,
                 batch_ordering,
                 max_batch_ex,
                 lengths_list,
                 epoch,
                 seed,
                 drop_last,
                 ):

        self._dataset = dataset
        self._ex_lengths = {}
        ex_ids = self._dataset.data_ids

        for indx in range(len(lengths_list)):
            self._ex_lengths[str(indx)] = lengths_list[indx]

        self._bucket_boundaries = np.array(
            self._get_boundaries_through_warping(
                max_batch_length=max_batch_length,
                num_quantiles=num_buckets
            )
        )

        self._max_batch_length = max_batch_length
        self._shuffle_ex = shuffle
        self._batch_ordering = batch_ordering
        self._seed = seed
        self._drop_last = drop_last
        self._max_batch_ex = max_batch_ex

        self._bucket_lens = [
            max(1, int(max_batch_length / self._bucket_boundaries[i]))
            for i in range(len(self._bucket_boundaries))
        ] + [1]

        self._epoch = epoch
        self._generate_batches()

    def _get_boundaries_through_warping(
            self, max_batch_length: int, num_quantiles: int,
    ) -> List[int]:
        # NOTE: the following lines do not cover that there is only one example in the dataset
        # warp frames (duration) distribution of train data
        print("Batch quantisation in latent space")
        # linspace set-up
        num_boundaries = num_quantiles + 1
        # create latent linearly equal spaced buckets
        latent_boundaries = np.linspace(
            1 / num_boundaries, num_quantiles / num_boundaries, num_quantiles,
        )
        # get quantiles using lognormal distribution
        quantiles = lognorm.ppf(latent_boundaries, 1)
        # scale up to to max_batch_length
        bucket_boundaries = quantiles * max_batch_length / quantiles[-1]
        # compute resulting bucket length multipliers
        length_multipliers = [
            bucket_boundaries[x + 1] / bucket_boundaries[x]
            for x in range(num_quantiles - 1)
        ]
        # logging
        print(
            "Latent bucket boundary - buckets: {} - length multipliers: {}".format(
                list(map("{:.2f}".format, bucket_boundaries)),
                list(map("{:.2f}".format, length_multipliers)),
            )
        )
        return list(sorted(bucket_boundaries))

    def _permute_batches(self):

        if self._batch_ordering == "random":
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampler = torch.randperm(
                len(self._batches), generator=g
            ).tolist()  # type: ignore
            tmp = []
            for idx in sampler:
                tmp.append(self._batches[idx])
            self._batches = tmp

        elif self._batch_ordering == "ascending":
            self._batches = sorted(
                self._batches,
                key=lambda x: max([self._ex_lengths[str(idx)] for idx in x]),
            )
        elif self._batch_ordering == "descending":
            self._batches = sorted(
                self._batches,
                key=lambda x: max([self._ex_lengths[str(idx)] for idx in x]),
                reverse=True,
            )
        else:
            raise NotImplementedError

    def _generate_batches(self):
        print("DynamicBatchSampler: Generating dynamic batches")
        if self._shuffle_ex:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampler = torch.randperm(len(self._dataset), generator=g).tolist()  # type: ignore
        else:
            # take examples as they are: e.g. they have been sorted
            sampler = range(len(self._dataset))  # type: ignore

        self._batches = []
        bucket_batches = [[] for i in self._bucket_lens]

        stats_tracker = [
            {"min": np.inf, "max": -np.inf, "tot": 0, "n_ex": 0}
            for i in self._bucket_lens
        ]

        for idx in sampler:
            # length of pre-sampled audio
            item_len = self._ex_lengths[str(idx)]
            # bucket to fill up most padding
            bucket_id = np.searchsorted(self._bucket_boundaries, item_len)
            # fill audio's duration into that bucket
            bucket_batches[bucket_id].append(idx)

            stats_tracker[bucket_id]["min"] = min(
                stats_tracker[bucket_id]["min"], item_len
            )
            stats_tracker[bucket_id]["max"] = max(
                stats_tracker[bucket_id]["max"], item_len
            )
            stats_tracker[bucket_id]["tot"] += item_len
            stats_tracker[bucket_id]["n_ex"] += 1
            # track #samples - why not duration/#frames; rounded up?
            # keep track of durations, if necessary

            if (
                    len(bucket_batches[bucket_id]) >= self._bucket_lens[bucket_id]
                    or len(bucket_batches[bucket_id]) >= self._max_batch_ex
            ):
                self._batches.append(bucket_batches[bucket_id])
                bucket_batches[bucket_id] = []
                # keep track of durations

        # Dump remaining batches
        if not self._drop_last:
            for batch in bucket_batches:
                if batch:
                    self._batches.append(batch)

        self._permute_batches()  # possibly reorder batches

        if self._epoch == 0:  # only log at first epoch
            # frames per batch & their padding remaining
            boundaries = [0] + self._bucket_boundaries.tolist()

            for bucket_indx in range(len(self._bucket_boundaries)):
                try:
                    num_batches = stats_tracker[bucket_indx]["tot"] // (
                        self._max_batch_length
                    )
                    pad_factor = (
                                         stats_tracker[bucket_indx]["max"]
                                         - stats_tracker[bucket_indx]["min"]
                                 ) / (
                                         stats_tracker[bucket_indx]["tot"]
                                         / stats_tracker[bucket_indx]["n_ex"]
                                 )
                except ZeroDivisionError:
                    num_batches = 0
                    pad_factor = 0

                print(
                    (
                            "DynamicBatchSampler: Bucket {} with boundary {:.1f}-{:.1f} and "
                            + "batch_size {}: Num Examples {:.1f}, Num Full Batches {:.3f}, Pad Factor {:.3f}."
                    ).format(
                        bucket_indx,
                        boundaries[bucket_indx],
                        boundaries[bucket_indx + 1],
                        self._bucket_lens[bucket_indx],
                        stats_tracker[bucket_indx]["n_ex"],
                        num_batches,
                        pad_factor * 100,
                    )
                )

            if self.verbose:
                batch_stats = {
                    "tot_frames": [],
                    "tot_pad_frames": [],
                    "pad_%": [],
                }
                for batch in self._batches:
                    tot_frames = sum(
                        [self._ex_lengths[str(idx)] for idx in batch]
                    )
                    batch_stats["tot_frames"].append(tot_frames)
                    max_frames = max(
                        [self._ex_lengths[str(idx)] for idx in batch]
                    )
                    tot_pad = sum(
                        [
                            max_frames - self._ex_lengths[str(idx)]
                            for idx in batch
                        ]
                    )
                    batch_stats["tot_pad_frames"].append(tot_pad)
                    batch_stats["pad_%"].append(tot_pad / tot_frames * 100)

                padding_details = "Batch {} with {:.1f} frames with {} files - {:.1f} padding, {:.2f} (%) of total."
                padding_details = "DynamicBatchSampler: " + padding_details
                for i in range(len(self._batches)):
                    print(
                        padding_details.format(
                            i,
                            batch_stats["tot_frames"][i],
                            len(self._batches[i]),
                            batch_stats["tot_pad_frames"][i],
                            batch_stats["pad_%"][i],
                        )
                    )

    def __iter__(self):
        for batch in self._batches:
            yield batch
        if self._shuffle_ex:  # re-generate examples if ex_ordering == "random"
            self._generate_batches()
        if self._batch_ordering == "random":
            # we randomly permute the batches only --> faster
            self._permute_batches()

    def set_epoch(self, epoch):
        """
        You can also just access self.epoch, but we maintain this interface
        to mirror torch.utils.data.distributed.DistributedSampler
        """
        self._epoch = epoch
        self._generate_batches()

    def __len__(self):
        return len(self._batches)


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
