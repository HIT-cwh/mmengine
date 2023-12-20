# Copyright (c) OpenMMLab. All rights reserved.
import os

from datasets import concatenate_datasets, load_dataset, load_from_disk
from mmengine import print_log
from torch import distributed as dist
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import threading
import json
import mmap
import torch
from pathlib import Path
import numpy as np
import itertools as it
import operator
from copy import deepcopy
from torch.utils.data import ConcatDataset, DataLoader
from mmengine import ConfigDict


DATASET_TYPE_IDS_MAP = {"en": 0, "cn": 1, "code": 2}
def get_dataset_type_id(path):
    import re

    match_idxes = []
    for key, idx in DATASET_TYPE_IDS_MAP.items():
        if re.search(rf"/[z_]*{key}/", path):
            match_idxes.append(idx)
    assert len(match_idxes) == 1, f"{path}, match_idxes should be 1, but got {match_idxes} from {DATASET_TYPE_IDS_MAP}"
    return match_idxes[0]


class JsonlDataset(torch.utils.data.Dataset):
    """

    JSONL format is expected to roughly follow that of The Pile.
    One-line-per-document of the form:
    ```
    {
        "tokens": List[int],
    }
    ```

    Note that only the "tokens" key is used.
    """

    def __init__(self, path: str, dataset_type_id: int = 0, min_length=50):
        self.path = path
        self.threadlocal = threading.local()
        resolved_path = Path(path).resolve()
        self.resolved_path = resolved_path
        self.meta = Path(f"{resolved_path}.meta")
        self.type_id = dataset_type_id

        # only build the cache in on the primary worker to prevent overloading nfs
        assert os.path.exists(self.meta), f"The cache file:{self.meta} is not found for file:{self.path}"
        try:
            with open(self.meta, "rb") as f:
                meta = np.load(f)
        except Exception as e:
            print(f"Cannot load file {self.meta}...")
            raise e
        self.offsets = meta[:, 0]
        self.lengths = meta[:, -1]

        if min_length > 0:
            mask = self.lengths >= min_length
            self.old_lengths = self.lengths.copy()
            self.old_length = len(self.offsets)
            self.offsets = self.offsets[mask]
            self.lengths = self.lengths[mask]

    def __getitem__(self, idx):
        f = self._get_mmap()
        position = self.offsets[idx]
        f.seek(position)
        item = f.readline().decode("utf-8")
        try:
            item = json.loads(item)
            item["length"] = len(item["tokens"])  # add a length info
            item["type_id"] = self.type_id
        except Exception as err:
            raise json.decoder.JSONDecodeError(
                doc=self.path,
                pos=position,
                msg=(
                    f"Error while loading JSONL line in file {self.path} at byte "
                    f"{position}. Contents of line:\n{item}\n{err}"
                ),
            )
        return item

    def get_dataset_name(self):
        return str(self.resolved_path)

    def _get_mmap(self):
        if not hasattr(self.threadlocal, "handles"):
            with open(self.path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.threadlocal.handles = [f, mm]
                if self.path.endswith(".gz") or self.path.endswith(".bz") or self.path.endswith(".bz2"):
                    raise NotImplementedError(
                        "Compressed files are not supported because .seek() would require "
                        "rereading the entire file, making performance too slow."
                    )
        return self.threadlocal.handles[-1]

    def __setstate__(self, state):
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        d = {}
        for i, v in self.__dict__.items():
            if i != "threadlocal":
                d[i] = v
        return d

    def __del__(self):
        if hasattr(self.threadlocal, "handles"):
            # cleanup files we opened on initialization
            while self.threadlocal.handles:
                self.threadlocal.handles.pop().close()

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def __len__(self):
        # Virtual length of the dataset depends on the epoch number if the number of documents
        # is not perfectly divisible by the data_subshard_count
        return len(self.offsets)

DEFAULT_SEED = 1024
class PackedDataset(torch.utils.data.Dataset):
    """
    The class PackedDataset takes in a dataset and aggregates samples of different
    lengths together based on the packed_length.

    Args:
        dataset: The original dataset to pack.
        max_length_per_sample: The maximum length of each original sample. Default is 2048.
        packed_length: The length of each packed sample. Default is 4096.
    """

    def __init__(
        self,
        dataset,
        max_length_per_sample: int = 2048,
        packed_length: int = 4096,
    ):
        assert hasattr(dataset, "lengths")
        assert len(getattr(dataset, "lengths")) == len(
            dataset
        ), "The dataset must have lengths attribute and have the same length as the dataset"
        self.dataset = dataset
        self.max_length_per_sample = max_length_per_sample
        self.lengths = getattr(self.dataset, "lengths")
        self.packed_length = packed_length
        # Force a seed to be fixed to prevent problems caused by the seed not being restored when restarting

        self.seed = DEFAULT_SEED
        self.sample_indices, self.len_samples_shuffled, self.acm_len_samples = self.accu_sample_len(seed=self.seed)
        self.num_tokens = sum(self.lengths)

    def get_dataset_name(self):
        return self.dataset.get_dataset_name()

    def accu_sample_len(self, seed=None):
        """accumulative length of samples"""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(self.seed - 1)

        sample_indices = np.arange(len(self.lengths))
        rng.shuffle(sample_indices)
        len_samples_shuffled = list(map(self.lengths.__getitem__, sample_indices))  # shuffle后每条数据的长度
        acm_len_samples = list(it.accumulate(len_samples_shuffled, operator.add))   # shuffle后的累积长度
        return sample_indices, len_samples_shuffled, acm_len_samples

    def __len__(self):
        # Line 405 of document_to_sequence.py in metaseq is directly spliced,
        # without additional consideration of sos or eos
        n_packs = self.num_tokens // self.packed_length
        return n_packs

    def cal_map(self, carriage_idx: int = 0):
        assert carriage_idx >= 0
        length_train = (carriage_idx + 1) * self.packed_length
        post_pos = np.searchsorted(self.acm_len_samples, length_train, side="left")
        return post_pos

    def mapping(self, pack_idx: int = 0):
        # pack_idx is zero-based
        pre_pos, pre_token_id = 0, 0
        if pack_idx > 0:
            pre_pos = self.cal_map(pack_idx - 1)  # 前一条packed数据结束的位置在第几条数据中
            pre_token_id = self.len_samples_shuffled[pre_pos] - (
                self.acm_len_samples[pre_pos] - (pack_idx) * self.packed_length
            )  # 前一条packed数据结束的位置是那条数据的第几个token
            if pre_token_id == self.len_samples_shuffled[pre_pos]:
                pre_pos += 1
                pre_token_id = 0

        pos = self.cal_map(pack_idx)
        token_id = self.len_samples_shuffled[pos] - (self.acm_len_samples[pos] - (pack_idx + 1) * self.packed_length)
        return pre_pos, pre_token_id, pos, token_id

    def build_pack(self, pre_pos: int, pre_token_id: int, pos: int, token_id: int):
        pack, cu_seqlens, indexes, labels, type_ids = [], [0], [], [], []

        while pre_pos < pos:
            sample_idx = self.sample_indices[pre_pos]
            sample = self.dataset[sample_idx]
            chunk = sample["tokens"][pre_token_id:]
            pack.extend(chunk)
            _labels = deepcopy(chunk)
            _labels = list(_labels[1:]) + [-100]
            assert len(_labels) == len(chunk), (_labels, chunk)
            labels.extend(_labels)
            type_ids.extend([sample.get("type_id", 0)] * len(chunk))
            num_new_samples, tokens_left = divmod(len(chunk), self.max_length_per_sample)
            for _ in range(num_new_samples):
                cu_seqlens.append(cu_seqlens[-1] + self.max_length_per_sample)
                indexes.extend(list(range(self.max_length_per_sample)))
            if tokens_left > 0:
                cu_seqlens.append(cu_seqlens[-1] + tokens_left)
                indexes.extend(list(range(tokens_left)))
            pre_pos = pre_pos + 1
            pre_token_id = 0

        sample_idx = self.sample_indices[pos]
        sample = self.dataset[sample_idx]
        chunk = sample["tokens"][pre_token_id:token_id]  # fragement of a sample
        pack.extend(chunk)
        _labels = deepcopy(chunk)
        if token_id == len(sample["tokens"]):
            _labels = list(_labels[1:]) + [-100]
        else:
            if token_id > len(sample["tokens"]):
                print(f"token_id {token_id}, len of sample {len(sample['tokens'])}")
            _labels = list(_labels[1:]) + [sample["tokens"][token_id]]
        assert len(_labels) == len(chunk), (_labels, chunk)
        labels.extend(_labels)
        type_ids.extend([sample.get("type_id", 0)] * len(chunk))
        num_new_samples, tokens_left = divmod(len(chunk), self.max_length_per_sample)
        for _ in range(num_new_samples):
            cu_seqlens.append(cu_seqlens[-1] + self.max_length_per_sample)
            indexes.extend(list(range(self.max_length_per_sample)))
        if tokens_left > 0:
            cu_seqlens.append(cu_seqlens[-1] + tokens_left)
            indexes.extend(list(range(tokens_left)))

        out = {"tokens": pack, "cu_seqlens": cu_seqlens, "indexes": indexes, "labels": labels, "type_ids": type_ids}
        return out

    def cal_pos_unpack(self, index):
        micro_bsz = 1
        if index == 0:
            pre_pos = 0
        else:
            pre_pos = index * micro_bsz

        pos = (index + 1) * micro_bsz
        return pre_pos, pos

    def build_unpack(self, index):

        pre_pos, pos = self.cal_pos_unpack(index)

        pack, cu_seqlens, indexes, labels, type_ids = [], [0], [], [], []

        while pre_pos < pos and pre_pos < len(self.dataset):
            sample_idx = self.sample_indices[pre_pos]
            sample = self.dataset[sample_idx]
            length = min(len(sample["tokens"]), self.max_length_per_sample)
            chunk = sample["tokens"][0:length]
            pack.extend(chunk)
            _labels = deepcopy(chunk)
            _labels = list(_labels[1:]) + [-100]
            assert len(_labels) == len(chunk), (_labels, chunk)
            labels.extend(_labels)
            type_ids.extend([sample.get("type_id", 0)] * len(chunk))
            cu_seqlens.append(cu_seqlens[-1] + len(chunk))
            indexes.extend(list(range(length)))
            pre_pos = pre_pos + 1

        if cu_seqlens[-1] != self.packed_length:
            pack = pack + [0] * (self.packed_length - cu_seqlens[-1])
            labels = labels + [0] * (self.packed_length - cu_seqlens[-1])
            type_ids = type_ids + [0] * (self.packed_length - cu_seqlens[-1])
            indexes.extend(list(range(self.packed_length - cu_seqlens[-1])))
            cu_seqlens.append(self.packed_length)

        assert len(pack) == self.packed_length

        out = {"tokens": pack, "cu_seqlens": cu_seqlens, "indexes": indexes, "labels": labels, "type_ids": type_ids}
        return out

    def __getitem__(self, item: int):
        """Given the index, it returns a dict as
        {
         'tokens': List[int],
         'cu_seqlens': List[int],
         'indexes': List[int], # denotes positional vector as 'tokens'
         'labels': List[int], # corresponds to 'tokens' and shifted yet, -100 means skipping prediction
        }
        """
        pos_before, token_id_before, pos_after, token_id_after = self.mapping(item)
        return self.build_pack(pos_before, token_id_before, pos_after, token_id_after)

def get_packed_dataset_without_short_length(
    folder,
    max_length_per_sample=2048,
    packed_length=4096,
    show_progress=False,
    min_length=50,
    min_length_dict=None,
    pack_into_one_sample=False,
):
    assert os.path.exists(folder), f"{folder} does not exist."
    datasets = []
    delete_samples = 0

    for root, dirs, files in os.walk(folder, followlinks=True):
        dirs.sort()  # Let the folder need to be returned in a fixed order
        num_token_in_folder = 0

        for fn in tqdm(sorted(files), total=len(files), leave=False, disable=not show_progress):
            if fn.endswith(".bin"):
                fp = os.path.join(root, fn)
                catch_ml_keys = []
                min_length_num = min_length
                if min_length_dict is not None:
                    for k, v in min_length_dict.items():
                        if k in fp:
                            min_length_num = v
                            catch_ml_keys.append(k)
                    assert (
                        len(catch_ml_keys) < 2
                    ), f"The file name `{fp}` matched the following resample keys:{catch_ml_keys}"

                ds_type_id = get_dataset_type_id(path=fp)
                ds = JsonlDataset(fp, ds_type_id, min_length=min_length_num)

                if hasattr(ds, "old_length"):
                    delete_samples += ds.old_length - len(ds)
                if len(ds) == 0:
                    continue

                ds = PackedDataset(ds, max_length_per_sample, packed_length)

                num_token_in_folder += len(ds) * packed_length
                datasets.append(ds)

    dataset = ConcatDataset(datasets=datasets)

    return dataset

class StaticBatchSampler:
    def __init__(
        self,
        datasets,
        batch_size=192,
        rampup_batch_size="6 2 8",
        micro_bsz=2,
        seed=0,
        drop_last=True,
        data_rank=0,
        data_world_size=1,
    ):
        assert drop_last is True, "Currently only support drop last"
        if rampup_batch_size:
            # In the process increase to batch_size
            start_bsz, bsz_incre, incre_every = map(int, rampup_batch_size.split())
        else:
            start_bsz, bsz_incre, incre_every = batch_size, batch_size, 1
        self.raw_rampup_batch_size = rampup_batch_size
        self.start_bsz = start_bsz
        self.bsz_incre = bsz_incre
        self.incre_every = incre_every

        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.batch_count = 0
        self.micro_bsz = micro_bsz
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.num_consumed_samples_in_epoch = 0
        self.datasets = datasets
        self.num_samples = sum([len(ds) for ds in datasets])

        self.get_indices()  # get data
        self.sampler = None

    def get_indices(self, old_indices=None):
        if old_indices is not None:
            assert (
                len(old_indices) <= self.num_samples
            ), f"The checkpoint has {len(old_indices)} samples, \
while the new restart use less samples ({self.num_samples})"

        else:
            old_indices = np.array([])

        # indices includes len(old_indices) but not self.num_samples
        indices = np.arange(len(old_indices), self.num_samples)
        self.rng_state = self.rng.get_state()
        self.rng.shuffle(indices)
        # Need to consider drop_last
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: \
{rampup_samples*self.data_world_size} Vs. self.num_samples: {self.num_samples}"

            num_samples = (self.num_samples - rampup_samples * self.data_world_size) // (
                self.batch_size * self.data_world_size
            )
            num_samples = num_samples * self.batch_size * self.data_world_size + rampup_samples * self.data_world_size
        else:
            num_samples = self.num_samples // (self.batch_size * self.data_world_size)
            num_samples = num_samples * self.batch_size * self.data_world_size
        indices = np.concatenate([old_indices, indices]).astype(int)  # It needs to be spliced with the previous
        indices = indices[:num_samples]
        self.indices = indices
        assert len(self.indices) >= self.batch_size, "The number of samples should be larger than batch_size"
        self.num_consumed_samples_in_epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng = np.random.RandomState(self.seed + self.epoch)

    def __len__(self):
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: {rampup_samples*self.data_world_size} \
Vs. self.num_samples: {self.num_samples}"

            num_batches = (self.num_samples - rampup_samples * self.data_world_size) // self.batch_size
            num_batches = num_batches // self.data_world_size + self.incre_every * ramp_steps
        else:
            num_batches = self.num_samples // self.batch_size // self.data_world_size

        return num_batches

    def __iter__(self):
        indices = self.indices[self.data_rank :: self.data_world_size]
        while self.num_consumed_samples_in_epoch < len(indices):
            batch_rampup_idx = self.batch_count // self.incre_every
            cur_batch_size = batch_rampup_idx * self.bsz_incre + self.start_bsz
            cur_batch_size = min(cur_batch_size, self.batch_size)
            batch = indices[self.num_consumed_samples_in_epoch : self.num_consumed_samples_in_epoch + cur_batch_size]
            yield batch
            self.num_consumed_samples_in_epoch += len(batch)  # Consider multiple processes.
            self.batch_count += 1
        self.get_indices()  # get a new round

    def state_dict(self):
        states = {
            "batch_size": self.batch_size,
            "raw_rampup_batch_size": self.raw_rampup_batch_size,
            "rng_state": self.rng_state,
            "epoch": self.epoch,
            "seed": self.seed,
            "data_world_size": self.data_world_size,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "batch_count": self.batch_count,  # The batch_count here is due to the existence of multiple processes,
            # the batch may be oversent, and it needs to be overwritten by the external batch_count
            "indices": self.indices,  # The sequence used to breakpoint retraining is the same as before
        }

        return states

    def load_state_dict(self, states):
        for name in ("data_world_size", "raw_rampup_batch_size", "seed"):  # 'batch_size'
            assert states[name] == getattr(self, name), (name, states[name], getattr(self, name))  # should not change
        self.rng.set_state(states["rng_state"])
        self.get_indices(old_indices=None)  # Regenerate indices based on random state
        self.epoch = states["epoch"]
        self.batch_count = states["batch_count"]
        self.num_consumed_samples_in_epoch = states["num_consumed_samples_in_epoch"]

    def copy(self):
        copy_sampler = StaticBatchSampler(
            self.datasets,
            self.batch_size,
            self.raw_rampup_batch_size,
            self.micro_bsz,
            self.seed,
            drop_last=True,
            data_rank=self.data_rank,
            data_world_size=self.data_world_size,
        )

        copy_sampler.load_state_dict(self.state_dict())
        return copy_sampler

def packed_collate_fn(batch, packed_length):

    xs, ys, cu_seqlens, indexes, ts = [], [], [], [], []
    for b in batch:
        assert (
            len(b["tokens"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['tokens'])} and {packed_length})"
        assert (
            len(b["labels"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['labels'])} and {packed_length})"
        assert (
            len(b["type_ids"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['type_ids'])} and {packed_length})"

        tokens = [abs(w) for w in b["tokens"]]
        labels = [w if w > 0 else -100 for w in b["labels"]]

        xs.append(torch.LongTensor(tokens))
        # The labels have been shifted here, so they are aligned with the output corresponding to the token
        ys.append(torch.LongTensor(labels))
        ts.append(torch.LongTensor(b["type_ids"]))
        cu_seqlens.append(torch.IntTensor(b["cu_seqlens"]))
        indexes.append(torch.LongTensor(b["indexes"]))

    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    ts = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
    indexes = torch.stack(indexes, dim=0)
    # if len(set(map(len, cu_seqlens))) == 1:  # if has uniform length, then stack to save device transfer time
    #     cu_seqlens = torch.stack(cu_seqlens, dim=0)

    assert xs.shape[1] == packed_length, (xs.shape[1], packed_length)
    assert len(batch) == 4
    max_seqlen = [(cu_seqlens[i][1:] - cu_seqlens[i][:-1]).max().item() for i in range(4)]
    data_dict = {"input_ids": xs, "cumulative_len": cu_seqlens, "indexes": indexes, "labels": ys, "max_seqlen": max_seqlen}

    return {'data': data_dict, 'data_samples': None}

def get_train_data_loader(
    num_worker: int = 0, dataset_generate_func = None, train_sampler=None, train_collate_fn=None
):
    # Get the dataset types
    dataset_types = None
    dataset_types = list(DATASET_TYPE_IDS_MAP.keys())
    data_cfg = ConfigDict(
        train_folder='/mnt/petrelfs/share_data/gaojianfei/maibao_kaoshi_7_5_v0213_8k_rc8_nowm/train',
        packed_length=8192,
        seq_len=8192,
        min_length=0,
        pack_sample_into_one=False,
        micro_num=4,
        rampup_batch_size="",
        micro_bsz=1,
    )

    # Get the sample weight dictionary
    train_folder = data_cfg.train_folder

    train_ds = get_packed_dataset_without_short_length(
        folder=data_cfg.train_folder,
        packed_length=data_cfg.packed_length,
        max_length_per_sample=data_cfg.seq_len,
        show_progress=dist.get_rank() == 0,
        min_length=data_cfg.min_length,
        min_length_dict=data_cfg.get("min_length_dict", {}),
        pack_into_one_sample=data_cfg.pack_sample_into_one,
    )

    if dataset_generate_func is None or not train_folder:
        # partition already completed
        assert isinstance(train_ds, (PackedDataset, ConcatDataset))
        # Create the training dataset sampler
        train_sampler = StaticBatchSampler(
            train_ds.datasets if isinstance(train_ds, ConcatDataset) else [train_ds],
            batch_size=data_cfg.micro_num,
            rampup_batch_size=data_cfg.rampup_batch_size,
            micro_bsz=data_cfg.micro_bsz,
            seed=1024,
            drop_last=True,
            data_rank=dist.get_rank(),
            data_world_size=dist.get_world_size(),
        )

    train_collate_fn = partial(packed_collate_fn, packed_length=data_cfg.packed_length)

    # Create the training data loader
    train_dl = DataLoader(
        # shuffle=False,
        dataset=train_ds,
        batch_sampler=train_sampler,
        num_workers=num_worker,
        pin_memory=True,
        collate_fn=train_collate_fn,
        persistent_workers=num_worker > 0,
    )

    return train_dl
