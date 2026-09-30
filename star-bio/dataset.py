"""Lazy, process-local HDF5 access for DataLoader workers."""
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectraDataset(Dataset):
    def __init__(self, cache, indices, train=False, batch_shift=False, augmentation_strength=1.0):
        self.cache = str(cache)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = train
        self.batch_shift = batch_shift
        self.augmentation_strength = float(augmentation_strength)
        self.h = None
        self.pid = None

    def __len__(self):
        return len(self.indices)

    def close(self):
        if self.h is not None:
            self.h.close()
            self.h = None
        self.pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(h=None, pid=None)
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, index):
        if self.h is None or self.pid != os.getpid():
            self.close()
            self.h = h5py.File(self.cache, "r")
            self.pid = os.getpid()
        source_index = self.indices[index]
        x = torch.from_numpy(self.h["x"][source_index].copy())
        y, batch = int(self.h["y"][source_index]), int(self.h["batch"][source_index])
        if self.train:
            if torch.rand(()) < 0.5:
                x = x + torch.randn_like(x) * 0.008
            if torch.rand(()) < 0.3:
                x = x * torch.empty(1).uniform_(0.95, 1.05)
            if torch.rand(()) < 0.2:
                width = int(torch.randint(8, min(64, x.shape[-1]) + 1, (1,)))
                start = int(torch.randint(0, x.shape[-1] - width + 1, (1,)))
                x[:, start:start + width] = 0
            if self.batch_shift and torch.rand(()) < 0.7:
                # Simulate batch-to-batch gain and baseline drift while
                # keeping the derivative/SNV channels consistent.
                t = torch.linspace(-1, 1, x.shape[-1], dtype=x.dtype)
                gain = torch.empty(1).uniform_(0.94, 1.06) ** self.augmentation_strength
                offset = torch.empty(1).uniform_(-0.04, 0.04) * self.augmentation_strength
                slope = torch.empty(1).uniform_(-0.025, 0.025) * self.augmentation_strength
                x[0] = x[0] * gain + offset + slope * t
                x[1] = x[1] * gain + slope
                x[2] = (x[0] - x[0].mean()) / (x[0].std() + 1e-6)
        return x, torch.tensor(y), torch.tensor(batch)
