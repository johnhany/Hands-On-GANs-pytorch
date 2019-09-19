import glob
import random
import os

import numpy as np
import scipy.ndimage as nd
import scipy.io as io
import torch

from torch.utils.data import Dataset
from PIL import Image


def getVoxelFromMat(path, cube_len=64):
    """Mat 데이터로 부터 Voxel 을 가져오는 함수"""
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels


class ShapeNetDataset(Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader

    Based on https://github.com/rimchang/3DGAN-Pytorch
    """

    def __init__(self, root, cube_len):
        self.root = root
        self.listdir = os.listdir(self.root)
        self.cube_len = cube_len

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.listdir[index]), "rb") as f:
            volume = np.asarray(getVoxelFromMat(f, self.cube_len), dtype=np.float32)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)
