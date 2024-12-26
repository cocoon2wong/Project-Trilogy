"""
@Author: Conghao Wong
@Date: 2024-12-16 11:00:09
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-26 21:03:45
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from qpid.utils import ROOT_TEMP_DIR, dir_check

TEMP_DIR = dir_check(os.path.join(ROOT_TEMP_DIR, 'Reverberation'))
GRID_SIZE = 1


def show_kernel(k: torch.Tensor, file_name: str, normalize: int | bool = False):
    _k = k.cpu().numpy()
    _k = np.mean(_k, axis=-3)

    # Normalize on EACH STEP
    if normalize:
        _min = np.min(_k, axis=-2, keepdims=True)
        _max = np.max(_k, axis=-2, keepdims=True)
        _k = (_k - _min)/(_max - _min)

    # Save as a image
    _k_save = 255 * (_k - _k.min())/(_k.max() - _k.min())
    _k_save = cv2.resize(_k_save, (GRID_SIZE * _k_save.shape[1],
                         GRID_SIZE * _k_save.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(TEMP_DIR, file_name), _k_save)

    title = f'Kernel {file_name}'
    plt.close(title)
    plt.figure(title)
    plt.imshow(_k)
    plt.colorbar()
    plt.show()
