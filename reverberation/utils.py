"""
@Author: Conghao Wong
@Date: 2024-12-16 11:00:09
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-16 11:18:38
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os

import cv2
import torch
from matplotlib import pyplot as plt
from PIL import Image

from qpid.utils import ROOT_TEMP_DIR, dir_check

TEMP_DIR = dir_check(os.path.join(ROOT_TEMP_DIR, 'Reverberation'))
GRID_SIZE = 1


def show_kernel(k: torch.Tensor, file_name: str):
    _k = k.cpu().numpy()[0]
    _k = 255 * (_k - _k.min())/(_k.max() - _k.min())
    _k = cv2.resize(_k, (GRID_SIZE * _k.shape[1],
                         GRID_SIZE * _k.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(p := os.path.join(TEMP_DIR, file_name), _k)

    title = f'Kernel {file_name}'
    plt.close(title)
    plt.figure(title)
    plt.imshow(Image.open(p))
    plt.show()
