"""
@Author: Conghao Wong
@Date: 2024-12-16 11:00:09
@LastEditors: Conghao Wong
@LastEditTime: 2025-04-23 16:40:46
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from qpid.utils import ROOT_TEMP_DIR, dir_check

from .__layers import compute_inverse_kernel

TEMP_DIR = dir_check(os.path.join(ROOT_TEMP_DIR, 'Reverberation'))


def show_kernel(k: torch.Tensor | None,
                name: str,
                partitions: int,
                obs_periods: int,
                pred_periods: int,
                normalize: int | bool = True):

    # Do nothing if `k` is not a Tensor
    if (isinstance(k, torch.nn.Module) or (k is None)):
        return

    # Kernel shape: (batch, steps, new_steps)
    _k: np.ndarray = k.cpu().numpy()
    _k = np.mean(_k, axis=-3)   # (steps, new_steps)

    # Normalize on EACH OUTPUT STEP
    if normalize:
        _min = np.min(_k, axis=-2, keepdims=True)
        _max = np.max(_k, axis=-2, keepdims=True)
        _k = (_k - _min)/(_max - _min)
    else:
        _k = _k ** 2

    _k = np.reshape(_k, [obs_periods, partitions, pred_periods])

    # Display kernels on each new step
    title = f'Kernel {name}'
    plt.close(title)

    fig = plt.figure(title)

    for _j in range(pred_periods):
        ax = fig.add_subplot(1, pred_periods, _j + 1)
        ax.imshow(_k[:, :, _j])
        ax.axis('off')

    plt.show()


def vis_kernels(R: torch.Tensor | None,
                G: torch.Tensor | None,
                name: str,
                setting: int = 1,
                partitions: int = 1):

    from .utils import show_kernel

    if (setting >= 1) and (isinstance(R, torch.Tensor)):
        [steps, T_f] = R.shape[-2:]
        T_h = steps // partitions
        show_kernel(R, f'{name}: Reverberation Kernel',
                    partitions, T_h, T_f)

    if (setting >= 2) and (isinstance(G, torch.Tensor)):
        T_h = G.shape[-2] // partitions
        K_g = G.shape[-1]
        show_kernel(G, f'{name}: Generating Kernel',
                    partitions, T_h, K_g)

    if setting >= 3:
        if isinstance(R, torch.Tensor):
            R_inv = compute_inverse_kernel(R)
            show_kernel(R_inv, f'{name}: Inverse Reverberation Kernel',
                        partitions, T_f, T_h)

        if isinstance(G, torch.Tensor):
            G_inv = compute_inverse_kernel(G)
            show_kernel(G_inv, f'{name}: Inverse Generating Kernel',
                        partitions, K_g, T_h)
