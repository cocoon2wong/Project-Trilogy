"""
@Author: Conghao Wong
@Date: 2024-12-16 11:00:09
@LastEditors: Conghao Wong
@LastEditTime: 2025-06-11 10:15:49
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

from qpid.utils import ROOT_TEMP_DIR, dir_check

from .__layers import compute_inverse_kernel

TEMP_DIR = dir_check(os.path.join(ROOT_TEMP_DIR, 'Reverberation'))


def show_kernel(k: torch.Tensor | None,
                name: str,
                partitions: int,
                obs_periods: int,
                pred_periods: int):

    # Do nothing if `k` is not a Tensor
    if not isinstance(k, torch.Tensor):
        return

    # For `batch_size > 1` cases
    _k: np.ndarray = k.cpu().numpy()
    _k = np.mean(_k, axis=-3)   # (steps, new_steps)

    # Separate partitions
    # Final kernel shape: (batch, steps, new_steps)
    _k = np.reshape(_k, [obs_periods, partitions, pred_periods])

    # Display curves on each partition
    title = f'Kernel {name}'
    plt.close(title)
    fig = plt.figure(title)

    for _p in range(partitions):
        rows = int(np.ceil(partitions/4))
        cols = min(4, partitions)
        ax = fig.add_subplot(rows, cols, _p + 1)

        # Remove the linear part
        _matrix = _k[:, _p, :] ** 2      # (obs, pred)
        _matrix = _matrix / np.sum(_matrix, axis=0, keepdims=True)

        for _o in range(obs_periods):
            _y = _matrix[_o]
            _x = np.arange(len(_y))

            # Draw as reverberation curves
            _x_interp = np.linspace(_x[0], _x[-1], 100)
            _y_interp = make_interp_spline(_x, _y)(_x_interp)

            ax.plot(_x_interp, _y_interp, label=f'Step {_o}')
            ax.plot(_x, _y, 'x', color='black')

            # Save meta data (txt)
            _path = os.path.join(TEMP_DIR, f'meta_{title}_p{_p}_o{_o}.txt')
            np.savetxt(_path, _y)

        ax.set_ylim(np.min(_matrix) - 0.1, np.max(_matrix) + 0.1)
        ax.legend()

        if partitions > 1:
            ax.set_title(f'Partition {_p}')

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
