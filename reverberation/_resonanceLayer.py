"""
@Author: Conghao Wong
@Date: 2024-12-11 20:00:42
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-30 20:57:01
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.args import Args
from qpid.model import layers
from qpid.utils import get_mask

from .__args import ReverberationArgs as RevArgs


class ResonanceLayer(torch.nn.Module):
    """
    Resonance Layer
    ---
    Compute the spectral similarities of *pure* observed trajectories between
    the ego agent and all other neighboring agents. An angle-based pooling will
    be applied to gather these spectral similarities within limited angle-based
    partitions.
    *NOTE*: Arg `full_steps` may change how it behaves. It will compute the
    spectral similarities on each temporal step (with specific temporal
    resolutions) when this arg is enabled, otherwise on a flattened feature
    that contains all temporal-spectral information.
    """

    def __init__(self, Args: Args,
                 traj_dim: int,
                 hidden_feature_dim: int,
                 output_feature_dim: int,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        # Args
        self.args = Args
        self.rev_args = self.args.register_subargs(RevArgs, 'rev')

        # Settings
        self.partitions = self.rev_args.partitions
        self.d = output_feature_dim
        self.d_h = hidden_feature_dim
        self.d_traj = traj_dim

        # Layers
        # Transform layers
        Ttype, _ = layers.get_transform_layers(self.rev_args.T)
        self.Tlayer = Ttype((self.args.obs_frames, self.d_traj))

        # Shapes
        self.steps, self.channels = self.Tlayer.Oshape
        self.Tsteps, self.Tchannels = self.Tlayer.Tshape

        # Trajectory encoding (pure trajectories)
        self.te = layers.TrajEncoding(self.channels, hidden_feature_dim,
                                      activation=torch.nn.ReLU,
                                      transform_layer=self.Tlayer)

        # Position encoding (not positional encoding)
        self.ce = layers.TrajEncoding(2, self.d//2, torch.nn.ReLU)

        # Encoding layers (for resonance features)
        if self.rev_args.full_steps:
            self.fc1 = layers.Dense(self.d_h, self.d_h, torch.nn.ReLU)
        else:
            self.fc1 = layers.Dense(self.d_h*self.Tsteps,
                                    self.d_h,
                                    torch.nn.ReLU)
        self.fc2 = layers.Dense(self.d_h, self.d_h, torch.nn.ReLU)
        self.fc3 = layers.Dense(self.d_h, self.d//2, torch.nn.ReLU)

    def forward(self, x_ego_2d: torch.Tensor, x_nei_2d: torch.Tensor):

        # Move the last point of trajectories to 0
        x_ego_pure = (x_ego_2d - x_ego_2d[..., -1:, :])[..., None, :, :]
        x_nei_pure = x_nei_2d - x_nei_2d[..., -1:, :]

        # Embed trajectories (ego + neighbor) together and then split them
        f_pack = self.te(torch.concat([x_ego_pure, x_nei_pure], dim=-3))
        f_ego = f_pack[..., :1, :, :]
        f_nei = f_pack[..., 1:, :, :]

        # Compute meta resonance features (for each neighbor)
        # Shape of the final output `f_re`: (batch, N, (obs), d/2)
        # The `(obs)` or `(steps)` dimension only exists when the arg
        # `full_steps` is enabled
        f = f_ego * f_nei   # -> (batch, N, obs, d)

        if not self.rev_args.full_steps:
            f = torch.flatten(f, start_dim=-2, end_dim=-1)

        f_re = self.fc3(self.fc2(self.fc1(f)))

        # Compute positional information in a SocialCircle-like way
        if self.rev_args.full_steps:
            # Time-resolution of the used transform
            t_r = int(np.ceil(self.steps / self.Tsteps))

            # `x_nei_2d` is relative to ego's last observation step
            x_nei_real = x_nei_2d + x_ego_2d[..., None, -1:, :]
            p_nei = x_nei_real - x_ego_2d[..., None, :, :]
            p_nei = p_nei[..., ::t_r, :]
        else:
            p_nei = x_nei_2d[..., -1, :]

        # Compute distances and angles (for all neighbors)
        f_distance = torch.norm(p_nei, dim=-1)
        f_angle = torch.atan2(p_nei[..., 0], p_nei[..., 1])
        f_angle = f_angle % (2 * np.pi)

        # Partitioning
        partition_indices = f_angle / (2*np.pi/self.partitions)
        partition_indices = partition_indices.to(torch.int32)

        # Mask neighbors
        if self.rev_args.full_steps:
            # Remove egos from neighbors (the self-neighbors)
            valid_nei_mask = get_mask(torch.sum(p_nei, dim=-1), torch.int32)
            non_self_mask = (f_distance > 0.005).to(torch.int32)
            final_mask = valid_nei_mask * non_self_mask
            j = -1      # Axis bias
        else:
            final_mask = get_mask(torch.sum(x_nei_2d, dim=[-1, -2]),
                                  torch.int32)
            j = 0

        partition_indices = (partition_indices * final_mask +
                             -1 * (1 - final_mask))

        # Angle-based pooling
        pos_list: list[list[torch.Tensor]] = []
        re_list: list[torch.Tensor] = []
        for _p in range(self.partitions):
            _mask = (partition_indices == _p).to(torch.float32)
            _mask_count = torch.sum(_mask, dim=-1+j)

            n = _mask_count + 0.0001

            pos_list.append([])
            pos_list[-1].append(torch.sum(f_distance * _mask, dim=-1+j) / n)
            pos_list[-1].append(torch.sum(f_angle * _mask, dim=-1+j) / n)
            re_list.append(torch.sum(f_re * _mask[..., None], dim=-2+j) /
                           n[..., None])

        # Stack all partitions
        # (batch, (steps), partitions, 2)
        positions = torch.stack([torch.stack(i, dim=-1) for i in pos_list],
                                dim=-2)

        # (batch, (steps), partitions, d/2)
        re_partitions = torch.stack(re_list, dim=-2)

        # Encode circle components -> (batch, (steps), partition, d/2)
        f_pos = self.ce(positions)

        # Concat resonance features -> (batch, (steps), partition, d)
        re_matrix = torch.concat([re_partitions, f_pos], dim=-1)

        return re_matrix, f_re
