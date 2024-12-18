"""
@Author: Conghao Wong
@Date: 2024-12-11 20:00:42
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-19 19:25:56
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
    TODO
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
        Ttype, iTtype = layers.get_transform_layers(self.rev_args.T)
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

        self.fc1 = layers.Dense(self.d_h, self.d_h, torch.nn.ReLU)
        self.fc2 = layers.Dense(self.d_h, self.d//2, torch.nn.ReLU)

    def forward(self, x_ego_2d: torch.Tensor, x_nei_2d: torch.Tensor):

        # Move the last point of trajectories to 0
        x_ego_pure = (x_ego_2d - x_ego_2d[..., -1:, :])[..., None, :, :]
        x_nei_pure = x_nei_2d - x_nei_2d[..., -1:, :]

        # Embed trajectories (ego + neighbor) together and then split them
        f_pack = self.te(torch.concat([x_ego_pure, x_nei_pure], dim=-3))
        f_ego = f_pack[..., :1, :, :]
        f_nei = f_pack[..., 1:, :, :]

        # Compute resonance features (for each neighbor)
        f = f_ego * f_nei               # (batch, N, Tsteps, d)
        f_re = self.fc2(self.fc1(f))    # (batch, N, Tsteps, d/2)

        # Compute positional information in a SocialCircle-like way
        # `x_nei_2d` are relative values to target agents' last obs step
        # Compute distances and angles (for all neighbors)
        f_distance = torch.norm(x_nei_2d, dim=-1)       # (batch, N, obs)

        f_angle = torch.atan2(x_nei_2d[..., 0], x_nei_2d[..., 1])
        f_angle = f_angle % (2 * np.pi)                 # (batch, N, obs)

        # Split the observation period into obs//2 intervals
        f_distance = f_distance[..., ::2]
        f_angle = f_angle[..., ::2]

        # Partitioning
        partition_id = f_angle / (2*np.pi/self.partitions)
        partition_id = partition_id.to(torch.int32)

        # Mask empty neighbors
        interval = self.steps // self.Tsteps
        valid_mask = get_mask(torch.sum(x_nei_2d, dim=-1), torch.int32)
        valid_mask = valid_mask[..., ::interval]

        # Remove ego agents from neighbors
        non_self_mask = (torch.sum(x_nei_2d[..., -1, :], dim=-1) != 0)
        non_self_mask = non_self_mask.to(torch.int32)[..., None]

        valid_mask = valid_mask * non_self_mask
        partition_id = partition_id * valid_mask + -1 * (1 - valid_mask)

        positions = []
        re_partitions = []
        for _p in range(self.partitions):
            _mask = (partition_id == _p).to(torch.float32)
            _mask_count = torch.sum(_mask, dim=-2)

            n = _mask_count + 0.0001

            positions.append([])
            positions[-1].append(torch.sum(f_distance * _mask, dim=-2) / n)
            positions[-1].append(torch.sum(f_angle * _mask, dim=-2) / n)

            re_partitions.append(
                torch.sum(f_re * _mask[..., None], dim=-3) / n[..., None])

        # Stack all partitions
        # -> (batch, Tsteps, partitions, 2)
        positions = [torch.stack(i, dim=-1) for i in positions]
        positions = torch.stack(positions, dim=-2)

        # -> (batch, Tsteps, partitions, d/2)
        re_partitions = torch.stack(re_partitions, dim=-2)

        # Encode circle components -> (batch, Tsteps, partitions, d/2)
        f_pos = self.ce(positions)

        # Concat resonance features -> (batch, Tsteps, partitions, d)
        re_matrix = torch.concat([re_partitions, f_pos], dim=-1)

        return re_matrix, f_re
