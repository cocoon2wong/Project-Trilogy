"""
@Author: Conghao Wong
@Date: 2024-12-30 14:51:43
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-30 20:20:30
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.args import Args
from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, process, transformer
from qpid.mods import segMaps
from qpid.mods.segMaps.settings import NORMALIZED_SIZE

from .__args import ReverberationArgs as RevArgs

INF = 1000000000
SAFE_THRESHOLDS = 0.05
MU = 0.00000001


class SceneReverberationLayer(torch.nn.Module):

    def __init__(self, Args: Args,
                 traj_dim: int,
                 input_ego_feature_dim: int,
                 input_re_feature_dim: int,
                 output_feature_dim: int,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        # Args
        self.args = Args
        self.rev_args = self.args.register_subargs(RevArgs, 'rev')

        # Settings
        self.d = output_feature_dim
        self.d_i_ego = input_ego_feature_dim
        self.d_i_re = input_re_feature_dim
        self.d_traj = traj_dim
        self.d_noise = self.args.noise_depth
        self.p = self.rev_args.partitions

        # Layers
        # Transform layers
        Ttype, iTtype = layers.get_transform_layers(self.rev_args.T)
        self.Tlayer = Ttype((self.args.obs_frames, self.d_traj))
        self.iTlayer = iTtype((self.args.pred_frames, self.d_traj))

        # Shapes
        self.Tsteps_en, self.Tchannels_en = self.Tlayer.Tshape
        self.Tsteps_de, self.Tchannels_de = self.iTlayer.Tshape
        self.time_res = self.args.obs_frames // self.Tsteps_en
        self.steps = self.Tsteps_en * self.p

        # Circle encoding
        self.ce = layers.TrajEncoding(3, self.d//2, torch.nn.ReLU)

        # Fusion layer (ego features and resonance features)
        self.concat_fc = layers.Dense(self.d_i_ego + self.d_i_re,
                                      self.d//2,
                                      torch.nn.ReLU)

        # Noise encoding
        self.ie = layers.TrajEncoding(self.d_noise, self.d//2, torch.nn.Tanh)

        # Transformer as the feature extractor
        self.T = transformer.Transformer(
            num_layers=2,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=self.Tchannels_en,
            target_vocab_size=self.Tchannels_en,
            pe_input=self.steps,
            pe_target=self.steps,
            include_top=False,
        )

        # FC layers for computing reverberation kernels
        self.k1 = layers.Dense(self.d, self.rev_args.Kc, torch.nn.Tanh)
        self.k2 = layers.Dense(self.d, self.Tsteps_de, torch.nn.Tanh)
        self.outer = layers.OuterLayer(self.steps, self.steps)
        self.decoder = layers.Dense(self.d, self.Tchannels_de)

        # Compute all pixels' indices
        pool_size = self.rev_args.scene_pool_size
        xs, ys = torch.meshgrid(torch.arange(NORMALIZED_SIZE//pool_size),
                                torch.arange(NORMALIZED_SIZE//pool_size),
                                indexing='ij')

        self.map_pos_pixel = torch.stack([xs.reshape([-1]),
                                          ys.reshape([-1])], dim=-1)
        self.map_pos_pixel = self.map_pos_pixel.to(torch.float32)
        self.map_pos_pixel = self.map_pos_pixel * pool_size + pool_size // 2

        if pool_size > 1:
            self.pool = torch.nn.MaxPool2d((pool_size, pool_size))
        else:
            self.pool = None

    def forward(self, model: Model, inputs: list[torch.Tensor],
                x_ego_diff: torch.Tensor,
                f_ego_diff: torch.Tensor,
                training=None, mask=None, *args, **kwargs):

        # Unpack inputs
        get = model.get_input

        # (batch, obs, dim)
        x_ego = get(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        # Segmentaion-map-related inputs (to compute the PhysicalCircle)
        # (batch, h, w)
        seg_maps = get(inputs, segMaps.INPUT_TYPES.SEG_MAP)

        # (batch, 4)
        seg_map_paras = get(inputs, segMaps.INPUT_TYPES.SEG_MAP_PARAS)

        # Process model inputs
        inputs_original = model.processor.inputs_original
        x_ego_original = get(inputs_original, INPUT_TYPES.OBSERVED_TRAJ)

        # Start computing the PhysicalCircle
        # PhysicalCircle will be computed on each agent's 2D center point
        x_ego_original_2d = model.picker.get_center(x_ego_original)[..., :2]

        # Time-resolution of the used transform
        x_ego_original_2d = x_ego_original_2d[..., ::self.time_res, :]

        # Pool the segmap to reduce computation load
        if self.pool:
            _seg_maps = self.pool(seg_maps[..., None, :, :])[..., 0, :, :]
        else:
            _seg_maps = seg_maps

        # Treat seg maps as a long sequence -> (batch, 1, a*a)
        _seg_maps = torch.flatten(_seg_maps, -2, -1)[..., None, :]

        # Compute velocity (moving length) during observation period
        moving_vector = x_ego[..., -1, :] - x_ego[..., 0, :]
        moving_length = torch.norm(moving_vector, dim=-1)   # (batch)

        # Compute pixel positions on seg maps
        W = seg_map_paras[..., :2][..., None, :]
        b = seg_map_paras[..., 2:4][..., None, :]

        # Compute angles and distances
        self.map_pos_pixel = self.map_pos_pixel.to(W.device)
        map_pos_real = (self.map_pos_pixel - b) / W

        # Compute distances and angles of all pixels
        direction_vectors = (map_pos_real[..., None, :, :] -
                             x_ego_original_2d[..., None, :])

        # (batch, steps, a*a)
        distances = torch.norm(direction_vectors, dim=-1)

        # Compute the `equivalent` distance
        equ_dis = (distances + MU) / (_seg_maps + MU)

        # (batch, steps, a*a)
        angles = torch.atan2(direction_vectors[..., 0],
                             direction_vectors[..., 1])     # (batch, a*a)

        # Partitioning
        angle_indices = (angles % (2*np.pi)) / (2*np.pi/self.p)
        angle_indices = angle_indices.to(torch.int32)

        map_safe_mask = (_seg_maps <= SAFE_THRESHOLDS).to(torch.float32)

        circle = []
        for ang in range(self.p):
            # Compute the partition's mask
            angle_mask = (angle_indices == ang).to(torch.float32)

            # Compute the minimum distance factor
            d = (0 * map_safe_mask +
                 (1 - map_safe_mask) * angle_mask * (equ_dis))

            # Find the non-zero minimum value
            zero_mask = (d == 0).to(torch.float32)
            d = (torch.ones_like(d) * zero_mask * INF +
                 d * (1 - zero_mask))
            min_d, _ = torch.min(d, dim=-1)

            # `d == INF` <=> there are no obstacles
            obstacle_mask = (min_d < INF).to(torch.float32)

            # The velocity factor
            if True:  # self.use_velocity:
                f_velocity = moving_length[..., None] * obstacle_mask
                circle.append(f_velocity)

            # The distance factor
            if True:  # self.use_distance:
                f_min_distance = min_d * obstacle_mask
                circle.append(f_min_distance)

            # The direction factor
            if True:  # self.use_direction:
                _angle = 2 * np.pi * (ang + 0.5) / self.p
                f_direction = _angle * obstacle_mask
                circle.append(f_direction)

        # Final shape: (batch, steps, partitions, d/2)
        circle = torch.stack(circle, dim=-1)
        circle = circle.reshape([-1, self.args.obs_frames//2, self.p, 3])

        # Rotate the circle (if needed)
        if (r_layer := model.processor.get_layer_by_type(process.Rotate)):
            circle = self.rotate(circle, r_layer.angles)

        f_circle = self.ce(circle)

        # Start forecasting
        # Pad features to keep the compatible tensor shape
        f_diff_pad = torch.repeat_interleave(f_ego_diff, self.p, -2)
        f_circle_pad = torch.flatten(f_circle, -3, -2)

        # Concat and fuse resonance matrices with trajectory features
        # -> (batch, steps*partitions, d)
        f_behavior = torch.concat([f_diff_pad, f_circle_pad], dim=-1)
        f_behavior = self.concat_fc(f_behavior)

        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        # Target value for queries
        traj_targets = self.Tlayer(x_ego_diff)
        traj_targets = torch.repeat_interleave(traj_targets, self.p, -2)

        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d)
            z = torch.normal(mean=0, std=1,
                             size=list(f_behavior.shape[:-1]) + [self.d_noise])
            re_f_z = self.ie(z.to(x_ego_diff.device))

            # (batch, steps, 2*d)
            re_f_final = torch.concat([f_behavior, re_f_z], dim=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.T(inputs=re_f_final,
                               targets=traj_targets,
                               training=training)

            # Outer product -> (batch, d, steps, steps)
            f_tran_t = torch.transpose(f_tran, -1, -2)
            f_o = self.outer(f_tran_t, f_tran_t)

            # Compute reverberation kernels
            k1 = self.k1(f_tran)        # (batch, steps, Kc)
            k2 = self.k2(f_tran)        # (batch, steps, Tsteps_de)

            if self.rev_args.draw_kernels:
                from .utils import show_kernel
                show_kernel(k1, 'scene_k1.png')
                show_kernel(k2, 'scene_k2.png')

            # Apply k1
            f1 = f_o @ k1[..., None, :, :]    # (batch, d, steps, Kc)

            # Apply k2
            f2 = torch.transpose(f1, -1, -2) @ k2[..., None, :, :]

            # Decode predictions
            f2 = torch.permute(f2, [0, 2, 3, 1])        # (b, Kc, Tsteps_de, d)

            y = self.iTlayer(self.decoder(f2))          # (b, Kc, pred, dim)
            all_predictions.append(y)

        all_predictions = torch.concat(all_predictions, dim=-3)
        return all_predictions

    def rotate(self, circle: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Rotate the physicalCircle. (Usually used after preprocess operations.)
        """
        # Rotate the circle <=> left or right shift the circle
        # Compute shift length
        angles = angles % (2*np.pi)
        partition_angle = (2*np.pi) / (self.partitions)
        move_length = (angles // partition_angle).to(torch.int32)

        # Remove paddings
        valid_circle = circle[..., :self.partitions, :]
        valid_circle = torch.concat([valid_circle, valid_circle], dim=-2)
        paddings = circle[..., self.partitions:, :]

        # Shift each circle
        rotated_circles = []
        for _circle, _move in zip(valid_circle, move_length):
            rotated_circles.append(_circle[_move:self.partitions+_move])

        rotated_circles = torch.stack(rotated_circles, dim=0)
        return torch.concat([rotated_circles, paddings], dim=-2)
