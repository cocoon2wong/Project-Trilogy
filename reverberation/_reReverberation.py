"""
@Author: Conghao Wong
@Date: 2024-12-16 14:56:33
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-19 19:19:07
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import Args
from qpid.model import layers, transformer

from .__args import ReverberationArgs as RevArgs


class SocialReverberationLayer(torch.nn.Module):
    """
    Social Reverberation Layer
    ---
    TODO
    """

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
        self.partitions = self.rev_args.partitions

        # Layers
        # Transform layers
        Ttype, iTtype = layers.get_transform_layers(self.rev_args.T)
        self.Tlayer = Ttype((self.args.obs_frames, self.d_traj))
        self.iTlayer = iTtype((self.args.pred_frames, self.d_traj))

        # Shapes
        self.Tsteps_en, self.Tchannels_en = self.Tlayer.Tshape
        self.Tsteps_de, self.Tchannels_de = self.iTlayer.Tshape
        self.max_steps = self.Tsteps_en * self.partitions

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
            pe_input=self.max_steps,
            pe_target=self.max_steps,
            include_top=False,
        )

        # FC layers for computing reverberation kernels
        self.k1 = layers.Dense(self.d, self.rev_args.Kc, torch.nn.Tanh)
        self.k2 = layers.Dense(self.d, self.Tsteps_de, torch.nn.Tanh)
        self.outer = layers.OuterLayer(self.max_steps, self.max_steps)
        self.decoder = layers.Dense(self.d, self.Tchannels_de)

    def forward(self, x_ego_diff: torch.Tensor,
                f_ego_diff: torch.Tensor,
                re_matrix: torch.Tensor,
                training=None, mask=None, *args, **kwargs):

        # Stack partitions as additional temporal steps
        # (batch, Tsteps, p, d/2) -> (batch, Tsteps*p, d/2)
        f_re = torch.flatten(re_matrix, -3, -2)

        # Repeat obs on all partitions -> (batch, Tsteps*p, d/2)
        traj_targets = torch.repeat_interleave(self.Tlayer(x_ego_diff),
                                               self.partitions, dim=-2)
        f_ego = torch.repeat_interleave(f_ego_diff, self.partitions, dim=-2)

        # Partition-wise concat -> (batch, Tsteps*p, d/2)
        f = self.concat_fc(torch.concat([f_re, f_ego], dim=-1))

        all_predictions = []
        repeats = self.args.K_train if training else self.args.K
        for _ in range(repeats):
            # Assign random noise and embedding -> (batch, Tsteps*p, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f.shape[:-1]) + [self.d_noise])
            f_z = self.ie(z.to(f.device))

            # -> (batch, Tsteps, d)
            f_final = torch.concat([f, f_z], dim=-1)

            # Transformer backbone -> (batch, Tsteps*p, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=traj_targets,
                               training=training)

            # Outer product -> (batch, d, Tsteps*d, Tsteps*d)
            f_tran_t = torch.transpose(f_tran, -1, -2)
            f_o = self.outer(f_tran_t, f_tran_t)

            # Compute reverberation kernels
            k1 = self.k1(f_tran)        # (batch, Tsteps*d, Kc)
            k2 = self.k2(f_tran)        # (batch, Tsteps*d, Tsteps_de)

            if self.rev_args.draw_kernels:
                from .utils import show_kernel
                show_kernel(k1, 're_k1.png')
                show_kernel(k2, 're_k2.png')

            # Apply k1
            f1 = f_o @ k1[..., None, :, :]    # (batch, d, Tsteps, Kc)

            # Apply k2
            f2 = torch.transpose(f1, -1, -2) @ k2[..., None, :, :]

            # Decode predictions
            f2 = torch.permute(f2, [0, 2, 3, 1])        # (b, Kc, Tsteps_de, d)

            y = self.iTlayer(self.decoder(f2))          # (b, Kc, pred, dim)
            all_predictions.append(y)

        all_predictions = torch.concat(all_predictions, dim=-3)
        return all_predictions
