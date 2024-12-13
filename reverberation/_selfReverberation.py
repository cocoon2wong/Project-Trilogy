"""
@Author: Conghao Wong
@Date: 2024-12-12 10:02:19
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-13 16:37:58
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import Args
from qpid.model import layers, transformer

from .__args import ReverberationArgs as RevArgs


class SelfReverberationLayer(torch.nn.Module):
    """
    Self-Reverberation Layer
    ---
    """

    def __init__(self, Args: Args,
                 traj_dim: int,
                 input_feature_dim: int,
                 output_feature_dim: int,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        # Args
        self.args = Args
        self.rev_args = self.args.register_subargs(RevArgs, 'rev')

        # Settings
        self.d = output_feature_dim
        self.d_i = input_feature_dim
        self.d_traj = traj_dim
        self.d_noise = self.args.noise_depth

        # Layers
        Ttype, iTtype = layers.get_transform_layers(self.rev_args.T)
        self.Tlayer = Ttype((self.args.obs_frames, self.d_traj))
        self.iTlayer = iTtype((self.args.pred_frames, self.d_traj))

        # Noise encoding
        self.ie = layers.TrajEncoding(self.d_noise, self.d_i, torch.nn.Tanh)

        # Shapes
        self.Tsteps_en, self.Tchannels_en = self.Tlayer.Tshape
        self.Tsteps_de, self.Tchannels_de = self.iTlayer.Tshape

        # Transformer as the feature extractor
        self.T = transformer.Transformer(
            num_layers=4,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=self.Tchannels_en,
            target_vocab_size=self.Tchannels_en,
            pe_input=self.Tsteps_en,
            pe_target=self.Tsteps_en,
            include_top=False,
        )

        # FC layers for reverberation kernels
        self.k1 = layers.Dense(self.d, self.rev_args.Kc, torch.nn.Tanh)
        self.k2 = layers.Dense(self.d, self.Tsteps_de, torch.nn.Tanh)
        self.decoder = layers.Dense(self.d, self.Tchannels_de)

        self.outer = layers.OuterLayer(self.Tsteps_en, self.Tsteps_en)

    def forward(self, f_diff: torch.Tensor, linear_fit: torch.Tensor,
                training=None, mask=None, *args, **kwargs):

        traj_targets = self.Tlayer(linear_fit)

        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        for _ in range(repeats):
            # Assign random noise and embedding -> (batch, Tsteps, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f_diff.shape[:-1]) + [self.d_noise])
            f_z = self.ie(z.to(linear_fit.device))

            # -> (batch, Tsteps, 2*d_i)
            f = torch.concat([f_diff, f_z], dim=-1)

            # Transformer backbone -> (batch, Tsteps, d)
            f_tran, _ = self.T(inputs=f,
                               targets=traj_targets,
                               training=training)

            # Outer
            f_tran_t = torch.transpose(f_tran, -1, -2)  # (batch, d, Tsteps)
            # (batch, d, Tsteps, Tsteps)
            f_o = self.outer(f_tran_t, f_tran_t)

            # Kernels
            k1 = self.k1(f_tran)        # (batch, Tsteps, Kc)
            k2 = self.k2(f_tran)        # (batch, Tsteps, Tsteps_de)

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
