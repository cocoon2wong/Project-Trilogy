"""
@Author: Conghao Wong
@Date: 2024-12-12 10:02:19
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-13 15:08:08
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import Args
from qpid.model import layers, transformer

from .__args import ReverberationArgs as RevArgs
from .__layers import KernelLayer


class SelfReverberationLayer(torch.nn.Module):
    """
    Self-Reverberation Layer
    ---
    Forecast the *self-decided* future trajectories only according to the
    observed trajectories of ego agents themselves.
    Two reverberation kernels will be computed to weighted sum historical
    features to *wiring* past information into the future:

    - **Self-Generation kernel**: Weighted sum features in different styles to
      achieve the random/characterized/multi-style prediction goal;
    - **Self-reverberation kernel**: Evaluate how much contribution that each
      historical frame (step) has made when planning future trajectories
      on each specific future frame (step).
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
        # Transform layers
        Ttype, iTtype = layers.get_transform_layers(self.rev_args.T)
        self.Tlayer = Ttype((self.args.obs_frames, self.d_traj))
        self.iTlayer = iTtype((self.args.pred_frames, self.d_traj))

        # Shapes
        self.Tsteps_en, self.Tchannels_en = self.Tlayer.Tshape
        self.Tsteps_de, self.Tchannels_de = self.iTlayer.Tshape

        # Noise encoding
        self.ie = layers.TrajEncoding(self.d_noise, self.d_i, torch.nn.Tanh)

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

        # FC layers for computing reverberation kernels
        if not self.rev_args.lite:
            self.k1 = KernelLayer(self.d, self.d, self.rev_args.Kc)
            self.k2 = KernelLayer(self.d, self.d, self.Tsteps_de)

        else:
            self.k1 = layers.Dense(self.d, self.rev_args.Kc, torch.nn.Tanh)
            self.k2 = layers.Dense(self.d, self.Tsteps_de, torch.nn.Tanh)

        self.outer = layers.OuterLayer(self.Tsteps_en, self.Tsteps_en)
        self.decoder = layers.Dense(self.d, self.Tchannels_de)

    def forward(self, f_ego_diff: torch.Tensor,
                linear_fit: torch.Tensor,
                training=None, mask=None, *args, **kwargs):

        # Target values for queries
        traj_targets = self.Tlayer(linear_fit)

        # Trajectory features (ego)
        f = f_ego_diff

        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        for _ in range(repeats):
            # Assign random noise and embedding -> (batch, Tsteps, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f.shape[:-1]) + [self.d_noise])
            f_z = self.ie(z.to(f.device))

            # -> (batch, Tsteps, 2*d_i)
            f_final = torch.concat([f, f_z], dim=-1)

            # Transformer backbone -> (batch, Tsteps, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=traj_targets,
                               training=training)

            # Outer product -> (batch, d, Tsteps, Tsteps)
            f_tran_t = torch.transpose(f_tran, -1, -2)
            f_o = self.outer(f_tran_t, f_tran_t)

            # Compute reverberation kernels
            k1 = self.k1(f_tran)        # (batch, Tsteps, Kc)
            k2 = self.k2(f_tran)        # (batch, Tsteps, Tsteps_de)

            if self.rev_args.draw_kernels:
                from .utils import show_kernel

                show_kernel(k1, 'self_k1', 1, self.Tsteps_en, self.rev_args.Kc)
                show_kernel(k2, 'self_k2', 1, self.Tsteps_en, self.Tsteps_de)

            # Apply k1
            f1 = f_o @ k1[..., None, :, :]    # (batch, d, Tsteps, Kc)

            # Apply k2
            f2 = torch.transpose(f1, -1, -2) @ k2[..., None, :, :]

            # Decode predictions
            f2 = torch.permute(f2, [0, 2, 3, 1])        # (b, Kc, Tsteps_de, d)

            y = self.iTlayer(self.decoder(f2))          # (b, Kc, pred, dim)
            all_predictions.append(y)

        # Stack random output -> (batch, K, n_key, dim)
        return torch.concat(all_predictions, dim=-3)
