"""
@Author: Conghao Wong
@Date: 2024-12-12 10:02:19
@LastEditors: Conghao Wong
@LastEditTime: 2025-04-15 20:05:05
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers, transformer

from .__layers import KernelLayer
from ._revberationTransfrom import (LinearMappingLayer,
                                    MultiStyleGeneratingLayer,
                                    ReverberationTransform)


class NonInteractivePrediction(torch.nn.Module):
    """
    Non-Interactive Prediction Layer
    ---
    Forecast the *self-decided* future trajectories only according to the
    observed trajectories of ego agents themselves.
    Two reverberation kernels will be computed to weighted sum historical
    features to *wiring* past information into the (rehearsal) future:

    - **Non-interactive-generating kernel**: Weighted sum features in different
      styles to achieve the random/characterized/multi-style prediction goal;
    - **Non-interactive-reverberation kernel**: Evaluate how much contribution
      hat each historical frame (step) has made when planning future
      trajectories on each specific future frame (step).
    """

    def __init__(self, input_feature_dim: int,
                 output_feature_dim: int,
                 noise_depth: int,
                 traj_generations: int,
                 transform_layer: layers.transfroms._BaseTransformLayer,
                 inverse_transform_layer: layers.transfroms._BaseTransformLayer,
                 enable_lite_mode: int | bool = False,
                 disable_G: bool = True,
                 disable_R: bool = True,
                 *args, **kwargs) -> None:

        super().__init__()

        # Variables and Settings
        self.d_i = input_feature_dim
        self.d = output_feature_dim
        self.d_noise = noise_depth

        self.K_g = traj_generations
        self.lite = enable_lite_mode

        self.disable_G = disable_G
        self.disable_R = disable_R

        # Layers
        # Transform layers
        self.Tlayer = transform_layer
        self.iTlayer = inverse_transform_layer

        # Shapes
        self.T_h, self.M_h = self.Tlayer.Tshape
        self.T_f, self.M_f = self.iTlayer.Tshape

        # Noise encoding
        self.ie = layers.TrajEncoding(self.d_noise, self.d_i, torch.nn.Tanh)

        # Transformer as the feature extractor
        self.T = transformer.Transformer(
            num_layers=4,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=self.M_h,
            target_vocab_size=self.M_h,
            pe_input=self.T_h,
            pe_target=self.T_h,
            include_top=False,
        )

        # FC layers for computing reverberation kernels
        if not self.lite:
            self.k1 = KernelLayer(self.d, self.d, self.K_g)
            self.k2 = KernelLayer(self.d, self.d, self.T_f)

        else:
            self.k1 = layers.Dense(self.d, self.K_g, torch.nn.Tanh)
            self.k2 = layers.Dense(self.d, self.T_f, torch.nn.Tanh)

        # Reverberation-transform-related layers
        self.rev = ReverberationTransform(
            historical_steps=self.T_h,
            future_steps=self.T_f,
        )

        if self.disable_G:
            # Use the MSK-like way to generate stochastic predictions
            # See "MSN: Multi-Style Network for Trajectory Prediction"
            self.G_layer = MultiStyleGeneratingLayer(
                feature_dim=self.d,
                style_channels=self.K_g,
            )

        if self.disable_R:
            # Forecast trajectories using direct FC layers
            self.R_layer = LinearMappingLayer(
                feature_dim=self.d,
                historical_steps=self.T_h,
                future_steps=self.T_f,
            )

        # Final output layer
        self.decoder = layers.Dense(self.d, self.M_f)

    def forward(self, f_ego_diff: torch.Tensor,
                linear_fit: torch.Tensor,
                repeats: int = 1,
                training=None, mask=None, *args, **kwargs):

        # Target values for queries
        traj_targets = self.Tlayer(linear_fit)

        # Trajectory features (ego)
        f = f_ego_diff

        all_predictions = []
        for _ in range(repeats):
            # Assign random noise and embedding -> (batch, T_h, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f.shape[:-1]) + [self.d_noise])
            f_z = self.ie(z.to(f.device))

            # -> (batch, T_h, 2*d_i)
            f_final = torch.concat([f, f_z], dim=-1)

            # Transformer backbone -> (batch, T_h, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=traj_targets,
                               training=training)

            # Reverberation kernels and transform
            G = self.k1(f_tran) if not self.disable_G else self.G_layer
            R = self.k2(f_tran) if not self.disable_R else self.R_layer
            f_rev = self.rev(f_tran, R, G)          # (batch, K_g, T_f, d)

            # Decode predictions
            y = self.decoder(f_rev)                 # (batch, K_g, T_f, M)
            y = self.iTlayer(y)                     # (batch, K_g, t_f, m)

            all_predictions.append(y)

        # Stack all outputs -> (batch, K, t_f, m)
        return torch.concat(all_predictions, dim=-3), G, R
