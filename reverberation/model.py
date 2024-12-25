"""
@Author: Conghao Wong
@Date: 2024-12-05 15:17:31
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-25 19:23:20
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers
from qpid.training import Structure

from .__args import ReverberationArgs
from ._diffLayer import LinearDiffEncoding
from ._resonanceLayer import ResonanceLayer
from ._selfReverberation import SelfReverberationLayer
from ._socialReverberation import SocialReverberationLayer


class ReverberationModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        self.as_final_stage_model = True

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.args._set('output_pred_steps', 'all')
        self.rev_args = self.args.register_subargs(ReverberationArgs, 'rev')

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        # Transform layers
        t_type, it_type = layers.get_transform_layers(self.rev_args.T)
        self.tlayer = t_type((self.args.obs_frames, self.dim))
        self.itlayer = it_type((self.args.pred_frames, self.dim))

        # Linear difference encoding
        self.linear = LinearDiffEncoding(
            obs_frames=self.args.obs_frames,
            pred_frames=self.args.pred_frames,
            output_units=self.d//2,
            transform_layer=self.tlayer,
        )

        if self.rev_args.compute_self_bias:
            # Self-reverberation layer
            self.self_rev = SelfReverberationLayer(
                Args=self.args,
                traj_dim=self.dim,
                input_feature_dim=self.d//2,
                output_feature_dim=self.d,
            )

        if self.rev_args.compute_re_bias:
            # Resonance feature
            self.resonance = ResonanceLayer(
                Args=self.args,
                traj_dim=self.dim,
                hidden_feature_dim=self.d,
                output_feature_dim=self.d//2,
            )

            # Re-reverberation layer
            self.re_rev = SocialReverberationLayer(
                Args=self.args,
                traj_dim=self.dim,
                input_ego_feature_dim=self.d//2,
                input_re_feature_dim=self.d//2,
                output_feature_dim=self.d,
            )

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # Unpack inputs
        x_ego = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        x_nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Encode difference features (for ego agents)
        f_ego_diff, linear_fit, linear_base = self.linear(x_ego)
        x_ego_diff = x_ego - linear_fit

        # Compute self-reverberation-bias
        if self.rev_args.compute_self_bias and self.rev_args.test_with_self_bias:
            self_rev_bias = self.self_rev(f_ego_diff, x_ego_diff)
        else:
            self_rev_bias = 0

        # Compute re-reverberation-bias
        if self.rev_args.compute_re_bias and self.rev_args.test_with_re_bias:
            re_matrix, f_re = self.resonance(self.picker.get_center(x_ego)[..., :2],
                                             self.picker.get_center(x_nei)[..., :2])

            re_rev_bias = self.re_rev(x_ego_diff, f_ego_diff, re_matrix)
        else:
            re_rev_bias = 0

        if self.rev_args.compute_linear_base and self.rev_args.test_with_linear_base:
            y = linear_base[..., None, :, :]
        else:
            y = 0

        return y + self_rev_bias + re_rev_bias


class Reverberation(Structure):
    MODEL_TYPE = ReverberationModel
