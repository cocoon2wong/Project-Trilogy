"""
@Author: Conghao Wong
@Date: 2024-12-05 15:17:31
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-13 16:39:51
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


class ReverberationModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        self.as_final_stage_model = True

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
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
        self.linear = LinearDiffEncoding(obs_frames=self.args.obs_frames,
                                         pred_frames=self.args.pred_frames,
                                         output_units=self.d//2,
                                         transform_layer=self.tlayer)

        # Resonance feature
        self.resonance = ResonanceLayer(Args=self.args,
                                        traj_dim=self.dim,
                                        hidden_dim=self.d,
                                        feature_dim=self.d//2)

        self.self_rev = SelfReverberationLayer(Args=self.args,
                                               traj_dim=self.dim,
                                               input_feature_dim=self.d//2,
                                               output_feature_dim=self.d)

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # Unpack inputs
        x_ego = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        x_nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Encode difference features (for ego agents)
        f_diff, linear_fit, linear_base = self.linear(x_ego)

        # Compute resonance feature
        # re_matrix, f_re = self.resonance(self.picker.get_center(x_ego)[..., :2],
        #                                  self.picker.get_center(x_nei)[..., :2])

        self_rev = self.self_rev(f_diff, x_ego - linear_fit)

        return linear_base[..., None, :, :] + self_rev


class Reverberation(Structure):
    MODEL_TYPE = ReverberationModel
