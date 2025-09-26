"""
@Author: Conghao Wong
@Date: 2025-09-19 10:08:08
@LastEditors: Conghao Wong
@LastEditTime: 2025-09-26 09:48:39
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES
from qpid.training.loss import BaseLossLayer

"""
NOTE: Classes defined in this file are used to conduct discussions in
Appendix. D ``Statistical Verifications''.
DO NOT use these metrics during training, or it MAY lead to worse results.
"""


class MeanADE(BaseLossLayer):
    """
    Average of ADE (for all generations).
    Predictions must have the same sequence-length as the groundtruth.

    NOTE: It only support 2D/3D coordinate prediction cases.
    """
    has_unit = True

    def _compute_all_ade(self, pred: torch.Tensor, label: torch.Tensor):
        """
        Compute all ADE, for each agent and each generated trajectory.
        Output shape: `(batch, K)`.
        """
        # Add the k-axis
        if pred.ndim == label.ndim:
            pred = pred[..., None, :, :]    # (..., K=1, pred, dim)

        all_ade = torch.mean(
            torch.norm(
                pred - label[..., None, :, :],
                p=2, dim=-1
            ), dim=-1
        )  # -> (..., K)

        return all_ade

    def forward(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor],
                mask=None, training=None, *args, **kwargs):

        # (..., (K), pred, dim)
        pred = outputs[0]

        # (..., pred, dim)
        label = self.model.get_label(labels, INPUT_TYPES.GROUNDTRUTH_TRAJ)

        all_ade = self._compute_all_ade(pred, label)
        return torch.mean(all_ade)


class StdADE(MeanADE):

    def forward(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor],
                mask=None, training=None, *args, **kwargs):

        # (..., (K), pred, dim)
        pred = outputs[0]

        # (..., pred, dim)
        label = self.model.get_label(labels, INPUT_TYPES.GROUNDTRUTH_TRAJ)

        all_ade = self._compute_all_ade(pred, label)
        all_std = torch.std(all_ade, dim=-1)
        return torch.mean(all_std)


class MeanFDE(MeanADE):

    def forward(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor],
                mask=None, training=None, *args, **kwargs):

        # (..., (K), pred, dim)
        pred = outputs[0]

        # (..., pred, dim)
        label = self.model.get_label(labels, INPUT_TYPES.GROUNDTRUTH_TRAJ)

        all_ade = self._compute_all_ade(pred[..., -1:, :], label[..., -1:, :])
        return torch.mean(all_ade)


class StdFDE(MeanADE):

    def forward(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor],
                mask=None, training=None, *args, **kwargs):

        # (..., (K), pred, dim)
        pred = outputs[0]

        # (..., pred, dim)
        label = self.model.get_label(labels, INPUT_TYPES.GROUNDTRUTH_TRAJ)

        all_ade = self._compute_all_ade(pred[..., -1:, :], label[..., -1:, :])
        all_std = torch.std(all_ade, dim=-1)
        return torch.mean(all_std)
