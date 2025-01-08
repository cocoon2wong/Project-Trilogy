"""
@Author: Conghao Wong
@Date: 2024-12-05 15:14:02
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-08 09:57:28
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class ReverberationArgs(EmptyArgs):

    @property
    def Kc(self) -> int:
        """
        The number of style channels when making predictions.
        """
        return self._arg('Kc', 20, argtype=STATIC,
                         desc_in_model_summary='Output channels')

    @property
    def T(self) -> str:
        """
        Transform type used to compute trajectory spectrums.

        It could be:
        - `none`: no transformations
        - `haar`: haar wavelet transform
        - `db2`: DB2 wavelet transform
        """
        return self._arg('T', 'haar', argtype=STATIC, short_name='T',
                         desc_in_model_summary='Transform type')

    @property
    def compute_linear_base(self) -> int:
        """
        Whether to learn to forecast the linear base during training.
        """
        return self._arg('compute_linear_base', 1, argtype=STATIC,
                         desc_in_model_summary='Train with linear base')

    @property
    def compute_self_bias(self) -> int:
        """
        Whether to learn to forecast the self-bias during training.
        """
        return self._arg('compute_self_bias', 1, argtype=STATIC,
                         desc_in_model_summary='Learn self-bias')

    @property
    def compute_re_bias(self) -> int:
        """
        Whether to learn to forecast the re-bias during training.
        """
        return self._arg('compute_re_bias', 1, argtype=STATIC,
                         desc_in_model_summary='Learn re-bias')

    @property
    def test_with_linear_base(self) -> int:
        """
        Whether to forecast the linear base when *testing* the model.
        """
        return self._arg('test_with_linear_base', 1, argtype=TEMPORARY)

    @property
    def test_with_self_bias(self) -> int:
        """
        Whether to forecast the self-bias when *testing* the model.
        """
        return self._arg('test_with_self_bias', 1, argtype=TEMPORARY)

    @property
    def test_with_re_bias(self) -> int:
        """
        Whether to forecast the re-bias when *testing* the model.
        """
        return self._arg('test_with_re_bias', 1, argtype=TEMPORARY)

    @property
    def partitions(self) -> int:
        """
        The number of partitions when computing the angle-based feature.
        """
        return self._arg('partitions', -1, argtype=STATIC,
                         desc_in_model_summary='Number of Angle-based Partitions')

    @property
    def draw_kernels(self) -> int:
        """
        Choose whether to draw and show visualized kernels when testing.
        It is typically used in the playground mode.
        """
        return self._arg('draw_kernels', 0, argtype=TEMPORARY)

    @property
    def full_steps(self) -> int:
        """
        Choose whether to compute all angle-based social partitions on all
        observation frames or just on a flattened feature.
        It may bring extra computation consumptions when this term is enabled.
        """
        return self._arg('full_steps', 0, argtype=STATIC,
                         desc_in_model_summary='Use full reverberation kernels')

    def _init_all_args(self):
        super()._init_all_args()

        if self.T == 'fft':
            self.log(f'Transform `{self.T}` is not supported!',
                     level='error', raiseError=ValueError)

        if self.partitions <= 0:
            self.log(f'Illegal partition settings ({self.partitions})! ' +
                     'Please add the arg `--partitions` to set the number of ' +
                     'angle-based partitions.',
                     level='error', raiseError=ValueError)
