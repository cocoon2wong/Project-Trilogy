"""
@Author: Conghao Wong
@Date: 2024-12-05 15:14:02
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-16 17:11:36
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class ReverberationArgs(EmptyArgs):

    @property
    def Kc(self) -> int:
        """
        The number of style channels in `Agent` model.
        """
        return self._arg('Kc', 20, argtype=STATIC,
                         desc_in_model_summary='Output channels')

    @property
    def T(self) -> str:
        """
        Transformation type used to compute trajectory spectrums
        on the ego agents.

        It could be:
        - `none`: no transformations
        - `haar`: haar wavelet transform
        - `db2`: DB2 wavelet transform
        """
        return self._arg('T', 'haar', argtype=STATIC, short_name='T',
                         desc_in_model_summary='Transform type')

    @property
    def compute_linear_base(self) -> int:
        return self._arg('compute_linear_base', 1, argtype=STATIC,
                         desc_in_model_summary='Compute linear base')

    @property
    def compute_self_bias(self) -> int:
        return self._arg('compute_self_bias', 1, argtype=STATIC,
                         desc_in_model_summary='Compute self-bias')

    @property
    def compute_re_bias(self) -> int:
        return self._arg('compute_re_bias', 1, argtype=STATIC,
                         desc_in_model_summary='Compute re-bias')

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
