"""
@Author: Conghao Wong
@Date: 2024-12-05 15:17:31
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-05 15:19:47
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from qpid.model import Model, layers
from qpid.training import Structure


class ReverberationModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)


class Reverberation(Structure):
    MODEL_TYPE = ReverberationModel
