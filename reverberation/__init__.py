"""
@Author: Conghao Wong
@Date: 2024-12-05 15:12:33
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-05 15:23:15
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import qpid

from .__args import ReverberationArgs
from .model import Reverberation, ReverberationModel

# Register new args and models
qpid.register(rev=[Reverberation, ReverberationModel])
