# from torch import nn as nn
from . import nn
from .process.inout import from_numpy
from torch import Tensor as array
import torch_mlu
from torch import add
from torch import sum
from torch import zeros
from torch import softmax
from torch import sqrt
from torch import transpose
from torch import matmul
from torch import div
# from torch import *
# cannot import *, otherwise raise TypeError: cannot assign 'torch.FloatTensor' as parameter 'weight' (torch.nn.Parameter or None expected) when call from_numpy
