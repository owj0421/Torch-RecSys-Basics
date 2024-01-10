import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data.fm.types import *


class FMLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1, keepdim=True) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term
    
