import os
import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

@ dataclass
class SparseFeat:
    feature_name: Optional[str] = None
    vocab_size: Optional[int] = None
    embedding_dim: Optional[int] = None
    dtype: torch.dtype = torch.float32

@ dataclass
class SparseFeatWithMultipleValues():
    feature_name: Optional[str] = None
    vocab_size: Optional[int] = None
    embedding_dim: Optional[int] = None
    max_len: Optional[int] = None
    padding_idx: Optional[int] = 0
    combiner: Literal['mean', 'sum', 'max'] = 'mean'
    dtype: Optional[torch.dtype] = torch.float32

@ dataclass
class DenseFeat:
    feature_name: Optional[str] = None
    feature_dim: Optional[int] = None
    dtype: Optional[torch.dtype] = torch.float32

