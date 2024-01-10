import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data.fm.types import *
from model.fm.base.fn import *
from model.fm.base.layer import *

class WideandDeep(torch.nn.Module):

    def __init__(
            self, 
            wide_features: List[Union[SparseFeat, SparseFeatWithMultipleValues, DenseFeat]],
            deep_features: List[Union[SparseFeat, SparseFeatWithMultipleValues, DenseFeat]],
            feature2index: Dict[str, Tuple[int]],
            *args
            ):
        super().__init__()
        self.wide_features = wide_features
        self.deep_features = deep_features

        self.feature2index = feature2index

        self.embedding_dict = create_embedding_dict(wide_features)
        
        self.linear = nn.Linear(compute_linear_input_dim(wide_features), 1, bias=True)
        nn.init.normal_(self.linear.weight, mean=0, std=0.0001)

        in_features = compute_dnn_input_dim(deep_features)
        self.dnn = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Linear(in_features, in_features),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Linear(in_features, 1)
            )

    def forward(self, x: Tensor):
        wide_sparse_inputs = get_sparse_inputs(x, self.wide_features, self.feature2index)
        deep_sparse_embedded_inputs = get_sparse_embedded_inputs(x, self.deep_features, self.feature2index, self.embedding_dict)
        dense_inputs =get_dense_inputs(x, self.deep_features, self.feature2index)

        wide_inputs = wide_sparse_inputs
        deep_inputs = combine_inputs([deep_sparse_embedded_inputs, dense_inputs])

        x = self.linear(wide_inputs) + self.dnn(deep_inputs)
        x = torch.sigmoid(x)
        return x
