import torch
import torch.nn as nn
from data.fm.types import *
from model.fm.base.fn import *
from model.fm.base.layer import *


class FactorizationMachines(torch.nn.Module):

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
        self.attention_net = AttentionNet()

    def forward(self, x: Tensor):
        wide_sparse_inputs = get_sparse_inputs(x, self.wide_features, self.feature2index)
        deep_sparse_embedded_inputs = get_sparse_embedded_inputs(x, self.deep_features, self.feature2index, self.embedding_dict)
        
        x = self.linear(wide_sparse_inputs) + self.attention_net(deep_sparse_embedded_inputs)
        x = torch.sigmoid(x)
        return x


class AttentionNet(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1, keepdim=True) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term