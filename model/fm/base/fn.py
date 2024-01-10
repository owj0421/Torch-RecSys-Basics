import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data.fm.types import *


# 모델 학습 및 참조 시 필요한 함수들

def combine_inputs(inputs: List[Optional[Tensor]]):
    combined_inputs = list(filter(lambda x: x is not None, inputs))
    combined_inputs = torch.cat(combined_inputs, dim=-1)
    return combined_inputs


def _multi_hot_encoder(labels, vocab_size, padding_idx, dtype=torch.float32):
    multi_hot_labels = torch.zeros(size=(len(labels), vocab_size), dtype=dtype, device=labels.get_device())
    for i, label in enumerate(labels):
        multi_hot_labels[i] = multi_hot_labels[i].scatter(dim=0, index=label, value=1.)
    return multi_hot_labels.to_sparse()


def get_sparse_inputs(
        x: Dict[str, List[Union[int, float, list, Tensor, np.ndarray]]],
        features: List[Union[SparseFeat, SparseFeatWithMultipleValues, DenseFeat]],
        feature2index: Dict[str, Tuple[int]],
        ):
    sparse_features = list(filter(lambda x: isinstance(x, SparseFeat), features))
    sparse_features_with_multiple_values = list(filter(lambda x: isinstance(x, SparseFeatWithMultipleValues), features))

    sparse_inputs = [
        _multi_hot_encoder(x[:, feature2index[feature.feature_name][0]:feature2index[feature.feature_name][1]], vocab_size=feature.vocab_size,
                           padding_idx=feature.padding_idx if isinstance(feature, SparseFeatWithMultipleValues) else None)
        for feature in sparse_features + sparse_features_with_multiple_values
        ]
    
    return torch.cat(sparse_inputs, dim=-1) if sparse_inputs else None


def get_sparse_embedded_inputs(
        x: Dict[str, List[Union[int, float, list, Tensor, np.ndarray]]],
        features: List[Union[SparseFeat, SparseFeatWithMultipleValues, DenseFeat]],
        feature2index: Dict[str, Tuple[int]],
        embedding_dict: nn.ModuleDict
        ):
    sparse_features = list(filter(lambda x: isinstance(x, SparseFeat), features))
    sparse_features_with_multiple_values = list(filter(lambda x: isinstance(x, SparseFeatWithMultipleValues), features))

    sparse_embedded_inputs = [
        embedding_dict[feature.feature_name]\
            (x[:, feature2index[feature.feature_name][0]:feature2index[feature.feature_name][1]])
        for feature in sparse_features + sparse_features_with_multiple_values
        ]
    
    return torch.cat(sparse_embedded_inputs, dim=-1) if sparse_embedded_inputs else None


def get_dense_inputs(
        x: Dict[str, List[Union[int, float, list, Tensor, np.ndarray]]],
        features: List[Union[SparseFeat, SparseFeatWithMultipleValues, DenseFeat]],
        feature2index: Dict[str, Tuple[int]],
        ):
    dense_features = list(filter(lambda x: isinstance(x, DenseFeat), features))

    dense_inputs = [
        x[:, feature2index[feature.feature_name][0]:feature2index[feature.feature_name][1]] 
        for feature in dense_features
        ]

    return torch.cat(dense_inputs, dim=-1) if dense_inputs else None


# 모델 생성 시 필요한 함수들

def create_embedding_dict(
        features: List[Union[SparseFeat, SparseFeatWithMultipleValues, DenseFeat]],
        init_std=0.0001
        ):

    sparse_features = list(filter(lambda x: isinstance(x, SparseFeat), features))
    sparse_features_with_multiple_values = list(filter(lambda x: isinstance(x, SparseFeatWithMultipleValues), features))

    embedding_dict = {}

    for feature in sparse_features:
        embedding_dict[feature.feature_name] = nn.EmbeddingBag(
            feature.vocab_size, feature.embedding_dim, dtype=feature.dtype)
        
    for feature in sparse_features_with_multiple_values:
        embedding_dict[feature.feature_name] = nn.EmbeddingBag(
            feature.vocab_size, feature.embedding_dim, padding_idx=feature.padding_idx, mode=feature.combiner, dtype=feature.dtype)
            
    embedding_dict = nn.ModuleDict(embedding_dict)
    
    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)
        # Do not use Xavier

    return embedding_dict


def compute_linear_input_dim(
        wide_features: List[Union[SparseFeat, SparseFeatWithMultipleValues, DenseFeat]]
        ):
    dim = 0
    for feature in wide_features:
        if isinstance(feature, SparseFeat):
            dim += feature.vocab_size
        elif isinstance(feature, SparseFeatWithMultipleValues):
            dim += feature.vocab_size
        elif isinstance(feature, DenseFeat):
            dim += feature.feature_dim
        else:
            raise TypeError()
        
    return dim


def compute_dnn_input_dim(
        dnn_features: List[Union[SparseFeat, SparseFeatWithMultipleValues, DenseFeat]]
        ):
    dim = 0
    for feature in dnn_features:
        if isinstance(feature, SparseFeat):
            dim += feature.embedding_dim
        elif isinstance(feature, SparseFeatWithMultipleValues):
            dim += feature.embedding_dim
        elif isinstance(feature, DenseFeat):
            dim += feature.feature_dim
        else:
            raise TypeError()
        
    return dim