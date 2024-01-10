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

from data.fm.types import *


def create_feature2index(
        features: List[Union[SparseFeat, SparseFeatWithMultipleValues, DenseFeat]]
        ):
    feature2index = dict()
    cur_index = 0
    for feature in features:
        if isinstance(feature, SparseFeat):
            feature2index[feature.feature_name] = (cur_index, cur_index + 1)
            cur_index += 1
        elif isinstance(feature, SparseFeatWithMultipleValues):
            feature2index[feature.feature_name] = (cur_index, cur_index + feature.max_len)
            cur_index += feature.max_len
        elif isinstance(feature, DenseFeat):
            feature2index[feature.feature_name] = (cur_index, cur_index + feature.feature_dim)
            cur_index += feature.feature_dim
        else:
            raise TypeError()
        
    return feature2index


def _pad_sequence(sequence, max_len, pad_token_id):
    pad = np.ones((max_len - len(sequence)), dtype=int) * pad_token_id
    padded_sequence = np.concatenate([sequence, pad])
    return padded_sequence


def preprocess_data(
        data_path: str,
        threshold: float = 3.0,
        n_user: int = None
        ):
    # Paths
    movie_data_path = os.path.join(data_path, 'movies.csv')
    rating_data_path = os.path.join(data_path, 'ratings.csv')
    tag_data_path = os.path.join(data_path, 'tags.csv')

    # Load and Preprocess Movie data
    movie_data = pd.read_csv(movie_data_path, engine='python')

    # Load and Preprocess Tag data
    tag_data = pd.read_csv(tag_data_path, engine='python')
    tag_data['tag'] = tag_data['tag'].str.lower()
    tag_data = tag_data.groupby('movieId').agg({'tag': set})

    # Load and Preprocess Rating data
    rating_data = pd.read_csv(rating_data_path, engine='python')
    if n_user is not None:
        user_ids = sorted(rating_data['userId'].unique())[:n_user]
        rating_data = rating_data[rating_data['userId'] <= max(user_ids)]

    # Merge Data
    movie_data = movie_data.merge(tag_data, on='movieId', how='left')
    rating_data = rating_data.merge(movie_data, on='movieId', how='left')

    # Preprocess Text Data
    rating_data = rating_data.drop(['title', 'tag'], axis=1)

    #
    label_encoder = {}
    features = []
    #
    for f in ['userId', 'movieId', 'timestamp']:
        label_encoder[f] = LabelEncoder()
        label_encoder[f].fit(rating_data[f])
        rating_data[f] = label_encoder[f].transform(rating_data[f])
        features.append(SparseFeat(feature_name=f, vocab_size=len(label_encoder[f].classes_), embedding_dim=8))
    #
    for f in ['genres']:
        rating_data[f] = rating_data[f].str.split('|')
        unique_values = set([])
        max_len = 0
        for row in rating_data[f]:
            unique_values = unique_values.union(set(row))
            max_len = max(len(row), max_len)
        unique_values = ['<pad>'] + list(unique_values)
        label_encoder[f] = LabelEncoder()
        label_encoder[f].fit(unique_values)
        rating_data[f] = rating_data[f].apply(lambda x: _pad_sequence(label_encoder[f].transform(x), max_len, 0))
        features.append(SparseFeatWithMultipleValues(feature_name=f, vocab_size=len(label_encoder[f].classes_), embedding_dim=4, max_len=max_len, padding_idx=label_encoder[f].transform(['<pad>'])[0]))
    #
    rating_data['target'] = rating_data['rating'].apply(lambda x: 1. if x > threshold else 0.)

    data = rating_data
    feature2index = create_feature2index(features)

    wide_features = features
    deep_features = features
    
    return movie_data, data, label_encoder, wide_features, deep_features, feature2index


def train_test_split(
        rating_data: pd.DataFrame, 
        n_test: int = 5
        ):
    rating_data['ratingOrder'] = rating_data.\
        groupby('userId')['timestamp'].rank(method='first', ascending=False)

    target_train = rating_data[rating_data['ratingOrder'] > n_test]
    train = rating_data[rating_data['ratingOrder'] > n_test]
    
    test = rating_data[rating_data['ratingOrder'] <= n_test]
    
    train = train.drop('ratingOrder', axis=1)
    test = test.drop('ratingOrder', axis=1)
    
    return train, test