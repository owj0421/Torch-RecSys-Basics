import os
import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
from pandas import DataFrame

import torch
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from data.fm.preprocess import *
from data.fm.types import *

def get_dataset(data_path, threshold, n_user, n_test):
    movie_data, data, label_encoder, wide_features, deep_features, feature2index = preprocess_data(data_path, threshold, n_user)
    train, test = train_test_split(data, n_test)
    train_dataset, test_dataset = FMMovieLensDataset(train), FMMovieLensDataset(test)
    return movie_data, train_dataset, test_dataset, label_encoder, wide_features, deep_features, feature2index

class FMMovieLensDataset(Dataset):
    
    def __init__(
            self,
            data: DataFrame
            ):
        super().__init__()
        self.target = torch.from_numpy(data['target'].to_numpy()).unsqueeze(1)
        data = data.drop(['rating'], axis=1)
        self.data = []
        for f in ['userId', 'movieId', 'timestamp']:
            self.data.append(data[f].to_numpy()[:, np.newaxis])
        #
        for f in ['genres']:
            self.data.append(np.stack(data[f].to_numpy()))
        self.data = np.concatenate(self.data, axis=-1)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_ = self.data[index]

        return self.target[index], input_