import os
import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from data.fm.dataset import *

def build_dataset(
        data_path: str,
        type: str,
        threshold: float = 4.0,
        n_user: int = None, 
        n_test: int = 5
        ) -> Any:
    if type == 'fm':
        movie_data, train_dataset, test_dataset, label_encoder, wide_features, deep_features, feature2index = get_dataset(data_path, threshold, n_user, n_test)
        return {
            'movie_data' : movie_data,
            'train_dataset' : train_dataset,
            'test_dataset' : test_dataset,
            'label_encoder' : label_encoder,
            'wide_features' : wide_features,
            'deep_features' : deep_features,
            'feature2index' : feature2index
            }
    else:
        raise ValueError()

        
if __name__ == '__main__':
    builder_output = build_dataset('F:\Projects\datasets\ml-latest-small', type='fm')
    for i in DataLoader(builder_output['train_dataset'], batch_size=3):
        print(i)
        break
    # print(movie_data)
        