import os
import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame
            ):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.loc[index, self.data.columns != 'target'], self.data.loc[index, 'target']


def preprocess_data(
        data_path: str,
        threshold: float = 4.0,
        n_user: int = 1000
        ):
    # Paths
    movie_data_path = os.path.join(data_path, 'movies.csv')
    rating_data_path = os.path.join(data_path, 'ratings.csv')
    tag_data_path = os.path.join(data_path, 'tags.csv')

    # Load and Preprocess Movie data
    movie_data = pd.read_csv(movie_data_path, engine='python')
    movie_data['genres'] = movie_data['genres'].apply(lambda x: list(x.split('|')))
    print('[COMPLETE] Load Movie Data...')

    # Load and Preprocess Tag data
    tag_data = pd.read_csv(tag_data_path, engine='python')
    tag_data['tag'] = tag_data['tag'].str.lower()
    tag_data = tag_data.groupby('movieId').agg({'tag': list})
    print('[COMPLETE] Load Tag Data...')

    # Load and Preprocess Rating data
    rating_data = pd.read_csv(rating_data_path, engine='python')
    user_ids = sorted(rating_data['userId'].unique())[:n_user]
    rating_data = rating_data[rating_data['userId'] <= max(user_ids)]
    rating_data['target'] = rating_data['rating'].apply(lambda x: 1 if x > threshold else 0)
    print('[COMPLETE] Load Rating Data...')

    # Final Data
    movie_data = movie_data.merge(tag_data, on='movieId', how='left')
    rating_data = rating_data.merge(movie_data, on='movieId', how='left')

    return movie_data, rating_data


def train_test_split(rating_data: pd.DataFrame, n_test: int = 5):
    rating_data['ratingOrder'] = rating_data.\
        groupby('userId')['timestamp'].rank(method='first', ascending=False)

    train = rating_data[
        rating_data['ratingOrder'] > n_test
        ]
    
    test = rating_data[
        rating_data['ratingOrder'] <= n_test
        ]
    
    train = train.drop('ratingOrder', axis=1)
    test = test.drop('ratingOrder', axis=1)
    
    return train, test


def get_dataset(
        data_path: str,
        threshold: float = 4.0,
        n_user: int = 1000, 
        n_test: int = 5) -> Any:
    
    movie_data, rating_data = preprocess_data(data_path, threshold, n_user)
    print('[COMPLETE] Load and Preprocess Data')

    train_df, test_df = train_test_split(rating_data, n_test)
    print('[COMPLETE] Split Data')

    train = MovieLensDataset(train_df)
    test = MovieLensDataset(test_df)
    print('[COMPLETE] Build Dataset')

    return movie_data, train, test

        
if __name__ == '__main__':
    movie_data, train, test = get_dataset('F:\Projects\datasets\ml-latest-small')
    print(train[0])
        