import os
import random
import numpy as np
import pandas as pd
from typing import Tuple, Literal
import torch
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder



class pycolor:
    BLACK = '\033[30my'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def read_base_data(file_path: str) -> Tuple[pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(file_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(file_path, 'test.csv'))
    return train_df, test_df


def use_low_memory(df: pd.DataFrame, float_32: bool = True) -> pd.DataFrame:
    numerics_type = ['int16', 'int32', 'int64', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2 
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics_type:
            if str(col_type)[:3] == 'int':
                col_min, col_max = df[col].min(), df[col].max()
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif float_32:
                df[col] = df[col].astype(np.float32)

    # end_mem = df.memory_usage().sum() / 1024 ** 2
    # print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df


def scaling_data(train_df: pd.DataFrame, test_df: pd.DataFrame, scaler_type: Literal['minmax', 'standard']) -> Tuple[pd.DataFrame]:
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        
    apply_cols = [col for col in train_df.columns if col not in ['stock_id', 'time_id', 'target']]
    train_df[apply_cols] = scaler.fit_transform(train_df[apply_cols])
    test_df[apply_cols] = scaler.transform(test_df[apply_cols])
    
    encoder = LabelEncoder()
    train_df['stock_id'] = encoder.fit_transform(train_df['stock_id'].values.reshape(-1, 1))
    test_df['stock_id'] = encoder.transform(test_df['stock_id'].values.reshape(-1, 1))
    
    return train_df, test_df


def submit_pred(test_df: pd.DataFrame, pred: np.ndarray) -> None:
    submit_df = pd.DataFrame()
    submit_df['row_id'] = test_df['stock_id'].astype(str) + '-' + test_df['time_id'].astype(str)
    submit_df['target'] = pred
    submit_df.to_csv('../output/submission.csv', index=False)