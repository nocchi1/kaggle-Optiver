import numpy as np
import pandas as pd
import os
import re
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Literal, Tuple
from sklearn.cluster import KMeans

from utils import use_low_memory



def fe_book_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df['spread_ask1_bid1'] = df['ask_price1'] / df['bid_price1']
    df['spread_ask2_bid2'] = df['ask_price2'] / df['bid_price2']
    df['spread_ask1_ask2'] = df['ask_price2'] / df['ask_price1']
    df['spread_bid1_bid2'] = df['bid_price1'] / df['bid_price2']
    
    df['wap1'] = (df['ask_price1'] * df['bid_size1'] + df['bid_price1'] * df['ask_size1']) / (df['ask_size1'] + df['bid_size1'])
    df['wap2'] = (df['ask_price2'] * df['bid_size2'] + df['bid_price2'] * df['ask_size2']) / (df['ask_size2'] + df['bid_size2'])
    df['spread_wap'] = df['wap1'] / df['wap2']
    
    df['log_return1'] = df.groupby('time_id')['wap1'].apply(lambda x: np.log(x).diff())
    df['log_return2'] = df.groupby('time_id')['wap2'].apply(lambda x: np.log(x).diff())
    
    df['ask_size_total'] = df['ask_size1'] + df['ask_size2']
    df['bid_size_total'] = df['bid_size1'] + df['bid_size2']
    df['size_spread_total'] = np.abs(df['ask_size_total'] / df['bid_size_total'] - 1)
    df['size_total'] = df['ask_size_total'] + df['bid_size_total']
    
    return df


def fe_trade_data(trade_df: pd.DataFrame, book_df: pd.DataFrame) -> pd.DataFrame:

    trade_df['size_per_order'] = trade_df['size'] / trade_df['order_count']
    trade_df['log_return_price'] = trade_df.groupby('time_id')['price'].apply(lambda x: np.log(x).diff())
    
    book_df['wap1'] = (book_df['ask_price1'] * book_df['bid_size1'] + book_df['bid_price1'] * book_df['ask_size1']) / (book_df['ask_size1'] + book_df['bid_size1'])
    trade_df = pd.merge(trade_df, book_df, how='left', on=['time_id', 'seconds_in_bucket'])

    trade_df['spread_ask1_trade'] = trade_df['ask_price1'] / trade_df['price']
    trade_df['spread_bid1_trade'] = trade_df['bid_price1'] / trade_df['price']
    trade_df['size_spread_trade_book'] = (trade_df['ask_size1'] + trade_df['ask_size2'] + trade_df['bid_size1'] + trade_df['bid_size2']) / (trade_df['size'] * 2)
    
    drop_col = [col for col in book_df.columns if col not in ['time_id', 'seconds_in_bucket']]
    trade_df.drop(drop_col, axis=1, inplace=True)
    book_df.drop(['wap1'], axis=1, inplace=True)
    
    return trade_df


def diff_min_max(series):
    return np.max(series) - np.min(series)


def realized_volatility(series):
    vol = np.sqrt(np.sum(series**2))
    return vol


def fe_change_in_window(base_df: pd.DataFrame, output_df: pd.DataFrame, window: str, add_start: int, add_end: int, data_type: Literal['book', 'trade']) -> pd.DataFrame:
    
    """
    Generate features that capture trends within the first 10 minutes.
    Specifically, further delimit the training data periods and compare 1~300sec vs 301~600sec and 301~450sec vs 451~600sec feture
    """

    if data_type == 'book':
        sec_col = 'seconds_in_bucket'
        agg_func_dict = {
            'log_return1': [realized_volatility],
            'log_return2': [realized_volatility],
            'spread_ask1_bid1': ['mean'],
            'spread_ask2_bid2': ['mean'],
            'spread_wap': ['mean'],
        }
    elif data_type == 'trade':
        sec_col = 'seconds_in_bucket_trade'
        agg_func_dict = {
            'seconds_in_bucket_trade': ['nunique'],
            'trade_size': ['sum'],
            'log_return_price': [realized_volatility],
        }
    
    tmp_df = base_df.groupby('time_id').apply(lambda df: df.loc[(add_start<=df[sec_col])&(df[sec_col]<add_end)]).reset_index(drop=True)
    tmp_df = tmp_df.groupby('time_id').agg(agg_func_dict)
    compare_col = ['_'.join(col) for col in tmp_df.columns.values]
    tmp_df.columns = ['_'.join(col) + f'_{window}_cp' for col in tmp_df.columns.values]
    for col in compare_col:
        output_df[f'{col}_change_{window}_{window}'] = output_df[f'{col}_{window}'] / tmp_df[f'{col}_{window}_cp']
        
    return output_df


def aggregate_book_data(book_df: pd.DataFrame, windows: list=[150, 300, 600]) -> pd.DataFrame:

    agg_func_dict = {
        'seconds_in_bucket': ['nunique'],
        'bid_price1': [diff_min_max],
        'ask_price1': [diff_min_max],
        'bid_price2': [diff_min_max],
        'ask_price2': [diff_min_max],
        'spread_ask1_bid1': ['mean'],
        'spread_ask2_bid2': ['mean'],
        'spread_ask1_ask2': ['mean'],
        'spread_bid1_bid2': ['mean'],
        'wap1': [diff_min_max],
        'wap2': [diff_min_max],
        'spread_wap': ['mean'],
        'log_return1': [realized_volatility, 'std'],
        'log_return2': [realized_volatility, 'std'],
        'size_spread_total': ['mean'],
        'size_total': ['sum', 'max'],
        'ask_size_total': ['sum'],
        'bid_size_total': ['sum'],
    }
    
    output_df = pd.DataFrame()
    for window in windows:
        tmp_df = book_df.groupby('time_id').apply(lambda df: df.loc[df['seconds_in_bucket']>=(600-window)]).reset_index(drop=True)
        tmp_df = tmp_df.groupby('time_id').agg(agg_func_dict)
        tmp_df.columns = ['_'.join(col) + f'_{str(window)}' for col in tmp_df.columns.values]
        tmp_df[f'size_total_diff_{window}'] = np.abs(tmp_df[f'ask_size_total_sum_{window}'] / (tmp_df[f'bid_size_total_sum_{window}'] + 1) - 1)
        tmp_df.drop([f'ask_size_total_sum_{window}', f'bid_size_total_sum_{window}'], axis=1, inplace=True)
        output_df = pd.concat([output_df, tmp_df], axis=1)

    output_df = fe_change_in_window(book_df, output_df, '300', 0, 300, 'book')
    output_df = fe_change_in_window(book_df, output_df, '150', 300, 450, 'book')
    output_df.reset_index(inplace=True)

    return output_df


def aggregate_trade_data(trade_df: pd.DataFrame, windows: list=[150, 300, 600]):

    trade_df.rename({'seconds_in_bucket': 'seconds_in_bucket_trade', 'size': 'trade_size'}, axis=1, inplace=True)
    
    agg_func_dict = {
        'seconds_in_bucket_trade': ['nunique'],
        'trade_size': ['sum', 'max'],
        'order_count': ['sum'],
        'size_per_order': ['mean', 'max'],
        'log_return_price': [realized_volatility, 'std'],
        'spread_ask1_trade': ['mean'],
        'spread_bid1_trade': ['mean'],
        'size_spread_trade_book': ['mean', 'max'],
    }

    output_df = pd.DataFrame()
    for window in windows:
        tmp_df = trade_df.groupby('time_id').apply(lambda df: df.loc[df['seconds_in_bucket_trade']>=600-window]).reset_index(drop=True)
        tmp_df = tmp_df.groupby('time_id').agg(agg_func_dict)
        tmp_df.columns = ['_'.join(col) + f'_{str(window)}' for col in tmp_df.columns.values]
        tmp_df[f'price_volatility_per_size_{window}'] = (tmp_df[f'log_return_price_realized_volatility_{window}'] / tmp_df[f'trade_size_sum_{window}']) * (1e8)
        output_df = pd.concat([output_df, tmp_df], axis=1)
        
    output_df = fe_change_in_window(trade_df, output_df, '300', 0, 300, 'trade')
    output_df = fe_change_in_window(trade_df, output_df, '150', 300, 450, 'trade')
    output_df.reset_index(inplace=True)

    return output_df


def make_book_df(file: str) -> pd.DataFrame:
    book_df = pd.read_parquet(file)
    book_df = fe_book_data(book_df)
    book_df = aggregate_book_data(book_df)
    stock_id = re.search(r'stock_id=[0-9]+', file).group(0).split('=')[1]
    book_df['stock_id'] = int(stock_id)
    book_df = use_low_memory(book_df, float_32=True)
    return book_df


def make_trade_df(book_file: str, trade_file: str) -> pd.DataFrame:
    book_df = pd.read_parquet(book_file)
    trade_df = pd.read_parquet(trade_file)
    trade_df = fe_trade_data(trade_df, book_df)
    del book_df
    
    trade_df = aggregate_trade_data(trade_df)
    stock_id = re.search(r'stock_id=[0-9]+', trade_file).group(0).split('=')[1]
    trade_df['stock_id'] = int(stock_id)
    trade_df = use_low_memory(trade_df, float_32=True)
    return trade_df


def make_base_feature(base_df: pd.DataFrame, data_path: str, data_type: Literal['train', 'test'], debug: bool) -> pd.DataFrame:
    if data_type == 'test':
        base_df.drop(['row_id'], axis=1, inplace=True)
        
    book_files = sorted(glob.glob(f'{data_path}/book_{data_type}.parquet/stock_id=*/*.parquet'))
    trade_files = sorted(glob.glob(f'{data_path}/trade_{data_type}.parquet/stock_id=*/*.parquet'))
    
    if data_type == 'train' and debug:
        book_files = book_files[:16]
        trade_files = trade_files[:16]
    
    book_dfs = Parallel(n_jobs=-1)(delayed(make_book_df)(file) for file in tqdm(book_files))
    trade_dfs = Parallel(n_jobs=-1)(delayed(make_trade_df)(b_file, t_file) for b_file, t_file in tqdm(zip(book_files, trade_files), total=len(book_files)))
    
    all_book_df = pd.concat(book_dfs, axis=0)
    all_trade_df = pd.concat(trade_dfs, axis=0)
    
    base_df = pd.merge(base_df, all_book_df, on=['stock_id', 'time_id'], how='left')
    base_df = pd.merge(base_df, all_trade_df, on=['stock_id', 'time_id'], how='left')
    del all_book_df, all_trade_df
    
    if data_type == 'train' and debug:
        base_df.dropna(axis=0, how='any', inplace=True)
    
    return base_df


def add_stock_cluster(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    table_df = df.loc[~df['target'].isna(), ['stock_id', 'time_id', 'target']].copy()
    table_df = table_df.pivot(index='time_id', columns='stock_id', values='target')

    corr_map = table_df.corr()
    stock_idx = corr_map.index
    kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(corr_map.values)
    cluster_df = pd.DataFrame(dict(stock_id=stock_idx, cluster_id=kmeans.labels_))
    df = pd.merge(df, cluster_df, on=['stock_id'], how='left')

    if df['cluster_id'].isna().sum() > 0:
        major_cluster = df['cluster'].value_counts().index[0]
        df.loc[df['cluster_id'].isna(), 'cluster_id'] = major_cluster

    return df


def fill_nan(df: pd.DataFrame) -> pd.DataFrame:
    df.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)
    df.fillna(0, inplace=True)
    return df


def make_agg_feature(df: pd.DataFrame) -> Tuple[pd.DataFrame]:

    # stock aggregation
    agg_dict = {
        'log_return1_realized_volatility_150': ['mean'],
        'log_return2_realized_volatility_150': ['mean'],
        'log_return1_realized_volatility_300': ['mean'],
        'log_return2_realized_volatility_300': ['mean'],
        'log_return1_realized_volatility_600': ['mean'],
        'log_return2_realized_volatility_600': ['mean'],
        'log_return_price_realized_volatility_150': ['mean'],
        'price_volatility_per_size_150': ['mean'],
        'log_return_price_realized_volatility_300': ['mean'],
        'price_volatility_per_size_300': ['mean'],
        'log_return_price_realized_volatility_600': ['mean'],
        'price_volatility_per_size_600': ['mean']
    }
    
    agg_df = df.groupby('stock_id').agg(agg_dict)
    agg_df.columns = ['_'.join(col) + '_by_stock' for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)
    df = pd.merge(df, agg_df, on=['stock_id'], how='left')
    apply_col = list(agg_dict.keys())
    for col in apply_col:
        df[f'{col}_diff_mean_by_stock'] = df[col] - df[f'{col}_mean_by_stock']

    # time aggregation
    add_dict = {
        'log_return1_realized_volatility_change_300_300': ['mean'],
        'log_return2_realized_volatility_change_300_300': ['mean'],
        'log_return1_realized_volatility_change_150_150': ['mean'],
        'log_return2_realized_volatility_change_150_150': ['mean'],
        'log_return_price_realized_volatility_change_300_300': ['mean'],
        'log_return_price_realized_volatility_change_150_150': ['mean'],
        'size_total_sum_150': ['sum'],
        'size_total_sum_300': ['sum'],
        'size_total_sum_600': ['sum'],
        'trade_size_sum_150': ['sum'],
        'trade_size_sum_300': ['sum'],
        'trade_size_sum_600': ['sum'],
        'trade_size_sum_change_300_300': ['mean'],
        'trade_size_sum_change_150_150': ['mean']
    }
    agg_dict.update(add_dict)
    agg_df = df.groupby('time_id').agg(agg_dict)
    agg_df.columns = ['_'.join(col) + '_by_time' for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)
    df = pd.merge(df, agg_df, on=['time_id'], how='left')
    apply_col = [col for col in list(agg_dict.keys()) if ('volatility' in col)]
    for col in apply_col:
        df[f'{col}_diff_mean_by_time'] = df[col] - df[f'{col}_mean_by_time']
    
    # time stock aggregation
    df = add_stock_cluster(df, n_clusters=7)
    agg_df = df.groupby(['time_id', 'cluster_id']).agg(agg_dict)
    agg_df.columns = ['_'.join(col) + '_by_cluster_time' for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)
    df = pd.merge(df, agg_df, on=['time_id', 'cluster_id'], how='left')
    apply_col = [col for col in list(agg_dict.keys()) if 'volatility' in col]
    for col in apply_col:
        df[f'{col}_diff_mean_by_cluster_time'] = df[col] - df[f'{col}_mean_by_cluster_time']
    
    df.drop(['cluster_id'], axis=1, inplace=True)

    train_df = df.loc[~df['target'].isna()]
    test_df = df.loc[df['target'].isna()].drop(['target'], axis=1)
    train_df, test_df = fill_nan(train_df), fill_nan(test_df)
    
    return train_df, test_df