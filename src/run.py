import os
import numpy as np
import pandas as pd

from utils import set_seed, read_base_data, pycolor, submit_pred
from feature_engineering import make_base_feature, make_agg_feature
from train_models.lgb_train_infer import lgb_train_infer
from train_models.tabnet_train_infer import tabnet_train_infer
from train_models.ffnn_train_infer import ffnn_train_infer



SEED = 10
FOLD = 5
BASE_DATA_PATH = '../data'
DEBUG = True # False



if __name__ == '__main__':
    
    set_seed(SEED)
        
    print(pycolor.GREEN + 'Load data' + pycolor.END)
    train_df, test_df = read_base_data(BASE_DATA_PATH)
    
    print('\n'+ pycolor.GREEN + 'Start feature engineering' + pycolor.END)
    train_df = make_base_feature(train_df, BASE_DATA_PATH, 'train', debug=DEBUG)
    test_df = make_base_feature(test_df, BASE_DATA_PATH, 'test', debug=DEBUG)
    all_df = pd.concat([train_df, test_df], axis=0)
    train_df, test_df = make_agg_feature(all_df)
    del all_df
    
    print('\n'+ pycolor.GREEN + 'Save data before train model' + pycolor.END)
    train_df.to_csv('../output/train_fe.csv', index=False)
    test_df.to_csv('../output/test_fe.csv', index=False)
    
    print('\n'+ pycolor.GREEN + 'Start lgb trainig' + pycolor.END)
    _, lgb_pred = lgb_train_infer(train_df, test_df, fold=FOLD)
    print('\n'+ pycolor.GREEN + 'Start tabnet trainig' + pycolor.END)
    _, tab_pred = tabnet_train_infer(train_df, test_df, fold=FOLD)
    print('\n'+ pycolor.GREEN + 'Start ffnn trainig' + pycolor.END)
    _, ffnn_pred = ffnn_train_infer(train_df, test_df, fold=FOLD)
    
    final_pred = lgb_pred * 0.40 + tab_pred * 0.30 + ffnn_pred * 0.30 # ensemble
    submit_pred(test_df, final_pred)
    print('\n' + pycolor.GREEN + 'Done all process' + pycolor.END)