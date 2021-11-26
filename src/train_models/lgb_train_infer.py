import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from typing import Tuple



params_1 = {
    'objective': 'rmse',
    'metric': 'None',
    'boosting': 'gbdt',
    'learning_rate': 0.01, 
    'num_iterations': 10000,
    'early_stopping_round': 200,
    'max_depth': 12,
    'num_leaves': 2500,
    'feature_fraction': 0.15,
    'bagging_fraction': 0.80,
    'bagging_freq': 3,
    'lambda_l1': 2.0,
    'lambda_l2': 5.0,
    'min_data_in_leaf': 400,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 10,
}

params_2 = {
    'objective': 'rmse',
    'metric': 'None',
    'boosting': 'gbdt',
    'learning_rate': 0.01,
    'num_iterations': 10000,
    'early_stopping_round': 200,
    'max_depth': 7,
    'num_leaves': 100,
    'feature_fraction': 0.15,
    'bagging_fraction': 0.80,
    'bagging_freq': 3,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'min_data_in_leaf': 1000,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}


# custom metric
def eval_rmspe(y_pred, dataset):
    y_true = dataset.get_label()
    eval_score = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    return 'RMSPE', eval_score, False


def lgb_train_infer(train_df: pd.DataFrame, test_df: pd.DataFrame, fold: int) -> Tuple[np.ndarray]:
    
    gkf = GroupKFold(n_splits=fold)
    model_oofs = []
    model_preds = []
    
    for params in [params_1, params_2]:
        
        oof = np.zeros(len(train_df))
        preds = []
        
        for train_idx, valid_idx in gkf.split(train_df, train_df['target'], groups=train_df['time_id']):
            train_data = train_df.iloc[train_idx]
            valid_data = train_df.iloc[valid_idx]
            
            X_train, y_train = train_data.drop(['target', 'time_id'], axis=1), train_data['target']
            X_valid, y_valid = valid_data.drop(['target', 'time_id'], axis=1), valid_data['target']
            
            # metric is RMSPE → optimize using weight
            train_weights = 1 / np.square(y_train)
            valid_weights = 1 / np.square(y_valid)
            
            lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=['stock_id'], weight=train_weights)
            lgb_valid = lgb.Dataset(X_valid, y_valid, categorical_feature=['stock_id'], weight=valid_weights)
            
            model = lgb.train(
                        params=params,
                        train_set=lgb_train,
                        valid_sets=[lgb_train, lgb_valid], 
                        verbose_eval=100,
                        feval=eval_rmspe
                    )
            
            valid_pred = model.predict(X_valid)
            oof[valid_idx] = valid_pred
            
            X_test = test_df.drop(['time_id'], axis=1)
            test_pred = model.predict(X_test)
            preds.append(test_pred)
        
        model_oofs.append(oof)
        mean_preds = np.mean(preds, axis=0)
        model_preds.append(mean_preds)
    
    lgb_oof = model_oofs[0] * 0.70 + model_oofs[1] * 0.30 # get for error analysis
    y_true = train_df['target'].values
    oof_score = np.sqrt(np.mean(np.square((y_true - lgb_oof) / y_true)))
    print('\n' + 'LightGBM validation score (after average): ' + str(oof_score) + '\n')
    
    lgb_pred = model_preds[0] * 0.70 + model_preds[1] * 0.30
    
    return lgb_oof, lgb_pred