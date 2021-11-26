import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import GroupKFold
import torch
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor
import psutil

from utils import scaling_data



params = {
    'n_d': 16,
    'n_a': 16,
    'n_steps': 2,
    'gamma': 2,
    'cat_idxs': [],
    'cat_dims': [],
    'cat_emb_dim': 8,
    'n_independent': 2,
    'n_shared': 2,
    'lambda_sparse': 1.0e-5,
    'verbose': 10,
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=2e-2),
    'scheduler_params': dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False),
    'scheduler_fn': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'mask_type': 'entmax',
    'seed': 21,
}


# custom metric
class RMSPE(Metric):
    def __init__(self):
        self._name = 'rmspe'
        self._maximize = False
    
    def __call__(self, y_true, y_score):
        rmspe = np.sqrt(np.mean(np.square((y_true - y_score) / y_true)))
        return rmspe


# custom loss
def RMSPELoss(y_pred, y_true):
    return torch.sqrt(torch.mean( ((y_true - y_pred) / y_true) ** 2 )).clone()


def tabnet_train_infer(train_df: pd.DataFrame, test_df: pd.DataFrame, fold: int) -> Tuple[np.ndarray]:
    
    num_workers = psutil.cpu_count()
    train_df, test_df = scaling_data(train_df, test_df, 'minmax')
    params['cat_idxs'] = [i for i, col in enumerate(train_df.columns) if col == 'stock_id']
    params['cat_dims'] = [train_df['stock_id'].nunique()]

    gkf = GroupKFold(n_splits=fold)
    oof = np.zeros(len(train_df))
    preds = []
    
    for train_idx, valid_idx in gkf.split(train_df, train_df['target'], groups=train_df['time_id']):
        train_data = train_df.iloc[train_idx]
        valid_data = train_df.iloc[valid_idx]
        
        X_train, y_train = train_data.drop(['target', 'time_id'], axis=1).values, train_data['target'].values.reshape(-1, 1)
        X_valid, y_valid = valid_data.drop(['target', 'time_id'], axis=1).values, valid_data['target'].values.reshape(-1, 1)
        
        unsupervised = TabNetPretrainer(**params)
        unsupervised.fit(
            X_train=X_train,
            eval_set=[X_valid],
            pretraining_ratio=0.2,
            max_epochs=2,
            patience=10,
        )
        
        clf = TabNetRegressor(**params)
        clf.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            max_epochs=300,
            patience=50,
            batch_size=1024*40,
            virtual_batch_size=128*40,
            num_workers=num_workers,
            drop_last=False,
            eval_metric=[RMSPE],
            loss_fn=RMSPELoss,
            from_unsupervised=unsupervised
        )
        
        oof_pred = clf.predict(X_valid)
        oof_pred = np.clip(oof_pred, a_min=0, a_max=None).reshape(-1,)
        oof[valid_idx] = oof_pred
        
        X_test = test_df.drop(['time_id'], axis=1).values
        test_pred = clf.predict(X_test)
        test_pred = np.clip(test_pred, a_min=0, a_max=None).reshape(-1,)
        preds.append(test_pred)
    
    y_true = train_df['target'].values
    oof_score = np.sqrt(np.mean(np.square((y_true - oof) / y_true)))
    print('\n' + 'TabNet validation score : ' + str(oof_score) + '\n')
    
    tab_pred = np.mean(preds, axis=0)
    
    return oof, tab_pred