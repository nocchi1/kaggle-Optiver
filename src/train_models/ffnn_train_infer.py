import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import GroupKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from utils import scaling_data



class FFNNModel:
    def __init__(self):
        self.es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, verbose=0, mode='min', restore_best_weights=True)
        self.scheduler = keras.experimental.CosineDecayRestarts(1e-2, 1000, t_mul=2.0, m_mul=0.65, alpha=1e-6)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.scheduler)
        self.batch_size =  8192
        self.val_batch_size = 8192 * 5
        self.epochs = 2000


    def root_mean_squared_per_error(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square((y_true - y_pred)/ y_true))) 


    def make_model(self, stock_dim: int, input_dim: int) -> keras.Model:
        nn_input = keras.Input(shape=(input_dim,))
        stock_input = keras.Input(shape=(1,))
        stock_emb = keras.layers.Embedding(stock_dim, 16, input_length=1)(stock_input)
        stock_emb = keras.layers.Flatten()(stock_emb)
        
        x = keras.layers.Concatenate()([stock_emb, nn_input])
        x = keras.layers.Dense(128, activation='swish')(x)
        x_ = keras.layers.Dense(128, activation='swish')(x)
        x = keras.layers.Add()([x, x_]) # skip connection
        x = keras.layers.Dense(64, activation='swish')(x)
        x_ = keras.layers.Dense(64, activation='swish')(x)
        x = keras.layers.Add()([x, x_]) # skip connection
        x = keras.layers.Dense(16, activation='swish')(x)
        out = keras.layers.Dense(1, activation='linear')(x)
        
        model = keras.Model(inputs=[nn_input, stock_input], outputs=out)
        model.compile(self.optimizer, self.root_mean_squared_per_error)
        
        return model
    

    def fit_model(self, train_data: Tuple[np.ndarray], valid_data: Tuple[np.ndarray], stock_dim: int, input_dim: int) -> keras.Model:
        model = self.make_model(stock_dim, input_dim)
        X_train, stock_train, y_train = train_data
        X_valid, stock_valid, y_valid = valid_data
        
        model.fit(
            [X_train, stock_train],
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=([X_valid, stock_valid], y_valid),
            validation_batch_size=self.val_batch_size,
            callbacks=[self.es],
            shuffle=True,
            verbose=1
        )
        
        return model
        
        
    def pred_model(self, model: keras.Model, test_data: Tuple[np.ndarray]) -> np.ndarray:
        X_test, stock_test = test_data
        preds = model.predict([X_test, stock_test]).reshape(-1,)
        print(preds.shape)
        return preds



def ffnn_train_infer(train_df: pd.DataFrame, test_df: pd.DataFrame, fold: int) -> Tuple[np.ndarray]:
    
    ffnn = FFNNModel()
    stock_dim = train_df['stock_id'].nunique()
    input_dim = len(train_df.columns) - 3
    gkf = GroupKFold(n_splits=fold)
    
    train_df, test_df = scaling_data(train_df, test_df, 'minmax')
    oof = np.zeros(len(train_df))
    preds = []
    stock_test = test_df['stock_id'].values
    X_test = test_df.drop(['time_id', 'stock_id'], axis=1).values
    
    for train_idx, valid_idx in gkf.split(train_df, train_df['target'], groups=train_df['time_id']):
        train_data = train_df.iloc[train_idx]
        valid_data = train_df.iloc[valid_idx]
        
        stock_train = train_data['stock_id'].values
        stock_valid = valid_data['stock_id'].values
        X_train, y_train = train_data.drop(['target', 'time_id', 'stock_id'], axis=1).values, train_data['target'].values
        X_valid, y_valid = valid_data.drop(['target', 'time_id', 'stock_id'], axis=1).values, valid_data['target'].values
        
        nn_train = (X_train, stock_train, y_train)
        nn_valid = (X_valid, stock_valid, y_valid)
        model = ffnn.fit_model(nn_train, nn_valid, stock_dim, input_dim)
        
        nn_valid = (X_valid, stock_valid)
        valid_pred = ffnn.pred_model(model, nn_valid)
        oof[valid_idx] = valid_pred
        
        nn_test = (X_test, stock_test)
        test_pred = ffnn.pred_model(model, nn_test)
        preds.append(test_pred)
        
    y_true = train_df['target'].values
    oof_score = np.sqrt(np.mean(np.square((y_true - oof) / y_true)))
    print('\n' + 'FFNN validation score : ' + str(oof_score) + '\n')
    
    ffnn_pred = np.mean(preds, axis=0)
    
    return oof, ffnn_pred