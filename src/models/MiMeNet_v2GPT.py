import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MiMeNet():

    def __init__(self, input_len, output_len, num_layer=1, layer_nodes=128, 
                 l1=0.0001, l2=0.0001, dropout=0.25, batch_size=1024, patience=40,
                 lr=0.0001, seed=42, gaussian_noise=0):

        reg = tf.keras.regularizers.L1L2(l1, l2)
        
        self.model = tf.keras.Sequential()
        for l in range(num_layer):
            self.model.add(tf.keras.layers.Dense(layer_nodes, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc" + str(l)))
            self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(tf.keras.layers.Dense(output_len, activation='linear', kernel_regularizer=reg, bias_regularizer=reg, name="output"))

        self.num_layer = num_layer
        self.layer_nodes = layer_nodes
        self.l1 = l1
        self.l2 = l2
        self.dropout = dropout
        self.learning_rate = lr

        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed

    def train(self, train_x, train_y):
        # Ensure consistent dtype
        train_x = np.asarray(train_x, dtype=np.float32)
        train_y = np.asarray(train_y, dtype=np.float32)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='MSE')
        es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=self.patience, restore_best_weights=True)

        self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=100000, 
                       callbacks=[es_cb], validation_split=0.2)
        return

    def test(self, test_x):
        # Ensure consistent dtype
        test_x = np.asarray(test_x, dtype=np.float32)
        return self.model.predict(test_x, verbose=0)

    
    def get_scores(self):
        w_list = []
        for l in self.model.layers:
            if len(l.get_weights()) > 0:
                if l.get_weights()[0].ndim == 2:
                    w_list.append(l.get_weights()[0])
        num_layers = len(w_list)
        scores = w_list[0]
        for w in range(1,num_layers):
            scores = np.matmul(scores, w_list[w])
        return scores

    def destroy(self):
        tf.keras.backend.clear_session()
        return

    def get_params(self):
        return self.num_layer, self.layer_nodes, self.l1, self.l2, self.dropout, self.learning_rate



def build_model(input_shape, output_shape,num_layer=1, layer_nodes=128, 
                 l1=0.0001, l2=0.0001, dropout=0.25, batch_size=1024, patience=40,
                 lr=0.0001, gaussian_noise=0):
        
        tf.keras.utils.set_random_seed(42)
        reg = tf.keras.regularizers.L1L2(l1=l1, l2=l2)
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=input_shape))
        
        for _ in range(num_layer):
            model.add(tf.keras.layers.Dense(layer_nodes, activation='relu', 
                                        kernel_regularizer=reg, bias_regularizer=reg))
            model.add(tf.keras.layers.Dropout(dropout))
        
        model.add(tf.keras.layers.Dense(output_shape, activation='linear', 
                                    kernel_regularizer=reg, bias_regularizer=reg))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
        return model


  
def tune_MiMeNet(train, seed=42):
    micro, metab = train

    micro = np.asarray(micro, dtype=np.float32)
    metab = np.asarray(metab, dtype=np.float32)

    input_shape = micro.shape[1]
    output_shape = metab.shape[1]

    best_params = {}
    best_score = -np.inf

    # Scorer: Spearman correlation mean across outputs
    def spearman_score(y_true, y_pred):
        return pd.DataFrame(y_true).corrwith(pd.DataFrame(y_pred), axis=0).mean()

    scorer = make_scorer(spearman_score, greater_is_better=True)

    # Hyperparameter search space
    param_dist = {
        "num_layer": [1, 2, 3],
        "layer_nodes": [32, 128, 512],
        "l1": [0.0],
        "l2": np.logspace(-4, -1, 10),
        "dropout": [0.1, 0.3, 0.5],
        "lr": [0.001]
    } 
    # KerasRegressor using scikeras
    regressor = KerasRegressor(
        model=build_model,
        input_shape=input_shape,
        output_shape=output_shape,
        epochs=1000,
        batch_size=1024,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)],
    )

    rscv = RandomizedSearchCV(
        regressor,
        param_distributions=param_dist,
        n_iter=20,
        scoring=scorer,
        cv=5,
        random_state=seed
    )

    try:
        rscv.fit(micro, metab)
        best_params = rscv.best_params_
        best_score = rscv.best_score_
        print(f"Best tuning score: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
    except Exception as e:
        print(f"RandomizedSearchCV failed: {e}")
        print("Using default parameters.")

        # Fallback values
        best_params = {
            "num_layer": 1,
            "layer_nodes": 128,
            "l1": 0.0001,
            "l2": 0.0001,
            "dropout": 0.25,
            "lr": 0.0001
        }

    return best_params