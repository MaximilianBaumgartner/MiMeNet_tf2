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
        
        tf.keras.utils.set_random_seed(seed)
        
        reg = tf.keras.regularizers.L1L2(l1=l1, l2=l2)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(input_len,)))

        for l in range(num_layer):
            self.model.add(tf.keras.layers.Dense(
                layer_nodes, activation='relu',
                kernel_regularizer=reg, bias_regularizer=reg, name=f"fc{l}"
            ))
            self.model.add(tf.keras.layers.Dropout(dropout))

        self.model.add(tf.keras.layers.Dense(
            output_len, activation='linear',
            kernel_regularizer=reg, bias_regularizer=reg, name="output"
        ))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='mse'
        )

        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = lr
        self.seed = seed

    def train(self, train):
        train_x, train_y = train
        es_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.patience,
            restore_best_weights=True
        )

        self.model.fit(
            train_x, train_y,
            batch_size=self.batch_size,
            validation_split=0.2,
            epochs=100000,  # High upper limit, but early stopping controls actual training
            verbose=0,
            callbacks=[es_cb]
        )

    def test(self, test):
        test_x, _ = test
        return self.model.predict(test_x, verbose=0)

    def get_scores(self):
        weights = [layer.get_weights()[0] for layer in self.model.layers if layer.weights]
        scores = weights[0]
        for w in weights[1:]:
            scores = np.matmul(scores, w)
        return scores

    def destroy(self):
        del self.model
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

    def get_params(self):
        return {
            "num_layer": self.model.num_layer,
            "layer_nodes": self.model.layer_nodes,
            "l1": self.model.l1,
            "l2": self.model.l2,
            "dropout": self.model.dropout,
            "lr": self.model.learning_rate,
        }
    
def tune_MiMeNet(train, seed=None):
    best_score = -np.inf
    best_params = {}

    micro, metab = train

    def build_model(num_layer=1, layer_nodes=128, 
                    l1=0.0001, l2=0.0001, dropout=0.25, lr=0.0001):

        reg = tf.keras.regularizers.L1L2(l1=l1, l2=l2)
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(micro.shape[1],)))
        for _ in range(num_layer):
            model.add(tf.keras.layers.Dense(layer_nodes, activation='relu',
                                            kernel_regularizer=reg, bias_regularizer=reg))
            model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(metab.shape[1], activation='linear',
                                        kernel_regularizer=reg, bias_regularizer=reg))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
        return model

    def custom_correlation(y_true, y_pred):
        return np.float64(pd.DataFrame(y_true).corrwith(pd.DataFrame(y_pred)).mean())

    scorer = make_scorer(custom_correlation, greater_is_better=True)

    param_dist = {
        "num_layer": [1, 2, 3],
        "layer_nodes": [32, 128, 512],
        "l1": [0],
        "l2": np.logspace(-4, -1, 10),
        "dropout": [0.1, 0.3, 0.5],
        "lr": [0.001]
    }

    es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=40, restore_best_weights=True)

    model = KerasRegressor(
        model=build_model,
        verbose=0,
        epochs=1000,
        batch_size=1024,
        callbacks=[es_cb],
        validation_split=0.2
    )

    for _ in range(20):
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=1,
            cv=5,
            scoring=scorer
        )
        search.fit(micro, metab)

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_params = search.best_params_

        tf.keras.backend.clear_session()

    print(f"Best score is: {best_score} using {best_params}")
    return best_params
