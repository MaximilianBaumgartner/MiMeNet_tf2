import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import tensorflow as tf
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

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

    def train(self, train):
        train_x, train_y = train
        # Ensure consistent dtype
        train_x = np.asarray(train_x, dtype=np.float32)
        train_y = np.asarray(train_y, dtype=np.float32)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='mse')
        es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=self.patience, restore_best_weights=True)

        self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=100000, 
                       callbacks=[es_cb], validation_split=0.2)
        return

    def test(self, test):
        test_x, test_y = test
        # Ensure consistent dtype
        test_x = np.asarray(test_x, dtype=np.float32)
        num_class = test_y.shape[1]
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
        
        reg = tf.keras.regularizers.L1L2(l1=l1, l2=l2)
        model = tf.keras.Sequential(name=f"MiMeNet_{num_layer}_{layer_nodes}_{dropout}")
        model.add(tf.keras.Input(shape=input_shape, name="microbiome_input"))
        
        for i in range(num_layer):
            model.add(tf.keras.layers.Dense(layer_nodes, activation='relu', 
                                        kernel_regularizer=reg, bias_regularizer=reg, name=f"dense_{i}"))
            model.add(tf.keras.layers.Dropout(dropout, name=f"dropout_{i}"))
        
        model.add(tf.keras.layers.Dense(output_shape, activation='linear', 
                                    kernel_regularizer=reg, bias_regularizer=reg,name="output"))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
        return model


def spearman_score_np(y_true, y_pred):
    return np.mean([spearmanr(y_true[:, i], y_pred[:, i]).correlation for i in range(y_true.shape[1])])

def tune_MiMeNet(train, seed=42, n_trials=30):
    micro, metab = train
    micro = np.asarray(micro, dtype=np.float32)
    metab = np.asarray(metab, dtype=np.float32)
    input_shape = micro.shape[1:]
    output_shape = metab.shape[1]

    tf.keras.utils.set_random_seed(seed)

    X_train, X_val, y_train, y_val = train_test_split(micro, metab, test_size=0.2, random_state=seed)

    def objective(trial):
        # Suggest hyperparameters
        num_layer = trial.suggest_int("num_layer", 1,3)
        layer_nodes = trial.suggest_categorical("layer_nodes", [32, 64, 128, 512])
        l1 = trial.suggest_categorical("l1", [0.0])
        l2 = trial.suggest_loguniform("l2", 1e-4, 1e-1)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_categorical("lr", [0.001])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        # Build and compile model
        model = build_model(
            input_shape=input_shape,
            output_shape=output_shape,
            num_layer=num_layer,
            layer_nodes=layer_nodes,
            l1=l1,
            l2=l2,
            dropout=dropout,
            lr=lr
        )

        # Fit model
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=40, restore_best_weights=True, verbose=0
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1000,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )

        # Predict and evaluate
        y_pred = model.predict(X_val, batch_size=batch_size, verbose=0)
        score = spearman_score_np(y_val, y_pred)

        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"Best tuning score: {study.best_value:.4f}")
    print(f"Best parameters: {best_params}")

    return best_params

