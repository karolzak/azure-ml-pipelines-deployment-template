import math
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import config as cfg


def train_step(X, y, alpha, model_path):
    model = Ridge(alpha=alpha)
    model.fit(X=X, y=y)
    joblib.dump(value=model, filename=model_path)
    return model


def run_train(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    # read data from CSV file
    df = pd.read_csv(input_path)
    X = df[cfg.feat_cols].values
    y = df[cfg.label_col].values    

    # model tunning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)
    rmses = []
    model_paths = []

    for alpha in cfg.alphas:
        model_path = os.path.join(output_path, cfg.model_name_base % alpha)
        model = train_step(X_train, y_train, alpha, model_path)
        y_pred = model.predict(X=X_test)

        rmses.append(math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)))
        model_paths.append(model_path)
    
    # eval models
    # select the model with lowest mse
    best_model_index = np.argmin(rmses)
    best_alpha = cfg.alphas[best_model_index]
    best_rmse = rmses[best_model_index]
    best_model_path = os.path.join(output_path, cfg.model_name_base % 'best')

    # train final model with the whole dataset
    final_model = train_step(X, y, best_alpha, best_model_path)

    return best_model_path, best_rmse, best_alpha