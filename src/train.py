import math
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


def train_step(X, y, alpha, model_path):
    model = Ridge(alpha=alpha)
    model.fit(X=X, y=y)
    joblib.dump(value=model, filename=model_path)
    return model


def run_train(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    # set parameters
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    model_name_base = 'model_alpha_%s.pkl'

    feat_cols = ['var1','var2','var3','var4','var5','var6','var7','var8','var9','var10']
    label_col = 'level'

    test_size = 0.2
    random_state = 66

    # read data from CSV file
    df = pd.read_csv(input_path)
    X = df.as_matrix(columns=feat_cols)
    y = df[label_col].values    

    # model tunning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    rmses = []
    model_paths = []

    for alpha in alphas:
        model_path = os.path.join(output_path, model_name_base % alpha)
        model = train_step(X_train, y_train, alpha, model_path)
        y_pred = model.predict(X=X_test)

        rmses.append(math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)))
        model_paths.append(model_path)
    
    # eval models
    # select the model with lowest mse
    best_model_index = np.argmin(rmses)
    best_alpha = alphas[best_model_index]
    best_rmse = rmses[best_model_index]
    best_model_path = os.path.join(output_path, model_name_base % 'best')

    # train final model with the whole dataset
    final_model = train_step(X, y, best_alpha, best_model_path)

    return best_model_path, best_rmse, best_alpha