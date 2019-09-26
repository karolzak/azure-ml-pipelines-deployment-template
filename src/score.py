import os
import pandas as pd
import datetime as dt
from sklearn.externals import joblib
import config as cfg


def run_score(input_path, output_path, model_path, runid):
    
    # load the data
    df = pd.read_csv(input_path)
    X = df[cfg.feat_cols].values
    #y = df[label_col].values    
    
    # load the model
    model = joblib.load(model_path)

    # run scoring
    y_pred = model.predict(X=X)
    df['level'] = y_pred
    df['runid'] = runid
    df['timestamp'] = dt.datetime.now()
    
    os.makedirs(output_path, exist_ok=True)    
    results_path = os.path.join(output_path, 'score_results.csv')
    df.to_csv(results_path, index=False)

    return results_path