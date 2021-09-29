#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

from fforma import FFORMA
# from fforma.utils_metrics import Naive2
from fforma.utils_ts_features import compute_tsfeatures

def main():
    complete_errors = pd.read_csv(r"R\data\train-errors-fforma.csv").set_index("unique_id")
    complete_features = pd.read_csv(r"R\data\train-feats-fforma.csv").set_index("unique_id")
    # If not precomputed we can use the below command to compute it from train data
    # complete_features = compute_tsfeatures(train_df, freq=12) #Freq id the seasonal period
    complete_predictions = pd.concat([
        pd.read_csv(r"R\data\preds-fforma-1.csv"),
        pd.read_csv(r"R\data\preds-fforma-2.csv"),
        pd.read_csv(r"R\data\preds-fforma-3.csv"),
        pd.read_csv(r"R\data\preds-fforma-4.csv"),
        
        ]).set_index(['unique_id', "ds"])
    
    #Training fforma

    # optimal params by hyndman
    optimal_params = {'n_estimators': 10,
                      'eta': 0.58,
                      'max_depth': 14,
                      'subsample': 0.92,
                      'colsample_bytree': 0.77}
    fforma = FFORMA(params=optimal_params)
    fforma.fit(errors=complete_errors,
               feats=complete_features)

    fforma_predictions = fforma.predict(complete_predictions, feats=complete_features)
    print(fforma_predictions)
    #evaluate predictions


if __name__=='__main__':
    main()
