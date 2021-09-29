#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

from fforma import FFORMA
# from fforma.utils_metrics import Naive2
from tsfeatures import tsfeatures


# def prepare_to_train_fforma(dataset, validation_periods, seasonality):

#     X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset, './data', 100)

#     # Preparing errors
#     y_holdout_train_df, y_val_df = temp_holdout(y_train_df, validation_periods)
#     meta_models = {
#         #'ARIMA': ARIMA(freq=seasonality, stepwise=False, approximation=False),
#         'ETS': ETS(freq=seasonality),
#         'ThetaF': ThetaF(freq=seasonality),
#         'Naive': Naive(freq=seasonality),
#         'SeasonalNaive': SeasonalNaive(freq=seasonality),
#         'Naive2': Naive2(seasonality=seasonality)
#     }
#     validation_meta_models = MetaModels(meta_models)
#     validation_meta_models.fit(train)
#     prediction_validation_meta_models = validation_meta_models.predict(y_val_df)

#     #Calculating errors
#     errors = calc_errors(prediction_validation_meta_models, y_holdout_train_df, seasonality)

#     #Calculating features
#     features = tsfeatures(y_holdout_train_df, seasonality)

#     #Calculating actual predictins
#     meta_models = MetaModels(meta_models)
#     meta_models.fit(y_train_df)

#     predictions = meta_models.predict(y_test_df[['unique_id', 'ds']]))

#     return errors, features, predictions


def main():
    complete_errors = pd.read_csv(r"R\data\train-errors-fforma.csv").set_index("unique_id")
    # complete_errors = complete_errors.loc[complete_errors.index.str.startswith("Y")]
    complete_features = pd.read_csv(r"R\data\train-feats-fforma.csv").set_index("unique_id")
    # complete_features = complete_features.loc[complete_features.index.str.startswith("Y")]
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
