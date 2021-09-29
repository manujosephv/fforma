import numpy as np
from numpy.random import seed
seed(1)

import pandas as pd
from math import sqrt

########################
# METRICS
########################

def mse(y, y_hat):
  """
  Calculates Mean Squared Error.
  
  Parameters
  ----------  
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  
  Returns
  -------    
  mse: float
    mean squared error
  """
  y = np.reshape(y, (-1,))
  y_hat = np.reshape(y_hat, (-1,))
  mse = np.mean(np.square(y - y_hat)).item()
  return mse

def mape(y, y_hat):
  """
  Calculates Mean Absolute Percentage Error.
  Parameters
  ----------
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  
  Returns
  -------
  mape: float
    mean absolute percentage error
  """
  y = np.reshape(y, (-1,))
  y_hat = np.reshape(y_hat, (-1,))
  mape = np.mean(np.abs(y - y_hat) / np.abs(y))
  return mape

def smape(y, y_hat):
  """
  Calculates Symmetric Mean Absolute Percentage Error.
  Parameters
  ----------  
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  
  Returns
  -------
  smape: float
    symmetric mean absolute percentage error
  """
  y = np.reshape(y, (-1,))
  y_hat = np.reshape(y_hat, (-1,))
  smape = np.mean(2.0 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)))
  return smape

def mase(y, y_hat, y_train, seasonality):
  """
  Calculates Mean Absolute Scaled Error.
  Parameters
  ----------
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  y_train: numpy array
    actual train values for Naive1 predictions
  seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
  
  Returns
  -------
  mase: float
    mean absolute scaled error
  """
  y_hat_naive = []
  for i in range(seasonality, len(y_train)):
      y_hat_naive.append(y_train[(i - seasonality)])

  masep = np.mean(abs(y_train[seasonality:] - y_hat_naive))
  mase = np.mean(abs(y - y_hat)) / masep
  return mase

########################
# PANEL EVALUATION
########################

def evaluate_panel(y_panel, y_hat_panel, metric,
                   y_insample=None, seasonality=None):
  """
  Calculates metric for y_panel and y_hat_panel
  y_panel: pandas df
    panel with columns unique_id, ds, y
  y_naive2_panel: pandas df
    panel with columns unique_id, ds, y_hat
  y_insample: pandas df
    panel with columns unique_id, ds, y (train)
    this is used in the MASE
  seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
  return: list of metric evaluations
  """
  metric_name = metric.__code__.co_name

  y_panel = y_panel.sort_values(['unique_id', 'ds'])
  y_hat_panel = y_hat_panel.sort_values(['unique_id', 'ds'])
  if y_insample is not None:
      y_insample = y_insample.sort_values(['unique_id', 'ds'])

  assert len(y_panel)==len(y_hat_panel)
  assert all(y_panel.unique_id.unique() == y_hat_panel.unique_id.unique()), "not same u_ids"

  evaluation = []
  for u_id in y_panel.unique_id.unique():
    top_row = np.asscalar(y_panel['unique_id'].searchsorted(u_id, 'left'))
    bottom_row = np.asscalar(y_panel['unique_id'].searchsorted(u_id, 'right'))
    y_id = y_panel[top_row:bottom_row].y.to_numpy()

    top_row = np.asscalar(y_hat_panel['unique_id'].searchsorted(u_id, 'left'))
    bottom_row = np.asscalar(y_hat_panel['unique_id'].searchsorted(u_id, 'right'))
    y_hat_id = y_hat_panel[top_row:bottom_row].y_hat.to_numpy()
    assert len(y_id)==len(y_hat_id)

    if metric_name == 'mase':
      assert (y_insample is not None) and (seasonality is not None)
      top_row = np.asscalar(y_insample['unique_id'].searchsorted(u_id, 'left'))
      bottom_row = np.asscalar(y_insample['unique_id'].searchsorted(u_id, 'right'))
      y_insample_id = y_insample[top_row:bottom_row].y.to_numpy()
      evaluation_id = metric(y_id, y_hat_id, y_insample_id, seasonality)
    else:
      evaluation_id = metric(y_id, y_hat_id)
    evaluation.append(evaluation_id)
  return evaluation

def owa(y_panel, y_hat_panel, y_naive2_panel, y_insample, seasonality):
  """
  Calculates MASE, sMAPE for Naive2 and current model
  then calculatess Overall Weighted Average.
  y_panel: pandas df
    panel with columns unique_id, ds, y
  y_hat_panel: pandas df
    panel with columns unique_id, ds, y_hat
  y_naive2_panel: pandas df
    panel with columns unique_id, ds, y_hat
  y_insample: pandas df
    panel with columns unique_id, ds, y (train)
    this is used in the MASE
  seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
  return: OWA
  """
  total_mase = evaluate_panel(y_panel, y_hat_panel, mase, 
                              y_insample, seasonality)
  total_mase_naive2 = evaluate_panel(y_panel, y_naive2_panel, mase,
                                     y_insample, seasonality)
  total_smape = evaluate_panel(y_panel, y_hat_panel, smape)
  total_smape_naive2 = evaluate_panel(y_panel, y_naive2_panel, smape)

  assert len(total_mase) == len(total_mase_naive2)
  assert len(total_smape) == len(total_smape_naive2)
  assert len(total_mase) == len(total_smape)
  
  naive2_mase = np.mean(total_mase_naive2)
  naive2_smape = np.mean(total_smape_naive2) * 100

  model_mase = np.mean(total_mase)
  model_smape = np.mean(total_smape) * 100
  
  model_owa = ((model_mase/naive2_mase) + (model_smape/naive2_smape))/2
  return model_owa, model_mase, model_smape

def evaluate_prediction_owa(y_hat_df, y_train_df, X_test_df, y_test_df, 
                            naive2_seasonality):
    """
    y_hat_df: pandas df
      panel with columns unique_id, ds, y_hat
    y_train_df: pandas df
      panel with columns unique_id, ds, y
    X_test_df: pandas df
      panel with columns unique_id, ds, x
    y_test_df: pandas df
      panel with columns unique_id, ds, y, y_hat_naive2
    naive2_seasonality: int
      seasonality for the Naive2 predictions (needed for owa)
    model: python class
      python class with predict method
    """
    y_panel = y_test_df.filter(['unique_id', 'ds', 'y'])
    y_naive2_panel = y_test_df.filter(['unique_id', 'ds', 'y_hat_naive2'])
    y_naive2_panel.rename(columns={'y_hat_naive2': 'y_hat'}, inplace=True)
    y_hat_panel = y_hat_df
    y_insample = y_train_df.filter(['unique_id', 'ds', 'y'])

    model_owa, model_mase, model_smape = owa(y_panel, y_hat_panel, 
                                             y_naive2_panel, y_insample,
                                             seasonality=naive2_seasonality)

    print(15*'=', ' Model evaluation ', 14*'=')
    print('OWA: {} '.format(np.round(model_owa, 3)))
    print('SMAPE: {} '.format(np.round(model_smape, 3)))
    print('MASE: {} '.format(np.round(model_mase, 3)))
    return model_owa, model_mase, model_smape