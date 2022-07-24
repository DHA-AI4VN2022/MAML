from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math 
import numpy as np

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def nse(y_true, y_pred):
    return (1-(np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)))

def mse(y_true, y_pred):
	return mean_squared_error(y_true, y_pred)

def mdape(y_true, y_pred):
	return np.median((np.abs(np.subtract(y_true, y_pred)/ y_true))) * 100