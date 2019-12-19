import numpy as np 
from time import time
import csv

def scaler(data, max_data, min_data):
	data = data.astype(np.float32)
	result = (data - min_data) / (max_data - min_data + 0.0001)
	result = 2 * result - 1
	return result

def inverse_scaler(data, max_data, min_data):
	result = (data + 1) * (max_data - min_data + 0.0001) / 2 + min_data
	return result

def log_scaler(data, max_data, min_data):
	max_data = max_data.astype('float32')
	min_data = min_data.astype('float32')

	max_data = np.log(max_data)
	min_data = np.log(min_data + 1)

	data = data.astype('float32')
	data = 1. * (np.log(data + 1) - min_data) / (max_data - min_data)
	data = data * 2 - 1

	return data

def inverse_log_scaler(data, max_data, min_data):
	max_data = max_data.astype('float32')
	min_data = min_data.astype('float32')

	max_data = np.log(max_data)
	min_data = np.log(min_data + 1)

	data = (data + 1.) / 2
	data = 1 * data * (max_data - min_data) + min_data
	data = np.exp(data) - 1

	return data

def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true + 0.001 - y_pred) / (y_true + 0.001))) * 100


