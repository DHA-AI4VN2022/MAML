import numpy as np 
import torch 
import pandas as pd 
import numpy as np
import math
import os
import csv
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from util.metric import mae, mse, rmse, mape, nse, mdape
import json

def get_data_groundtruth(config, target_station):
	data_dir = config['data_dir'] + 'gauges/' + target_station +'.csv'
	lst_pm25 = pd.read_csv(data_dir)['PM2.5'].tolist()
	gt = lst_pm25[int(len(lst_pm25)* (1-config['test_size']) ): ]
	input_size = config['input_len']
	return gt[input_size:]

def save_result(args, config, res_train, res_test):
	# import pdb; pdb.set_trace()


	num_station, selection_strategy, train_ratio, mem_used, train_losses, val_losses, target, predict, log_dir, target_station, num_params, train_time, test_time = \
		args.num_input_station, args.station_selection_strategy, res_train['train_ratio'], res_train['mem_used'], res_train['train_losses'], res_train['val_losses'], res_test['groundtruth'], res_test['predict'], res_test['base_dir'], \
				res_test['target_station'], res_train['num_params'], res_train['train_time'], res_test['test_time']
	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	input_length = config['input_len']
	output_length = config['output_len']

	path = os.path.join(log_dir, "result.csv")

	# single task 
	if args.multi_task == False:
		y_true = target
		y_pred = predict 

		m_mae = mae(y_true, y_pred)
		m_mape = mape(y_true, y_pred)
		m_rmse = rmse(y_true, y_pred)
		m_mse = mse(y_true, y_pred)
		m_r2score = r2_score(y_true, y_pred)
		m_mdape = mdape(y_true, y_pred)

		results = [ dt_string, target_station, num_station, selection_strategy, input_length,  output_length, train_ratio, num_params, mem_used, train_time, test_time, m_mae, m_mape, m_mdape, m_rmse, m_mse, m_r2score]

		# groundtruth = get_data_groundtruth(config, target_station)

		visualize_result(y_true, y_pred, log_dir, target_station)
		visualize_train_val_loss(train_losses, val_losses, log_dir, target_station)

		file_existed = os.path.exists(path)
		with open(path, 'a+') as file:
			writer = csv.writer(file)
			if not file_existed:
				writer.writerow(config['input_features'])
				writer.writerow(["date", "Target Station", "Num Input Station", "Selection strategy", "Input length", "Output length","Train Ratio", "Num params", "Memory used (MB)","Train time", "Test time", "MAE", "MAPE", "MDAPE", "RMSE", "MSE", "R2Score"])
			writer.writerow(results)
	else: # multitask
		for stat in target.keys():
			y_true = target[stat]
			y_pred = predict[stat]

			m_mae = mae(y_true, y_pred)
			m_mape = mape(y_true, y_pred)
			m_rmse = rmse(y_true, y_pred)
			m_mse = mse(y_true, y_pred)
			m_r2score = r2_score(y_true, y_pred)
			m_mdape = mdape(y_true, y_pred)

			results = [ dt_string, stat, num_station, selection_strategy, input_length,  output_length, train_ratio, num_params, mem_used, train_time, test_time, m_mae, m_mape, m_mdape, m_rmse, m_mse, m_r2score]

			visualize_result(y_true, y_pred, log_dir, stat)
			visualize_train_val_loss(train_losses, val_losses, log_dir, stat)

			file_existed = os.path.exists(path)
			with open(path, 'a+', encoding='utf-8') as file:
				writer = csv.writer(file)
				if not file_existed:
					writer.writerow(["date", "Target Station", "Num Input Station", "Selection strategy", "Input length", "Output length","Train Ratio", "Num params", "Memory used (MB)","Train time", "Test time", "MAE", "MAPE", "MDAPE", "RMSE", "MSE", "R2Score"])
				writer.writerow(results)

				
def save_config(config):
	path = os.path.join(config['base_dir'], "config.json")
	with open(path, 'a+', encoding='utf-8') as f:
	    json.dump(config, f, ensure_ascii=False, indent=4)

def visualize_result(y_true, y_pred, log_dir, target_station):
	result_dir = log_dir + "/result/" 
	vis_dir = log_dir + "/visualize/"

	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	if not os.path.exists(vis_dir):
		os.makedirs(vis_dir)
	# import pdb; pdb.set_trace
	# gt = groundtruth[:len(y_true)]
	# df = pd.DataFrame(data={'Groundtruth_from_file': gt, 'Groundtruth':y_true, 'Predict': y_pred})
	df = pd.DataFrame({'Groundtruth':y_true, 'Predict': y_pred})
	df.to_csv(result_dir + 'gt_predict_{}.csv'.format(target_station))

	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(12,8))
	ax = plt.axes()

	ax.plot(y_pred, label='preds')
	ax.plot(y_true, label='gt')
	ax.legend()
	fig.savefig(vis_dir + 'result_visualize_{}.png'.format(target_station))
	plt.close()

def visualize_train_val_loss(train_loss, val_loss, log_dir, target_station):

	result_dir = log_dir + "/result/" 
	vis_dir = log_dir + "/visualize/"

	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	if not os.path.exists(vis_dir):
		os.makedirs(vis_dir)

	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(12,8))
	ax = plt.axes()

	df = pd.DataFrame(data={'train':train_loss, 'val': val_loss})
	df.to_csv(result_dir + 'train_val_loss_{}.csv'.format(target_station))

	ax.plot(train_loss, label='train_loss')
	ax.plot(val_loss, label='val_loss')
	ax.legend()
	fig.savefig(vis_dir + 'train_val_visualize_{}.png'.format(target_station))
	plt.close()

def generate_log_dir(args, base_log_dir=''):
	if base_log_dir != '' and os.path.exists(base_log_dir + args.experimental_mode + '/'):
		log_dir = base_log_dir + args.experimental_mode
	else:
		log_dir = os.path.join("log", args.model, args.experimental_mode)
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
	print(log_dir)
	return log_dir

def load_optimizer(config, model):
    optimizer_type = config['train']['optimizer']
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    return optimizer

def save_checkpoint(model, optimizer, path):
    checkpoints = {
        "model_dict": model.state_dict(),
        "optimizer_dict": optimizer.state_dict(),
    }
    torch.save(checkpoints, path)

def load_model(model, checkpoint_path):
    return model.load_state_dict(torch.load(checkpoint_path)["model_dict"])