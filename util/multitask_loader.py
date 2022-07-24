import yaml 
import os 
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import yaml
import os 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random 

def get_data_array(stations, args,config):
    file_path = config['data_dir']

    file_gauge = file_path + 'gauges/'
    
    # print(list_station)

    num_k_best_stats = args.num_input_station
    selection_strategy = args.station_selection_strategy

    # list_k_stations = get_k_best_stations(file_gauge, file_location, list_station, target_station, num_k_best_stats, selection_strategy)
    list_input_ft = config['input_features']

    # location_df = pd.read_csv(file_location)
    scaler = MinMaxScaler()
    # location_ = []
    list_arr = []

    for stat in stations:
        # row_stat = location_df[location_df['location'] == stat]  # name, lat, lon
        # location_it = row_stat.values[:, [0, 2, 1]]
        # location_.append(location_it)

        df = pd.read_csv(file_gauge + f"{stat}.csv")
        df_ = df[list_input_ft]
        arr = df_.iloc[:, :].astype(np.float32).values
        list_arr.append(arr)
    num_ft = list_arr[0].shape[-1]  # 14
    list_arr = np.concatenate(list_arr, axis=0)  # 8640 * 10, 8
    # transformed_data = np.transpose(np.array(list_arr), (1,0,2))

    scaler.fit(list_arr)
    transformed = scaler.transform(list_arr)
    # transformed = list_arr.copy()
    transformed_data = transformed.reshape(len(stations), -1, num_ft)  # 33, 8642, 14
    transformed_data = np.transpose(transformed_data, (1, 0, 2))
    # location_ = np.concatenate(location_, axis=0)
    return transformed_data, scaler

def unscale_tensor(config, tensor, scaler):
    num_stations = config['num_input_station']
    output_len = config['output_len']
    target_ft_idx = config['input_features'].index(config['target_features'][0])

    sequence = tensor
    sequence = sequence.cpu().detach().numpy()
    sequence = sequence.reshape(-1, 1)
    
    padded_sequence = np.zeros(shape = (tensor.shape[0] * num_stations * output_len, len(config['input_features'])) )
    padded_sequence[:,target_ft_idx] = sequence[:,0]

    unscaled_sequence = scaler.inverse_transform(padded_sequence)[:,0]
    unscaled_data = unscaled_sequence.reshape(-1, num_stations, output_len)
    unscaled_data = np.transpose(unscaled_data, (1,0,2))
    output = unscaled_data.reshape(num_stations, -1)
    return output
    
