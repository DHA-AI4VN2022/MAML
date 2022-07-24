import os
from random import random
from turtle import shape
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from sklearn.preprocessing import MinMaxScaler
import yaml


def unscale_tensor(config, tensor, scaler):
    num_stations = config['num_stations']
    output_len = config['output_len']
    # target_ft_idx = config['meteo_features'].index(config['target_features'][0])

    sequence = tensor
    sequence = sequence.cpu().detach().numpy()
    #reshape tp batch_size, time_steps, num_stations
    sequence = sequence.reshape(sequence.shape[0], output_len, num_stations)
    # sequence = sequence.reshape(-1, 1)
    # print(sequence.shape)

    padded_sequence = np.zeros( (sequence.shape[0], output_len, (len( config['meteo_features']) + 1) * num_stations) )
    padded_sequence[:,:,:num_stations] = sequence
    # print(padded_sequence.shape)
    # padded_sequence = np.zeros(shape=(tensor.shape[0] * num_stations * output_len, (len( config['data']['meteo_features']) + 1) * num_stations))
    # print("papadded_sequence.shape)
    # padded_sequence[:, 0] = sequence[:, 0]
    unscaled_sequence = np.zeros( (sequence.shape[0], output_len, (len( config['meteo_features']) + 1) * num_stations) )
    for time_steps in range(output_len):
        time_step_sequence = padded_sequence[:,time_steps,:]
        # print(time_step_sequence.shape)
        unscaled_time_step_sequence = scaler.inverse_transform(time_step_sequence)
        unscaled_sequence[:,time_steps,:] = unscaled_time_step_sequence
    # print(unscaled_sequence.shape)
    # print(unscaled_sequence.shape)
    unscaled_data = unscaled_sequence[:,:,:num_stations]
    output = unscaled_data.reshape(sequence.shape[0], num_stations, output_len)
    # unscaled_data = np.transpose(unscaled_data, (1, 0, 2))
    # output = unscaled_data.reshape(num_stations, -1)
    return output


def get_poi_data_array(config):
    file_path = config['data_dir']
    file_poi = file_path + 'poi/POI.csv'

    poi_df = pd.read_csv(file_poi)
    arr = poi_df.iloc[:, 1:].astype(np.float32).values
    return arr


def get_data_array(args, config):
    file_path = config['data_dir']

    file_gauge = file_path + 'gauges/'
    file_location = file_path + 'location.csv'
    list_station = [stat.split('.csv')[0] for stat in os.listdir(file_gauge)]

    list_k_stations = config['target_station']
    meteo_features = config['meteo_features']
    pm25_features = config['target_features']

    location_df = pd.read_csv(file_location)
    scaler = MinMaxScaler()
    location_ = []
    list_pm25_arr = []
    list_meteo_arr = []

    for stat in list_k_stations:
        row_stat = location_df[location_df['location'] == stat]  # name, lat, lon
        location_it = row_stat.values[:, [0, 2, 1]]
        location_.append(location_it)

        df = pd.read_csv(file_gauge + f"{stat}.csv")
        df_pm = df[pm25_features]
        df_meteo = df[meteo_features]
        pm_arr = df_pm.iloc[:, :].astype(np.float32).values
        meteo_arr = df_meteo.iloc[:, :].astype(np.float32).values
        list_pm25_arr.append(pm_arr)
        list_meteo_arr.append(meteo_arr)
    list_pm25_arr = np.concatenate(list_pm25_arr, axis=-1)  # (num_time_steps, num_stations)
    list_meteo_arr = np.concatenate(list_meteo_arr, axis=-1)  # (num_time_steps, num_stations *  num_features)
    list_all_features_arr = np.concatenate((list_pm25_arr, list_meteo_arr), axis=1)
    transformed_data = scaler.fit_transform(list_all_features_arr)
    transformed_pm25 = transformed_data[:, :list_pm25_arr.shape[1]]

    transformed_meteo = transformed_data[:, list_pm25_arr.shape[1]:]
    location_ = np.concatenate(location_, axis=0)
    return (transformed_pm25, transformed_meteo), location_, list_k_stations, scaler



class MultiTaskPM25Dataset(Dataset):

    def __init__(self,  pm25_data, meteo_data, config):
        self.pm25_data = pm25_data
        self.meteo_data = meteo_data
        self.sequence_length = config['input_len']
        self.horizon = config['output_len']

    def __len__(self):
        return len(self.meteo_data)

    def __getitem__(self, i):
        if i >= self.sequence_length:
            i_start = i - self.sequence_length
            xpm25 = torch.tensor(self.pm25_data[i_start:i]).float()
            xmeteo = torch.tensor(self.meteo_data[i_start:i]).float()
            # xmeteo = xmeteo.flatten()
        else:
            padding = torch.tensor(self.pm25_data[0]).float()
            padding = padding.repeat(self.sequence_length - i, 1)
            xpm25 = torch.tensor(self.pm25_data[0:i]).float()
            xpm25 = torch.cat((padding, xpm25), 0)

            padding = torch.tensor(self.meteo_data[0]).float()
            padding = padding.repeat(self.sequence_length - i, 1)
            xmeteo = torch.tensor(self.meteo_data[0:i]).float()
            xmeteo = torch.cat((padding, xmeteo), 0)
            # xmeteo = xmeteo.flatten()

        if i <= len(self) - self.horizon:
            ypm25 = torch.tensor(self.pm25_data[i:i + self.horizon]).float()
            ypm25 = ypm25.view(ypm25.shape[1], ypm25.shape[0])

        else:
            ypm25 = torch.tensor(self.pm25_data[i:]).float()
            padding = torch.tensor(self.pm25_data[-1]).float()
            padding = padding.repeat(self.horizon - len(self) + i, 1)
            ypm25 = torch.cat((padding, ypm25), 0)
            ypm25 = ypm25.view(ypm25.shape[1], ypm25.shape[0])
        # ymeteo = torch.clone(xmeteo)
        return xpm25, ypm25, xmeteo


def get_dataloader(pm25_data, meteo_data, args, config, train_ratio):
    len_dataset = len(pm25_data)
    train_pct = config['train_size']
    valid_pct = config['valid_size']
    test_pct = config['test_size']

    train_pm_data = pm25_data[:int(len_dataset * train_pct * train_ratio)]
    valid_pm_data = pm25_data[int(len_dataset * (1 - test_pct - valid_pct)): int(len_dataset * (1 - test_pct))]
    test_pm_data = pm25_data[int(len_dataset * (1 - test_pct)):]

    train_meteo_data = meteo_data[:int(len_dataset * train_pct * train_ratio)]
    valid_meteo_data = meteo_data[int(len_dataset * (1 - test_pct - valid_pct)): int(len_dataset * (1 - test_pct))]
    test_meteo_data = meteo_data[int(len_dataset * (1 - test_pct)):]

    # if args.data_splitting == 'hold-out':
    if 0 == 0:
        train_dataset = MultiTaskPM25Dataset(
            pm25_data=train_pm_data,
            meteo_data=train_meteo_data,
            config=config
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            drop_last=True
        )
        validation_dataset = MultiTaskPM25Dataset(
            pm25_data=valid_pm_data,
            meteo_data=valid_meteo_data,
            config=config
        )
        valid_dataloader = DataLoader(
            validation_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            drop_last=True
        )

        test_dataset = MultiTaskPM25Dataset(
            pm25_data=test_pm_data,
            meteo_data=test_meteo_data,
            config=config
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            drop_last=True
        )

    elif args.data_splitting == 'time-series-cross-validation':
        pass
    elif args.data_splitting == 'blocking-cross-validation':
        pass
    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == '__main__':
    args = {'num_input_station': 10, 'station_selection_strategy': 'correlation'}
    with open("../../config/mlae.yml", encoding= 'utf-8') as f:
        config = yaml.safe_load(f)

    (pm_array, meteo_array), location, list_k_stations, scaler = get_data_array( args=args,
                                                                                config=config)
    # print(meteo_array, meteo_array.shape)
    dataset = MultiTaskPM25Dataset(pm25_data=pm_array, meteo_data=meteo_array, config=config)
    train_loader, valid_loader, test_loader = get_dataloader(pm25_data=pm_array, meteo_data=meteo_array, args=args,
                                                             config=config, train_ratio=0.3)
    for i, data in enumerate(train_loader):
        xpm, ypm, xmeteo = data
        print(xpm.shape, ypm.shape, xmeteo.shape)
        print(unscale_tensor(config,ypm, scaler))
        break
