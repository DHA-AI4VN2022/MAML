from select import select
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
import yaml
import os
import csv
from datetime import datetime

def get_dataloader(station, data, location, config, target_station, train_ratio):
    len_dataset = len(data)
    train_pct = config['train_size']
    valid_pct = config['valid_size']
    test_pct = config['test_size']
    # import pdb; pdb.set_trace()

    train_data = data[:int(len_dataset * train_pct * train_ratio)]
    valid_data = data[int(len_dataset * (1 - test_pct - valid_pct)): int(len_dataset * (1 - test_pct))]
    test_data = data[int(len_dataset * (1 - test_pct)):]

    train_dataset = AQDataset(
        station=station,
        data=train_data,
        location=location,
        config=config,
        target_station=target_station
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True
    )

    validation_dataset = AQDataset(
        station=station,
        data=valid_data,
        location=location,
        config=config,
        target_station=target_station
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True
    )

    test_dataset = AQDataset(
        station=station,
        data=test_data,
        location=location,
        config=config,
        target_station=target_station
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True
    )

    return train_dataloader, validation_dataloader, test_dataloader

class AQDataset(Dataset):
    def __init__(self, station, data, location, config, target_station):
        super().__init__()
        self.data = data  # (day, station, feature )
        self.location = location
        self.config = config
        self.station_list_name = station

        self.target_station_name = target_station
        self.target_station_idx = self.station_list_name.index(target_station)

        self.input_len = self.config['input_len']
        self.output_len = self.config['output_len']

        self.adj_matrix = self.get_adjacency_matrix()
        self.target_ft_idx = self.config['input_features'].index(self.config['target_features'][0])

    def laplacian_matrix(self, adj):
        n = adj.shape[0]
        deg = np.zeros([n, n])
        for i in range(n):
            degree = np.count_nonzero(adj[i])
            deg[i][i] = degree

        deg_norm = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten())

        identity = np.identity(adj.shape[0])
        laplacian = identity - deg_norm.dot(adj.dot(deg_norm))
        return laplacian

            
    def get_distance(self, coords1, coords2):
        from geopy import  distance
        return distance.geodesic(tuple(coords1), tuple(coords2)).km


    def get_adjacency_matrix(self):
        dist_matrix = self.get_distance_matrix()
        # import pdb; pdb.set_trace()

        reverse_dist = 1 / dist_matrix
        adj_matrix = reverse_dist / reverse_dist.sum(axis=0)

        num_input_day = self.config['input_len']
        adj_matrix = np.expand_dims(adj_matrix, axis=0)
        adj_matrix_ = np.repeat(adj_matrix, num_input_day, axis=0)
        return adj_matrix_

    def get_distance_matrix(self):
        dist_matrix = []
        for i, _ in enumerate(self.station_list_name):
            tmp_matrix = []
            for j, _ in enumerate(self.station_list_name):
                dist = self.get_distance(self.location[i, 1:], self.location[j, 1:])
                if i == j:
                    dist += 1
                tmp_matrix.append(dist)
            dist_matrix.append(tmp_matrix)
        return np.array(dist_matrix)

    def get_scaled_laplacian_matrix(self):
        # Calculate the laplacian matrix
        adj = self.adj_matrix[0]
        laplacian = self.laplacian_matrix(adj)
        # Calculate the maximum eigenvalue
        eigen_values, _ = np.linalg.eig(laplacian)
        eigen_max = max(eigen_values)
        # Identity matrix
        i = np.identity(adj.shape[0])

        # Calculate L
        scaled_laplacian = (2 * laplacian) / eigen_max - i
        # Repeat the laplacian for number of input time steps
        scaled_laplacian = np.array([scaled_laplacian for i in range(self.input_len)])
        return scaled_laplacian

    def __getitem__(self, index: int):
        x_spatial = self.data[
                    index: index + self.input_len, :, :
                    ]
        y = self.data[
            index + self.input_len: index + self.input_len + self.output_len,
            self.target_station_idx,
            self.target_ft_idx  # PM2.5
            ]
        G = self.get_scaled_laplacian_matrix()

        x_spatial, y, G = x_spatial.astype('float32'), y.astype('float32'), G.astype('float32')

        x_spatial = torch.from_numpy(x_spatial)
        y = torch.from_numpy(y)
        G = torch.from_numpy(G)
        return {'x': x_spatial, 'y':y, 'G':G}

    def __len__(self) -> int:
        return self.data.shape[0] - self.input_len - self.output_len + 1

    def get_dataset(self):
        x_tensor = []
        laplacian_tensor = []
        y_tensor = []

        for i in range(len(self)):
            x, y, laplacian = self[i]
            x_tensor.append(x)
            y_tensor.append(y)
            laplacian_tensor.append(laplacian)

        x_tensor = torch.stack(x_tensor, 0)
        laplacian_tensor = torch.stack(laplacian_tensor, 0)
        y_tensor = torch.stack(y_tensor, 0)
        return x_tensor, y_tensor, laplacian_tensor


# def get_dataloader(station, lst_data, location, config, target_station):
#     len_dataset = len(lst_data)
#     train_pct = config['data']['train_size']
#     validation_pct = config['data']['valid_size']
#     test_pct = config['data']['test_size']
#     # import pdb; pdb.set_trace()

#     train_dataset = AQDataset(
#         station=station,
#         data=lst_data[:int(len_dataset * train_pct)],
#         # data=lst_data[:200],
#         location=location,
#         config=config,
#         target_station=target_station
#     )
#     train_dataloader = DataLoader(
#         train_dataset, batch_size=config['data']['batch_size'], shuffle=True, drop_last=True
#     )

#     validation_dataset = AQDataset(
#         station=station,
#         data=lst_data[int(len_dataset * (1 - test_pct - validation_pct)): int(len_dataset * (1 - test_pct))],
#         # data=lst_data[int(len_dataset * train_pct): int(len_dataset * train_pct) + 200],
#         location=location,
#         config=config,
#         target_station=target_station
#     )
#     validation_dataloader = DataLoader(
#         validation_dataset, batch_size=config['data']['batch_size'], shuffle=False, drop_last=True
#     )

#     test_dataset = AQDataset(
#         station=station,
#         data=lst_data[int(len_dataset * (1 - test_pct)):],
#         location=location,
#         config=config,
#         target_station=target_station
#     )
#     test_dataloader = DataLoader(
#         test_dataset, batch_size=config['data']['batch_size'], shuffle=False, drop_last=True
#     )

#     return (train_dataset, train_dataloader), (validation_dataset, validation_dataloader), (
#     test_dataset, test_dataloader)


def save_results(m_mae, m_mape, m_mse, m_rmse, m_r2score, log_dir, target_station, total_inference_time):
#   m_mae = mae(y_true, y_pred)
#   m_mape = mape(y_true, y_pred)
#   m_mse = mse(y_true, y_pred)
#   m_r2score = r2_score(y_true, y_pred)
#   m_nse = nse(y_true, y_pred)
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    results = [ dt_string, target_station, m_mae, m_mape,m_mse, m_rmse ,  m_r2score, total_inference_time]
#   path = os.path.join(log_dir, "metrics.csv")
    path =log_dir + "metrics.csv"

    with open(path, 'a', newline='', encoding = 'utf8') as file:
        writer = csv.writer(file)
        writer.writerow(["date", "Target Station", "MAE", "MAPE", "MSE", "RMSE", "R2_score", "Inference time"])
        writer.writerow(results)
    res =  {'MAE': m_mae, 'MAPE': m_mape, 'RMSE': m_rmse, 'MSE': m_mse, 'R2_score': m_r2score, 'Inference time': total_inference_time }
    return res


def save_checkpoint(state, file_name):
    torch.save(state, file_name)

def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"], strict = False)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def plot_results(plot_path, target_station, y, output):
    plt.figure(figsize=(25,15))
    plt.plot(y, label = "target")
    plt.plot(output, label = 'prediction')
    plt.legend(fontsize= 16)
    plt.suptitle( ("Station: {0}".format(target_station) )
            ,fontsize = 30  )
    plt.xlabel('Time Steps', fontsize = 15)
    plt.ylabel('PM 2.5', fontsize = 15)
    plt.savefig(fname = plot_path + '{0}.png'.format(target_station))

if __name__ == "__main__":
    config_file = "../../config/gc_lstm.yml"
    with open(config_file, encoding="utf8") as f:
        config = yaml.safe_load(f)
    target_station = config['data']['target_station'][0]
    list_data, location, list_station, scaler = get_data_array(target_station, config)
    print("Demo target station: ", target_station)
    print("Data array shape: {0}, Location array shape: {1}".format(list_data.shape, location.shape))
    print("Stations chosen:", list_station)

    (train_dataset,train_dataloader), (valid_dataset, validation_dataloader), (test_dataset,test_dataloader) = get_dataloader(station=list_station, lst_data=list_data, location=location,config=config,target_station=target_station)
    for batch_idx, (src, tgt, laplacian) in enumerate(train_dataloader):
        if batch_idx == 0:
            print()
            print('A batch demo')
            print("X shape {0}, Y shape {1}, Laplacian matrix shape: {2}"
                  .format( src.shape, tgt.shape, laplacian.shape))