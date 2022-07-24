from ast import Str
import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from util.multi_loader import get_k_best_stations
import os
import glob

def get_dataloader(station, data, location, config, target_station, train_ratio):
    len_dataset = len(data)
    train_pct = config['train_size']
    valid_pct = config['valid_size']
    test_pct = config['test_size']

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
        train_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True
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

    test_dataset = AQDatasetTest(
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
        self.target_ft_idx = self.config['input_features'].index(self.config['target_features'][0])
    
    def __getitem__(self, index: int):
        x_spatial = self.data[
                    index: index + self.input_len, :, :
                    ]
        x_target_station = x_spatial[:, self.target_station_idx, :]
        x_other_station = x_spatial[:, [idx for idx in range(len(self.station_list_name)) if idx != self.target_station_idx], self.target_ft_idx]
        
        
        x = np.concatenate((x_target_station, x_other_station), axis=1)
        y = self.data[
            index + self.input_len: index + self.input_len + self.output_len,
            self.target_station_idx,
            self.target_ft_idx  # PM2.5
            ]

        x, y= x.astype('float32'), y.astype('float32')
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return {'x': x, 'y':y}

    def __len__(self) -> int:
        return self.data.shape[0] - self.input_len - self.output_len + 1

class AQDatasetTest(Dataset):
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
        self.target_ft_idx = self.config['input_features'].index(self.config['target_features'][0])
    
    def __getitem__(self, index: int):
        x_spatial = self.data[
                    index: index + self.input_len, :, :
                    ]
        x_target_station = x_spatial[:, self.target_station_idx, :]
        x_other_station = x_spatial[:, [idx for idx in range(len(self.station_list_name)) if idx != self.target_station_idx], self.target_ft_idx]
        
        
        x = np.concatenate((x_target_station, x_other_station), axis=1)
        y = self.data[
            index + self.input_len: index + self.input_len + self.output_len,
            self.target_station_idx,
            self.target_ft_idx  # PM2.5
            ]

        x, y= x.astype('float32'), y.astype('float32')
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return {'x': x, 'y':y}

    def __len__(self) -> int:
        return self.data.shape[0] - self.input_len - self.output_len + 1

# def get_dataloader(dataset_class, batch_size, shuffle=True):
#     return DataLoader(
#         dataset_class, 
#         batch_size=batch_size, 
#         shuffle=shuffle
#     )

# class Generator(Dataset):
#     def __init__(self, inp, out):
#         self.inp = inp
#         self.out = out
#     def __len__(self):
#         return len(self.inp)
#     def __getitem__(self, index):
#         return {'x':self.inp[index],'y': self.out[index]}

# def get_data(args, config, folder, stations, station_name):
#     #Dataframe lưu giá trị của tất cả các trạm
#     file_path = config['data_dir']
#     file_gauge = file_path + 'gauges/'
#     file_location = file_path + 'location.csv'

#     num_k_best_stats = args.num_input_station
#     selection_strategy = args.station_selection_strategy

#     df_aqme = {}
#     lst_features = config['input_features']

#     #Danh sách tên tất cả các trạm
#     index = stations.index(station_name)
#     print(index)
    
#     for station in stations:
#         if station != '':
#             file_name = folder + "{}.csv".format(station)
#             #if file_name.split('.')[0] != 'location':
#             # import pdb; pdb.set_trace()

#             df_aqme[station] = pd.read_csv(file_name)
#             #stations.append(file_name.split('.')[0])
#     print(stations)

#     # Chọn trạm đầu tiên để dự đoán
#     df_aqme1 = df_aqme[stations[index]]
#     df_aqme1 = df_aqme1[lst_features]

#     # #Lấy chỉ số PM2.5 của tất cả các trạm và đánh giá hệ số tương quan
#     # pm2_5_all = pd.DataFrame()
#     # for station in stations:
#     #     if station != '':
#     #         pm2_5_all[station] = df_aqme[station]['PM2.5']    
#     # pm2_5_all.columns = [str(i) for i in range(0, len(pm2_5_all.columns))]
#     # station_corr = pm2_5_all.corr(method = 'pearson')

#     #Chọn k trạm có độ tương quan cao nhất 
#     # data = station_corr.sort_values(by=str(index), ascending=False).index[1:17].to_numpy() 
#     k_selected_stations = get_k_best_stations(file_gauge, file_location, stations, station_name, num_k_best_stats, selection_strategy)
#     # #print(station_name)
#     k_selected_stations = [stat for stat in k_selected_stations if stat != station_name]

#     # for i in data:
#     #     s = 'PM2.5_' + str(i)
#     #     df_aqme1[s] = df_aqme[stations[int(i)]]["PM2.5"]
#     for stat in k_selected_stations:
#         s = 'PM2.5_' + stat
#         #print(s)
#         df_aqme1[s] = df_aqme[stat]["PM2.5"]
#     print(df_aqme1.columns)
#     return df_aqme1

# def preprocess(args, config, name_station, train_ratio):
#     folder = config["dataset_dir"]
#     seq_len = config["input_len"]
#     horizon = config["output_len"]

#     target_stations = []

#     for path in glob.glob(os.path.join(folder, '*.csv')):
#         file_name = os.path.basename(path)
#         station_name = file_name.split('.')[0]
#         target_stations.append(station_name)

#     train_size = config['train_size']
#     valid_size = config['valid_size']
#     test_size = config['test_size']

#     df_aqme1 = get_data(args, config, folder, target_stations, name_station)
#     X = df_aqme1
#     y = df_aqme1[['PM2.5']]
#     # import pdb; pdb.set_trace()

#     # Xử lý outliers
#     #X = remove_outlier(X)
#     #y = remove_outlier(y)

#     # Phân chia tập train, test, validation
#     X_train = X[:int(len(X) * train_size * train_ratio)]
#     X_val = X[int(len(X)  * (1 - test_size - valid_size)): int(len(X) * (1 - test_size))]
#     X_test = X[int(len(X) * (1 - test_size)):]

#     y_train = y[:int(len(y) * train_size * train_ratio)]
#     y_val =  y[int(len(y)  * (1 - test_size - valid_size)): int(len(y) * (1 - test_size))]
#     y_test = y[int(len(y) * (1 - test_size)):]

#     # Dùng MinMaxScaler
#     scale_X = MinMaxScaler()
#     scale_y = MinMaxScaler()

#     scale_X.fit(X_train)
#     scale_y.fit(y_train)

#     X_train_data = scale_X.transform(X_train)
#     X_test_data = scale_X.transform(X_test)
#     X_valid_data = scale_X.transform(X_val)
#     y_valid_data = scale_y.transform(y_val)
#     y_train_data = scale_y.transform(y_train)
#     y_test_data = scale_y.transform(y_test)


#     X_train = np.array([X_train_data[i:i+seq_len] for i in range(0, len(X_train_data) - seq_len - horizon)])
#     X_test = np.array([X_test_data[i:i+seq_len] for i in range(0, len(X_test_data) - seq_len - horizon)])
#     X_valid = np.array([X_valid_data[i:i+seq_len] for i in range(0, len(X_valid_data) - seq_len - horizon)])

#     y_train = np.array([y_train_data[i+seq_len: i+seq_len+horizon] for i in range(0, len(y_train_data) - seq_len - horizon)])
#     y_test = np.array([y_test_data[i+seq_len: i+seq_len+horizon] for i in range(0, len(y_test_data) - seq_len - horizon)])
#     y_valid = np.array([y_valid_data[i+seq_len: i+seq_len+horizon] for i in range(0, len(y_valid_data) - seq_len - horizon)])

#     # Chuyển sang tensor
#     X_train = torch.FloatTensor(X_train)
#     X_test = torch.FloatTensor(X_test)
#     X_valid = torch.FloatTensor(X_valid)
#     y_train = torch.FloatTensor(y_train)
#     y_test = torch.FloatTensor(y_test)
#     y_valid = torch.FloatTensor(y_valid)
    
#     return X_train, y_train, X_valid, y_valid, X_test, y_test, scale_y