import yaml
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np

from util.multi_loader import get_data_array

def get_distance( coords1, coords2):
  from geopy import distance
  return distance.geodesic(tuple(coords1), tuple(coords2)).km

class AQDataset(Dataset):
    def __init__(self, station, data, location, config, target_station):
        super().__init__()
        self.data = data # (day, station, feature )
        self.location = location
        self.config = config  
        self.station_list_name = station

        self.target_station_name = target_station
        self.target_station_idx = self.station_list_name.index(target_station)
        
        self.input_len = self.config['input_len']
        self.output_len = self.config['output_len']

        self.adj_matrix = self.get_adjacency_matrix()

        self.target_ft_idx = self.config['input_features'].index(self.config['target_features'][0])

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
        for i,_ in enumerate(self.station_list_name):
            tmp_matrix = []
            for j,_ in enumerate(self.station_list_name):
                dist = get_distance(self.location[i, 1:], self.location[j,1:])
                if i == j:
                    dist += 1
                tmp_matrix.append(dist)
            dist_matrix.append(tmp_matrix)
        return np.array(dist_matrix)
    
    def __getitem__(self, index: int):
        # import pdb; pdb.set_trace()
        # x_temporal = self.data[
        #     index: index +self.input_len, 
        #     self.target_station_idx,
        #     :
        # ]
        
        x_spatial = self.data[
            index : index + self.input_len , :,:
        ]
        y = self.data[
            index + self.input_len: index + self.input_len+ self.output_len, 
            self.target_station_idx,
            self.target_ft_idx # PM2.5
        ]
        G = self.adj_matrix
        return {'x': torch.from_numpy(x_spatial), 
                'y': torch.from_numpy(y), 
                'G': np.array(G, dtype=np.float32) 
            }

    def __len__(self) -> int:
        return self.data.shape[0] - self.input_len - self.output_len +1   

def get_dataloader(station, data, location, config, target_station,train_ratio):
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
        # data=lst_data[:200],
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
        # data=lst_data[int(len_dataset * train_pct): int(len_dataset * train_pct) + 200],
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

if __name__=='__main__':
    config_file = './config/spatio_attention_embedded_rnn.yml'
    with open(config_file) as f:
        config = yaml.safe_load(f)
    transformed_data,  location_, list_station, scaler = get_data_array(config)
    print(transformed_data[0][0])
    target_station = config['data']['target_station'][0]

    train_dataloader, validation_dataloader, test_dataloader = get_dataloader(station=list_station, lst_data=transformed_data, location=location_, config=config, target_station=target_station)

    for i, it in enumerate(train_dataloader):
        x_it = it['x']
        y_it = it['y']
        with open('./log/spatio_attention_embedded_rnn/input.txt' , 'w') as f:
            f.write(x_it)
            f.write(y_it)
        break