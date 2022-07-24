'''
    Dataloader for single stations 
    get_data_array: get minmax scaled data in numpy form,  scaler
'''
import yaml 

import os 
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import yaml
import os 
import torch
from  torch.utils.data import Dataset, DataLoader 

def get_data_array(config, target_station):
    file_path = config['data_dir']

    file_gauge = file_path + 'gauges/'

    list_input_ft = config['input_features']
    scaler = MinMaxScaler()

    df = pd.read_csv(file_gauge  + f"{target_station}.csv")[list_input_ft]
    arr = df.iloc[:,:].astype(np.float32).values

    scaler.fit(arr)
    transformed_data = scaler.transform(arr)
    return transformed_data,  scaler

def get_dataloader(data, args, config, train_ratio):
    len_dataset = len(data)  
    train_pct = config['train_size']
    valid_pct = config['valid_size']
    test_pct = config['test_size']

    train_data = data[:int(len_dataset * train_pct * train_ratio)]
    valid_data = data[int(len_dataset * (1 - test_pct - valid_pct)): int(len_dataset * (1 - test_pct))]
    test_data = data[int(len_dataset * (1 - test_pct)):]

    if args.data_splitting == 'hold-out':
        train_dataset = AQDataset(
            data= train_data, 
            config=config
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            shuffle=True, 
            drop_last=True
        )
        validation_dataset = AQDataset(
            data=valid_data, 
            config=config
        )
        valid_dataloader = DataLoader(
            validation_dataset,
            batch_size=config['batch_size'],
            shuffle=False, 
            drop_last=True
        )

        test_dataset = AQDataset(
            data=test_data,
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

class AQDataset(Dataset):
    def __init__(self, data, config):
        super().__init__()
        self.data = data 
        self.config = config 
        self.input_len =  self.config['input_len']
        self.output_len = self.config['output_len']

        self.input_feats =  self.config["input_features"]
        self.target_feat = self.config['target_features']
        self.target_ft_idx = self.config['input_features'].index(self.config['target_features'][0])
    
    def __getitem__(self, index: int):
        x = self.data[
            index: index + self.input_len, 
            :
        ]
        y = self.data[
            index + self.input_len: index + self.input_len + self.output_len, 
            self.target_ft_idx
        ]
        return {
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y)
        }

    def __len__(self) -> int:
        return self.data.shape[0]  - self.input_len - self.output_len +1

if __name__ == '__main__':
    config_path = './config/lstm.yml'
    with open(config_path, 'r') as f:
        config= yaml.safe_load(f)
    data, scaler = get_data_array(config) 
    print(data.shape)