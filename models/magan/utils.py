import numpy as np 
import pandas as pd 
import torch
from  torch.utils.data import Dataset, DataLoader 
import  yaml
from sklearn.preprocessing import MinMaxScaler

class AQDataset(Dataset):
    def __init__(self, data, config):
        super().__init__()
        self.data = data 
        self.config = config 
        self.input_len =  self.config['window_size']
        self.output_len = self.config['output_size']

        self.input_feats =  self.config["input_features"]
        self.target_feat = self.config['target_features']
    
    def __getitem__(self, index: int):
        x = self.data[
            index: index + self.input_len, 
            :
        ]
        y = self.data[
            index + self.input_len: index + self.input_len + self.output_len, 
            0
        ]
        target = self.data[
            index: index + self.input_len, 
            0
        ]
        next = self.data[
            index + self.input_len: index + self.input_len + self.output_len,
            :
        ]
        return {
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
            'target': torch.from_numpy(target),
            'next': torch.from_numpy(next)
        }

    def __len__(self) -> int:
        return self.data.shape[0]  - self.input_len - self.output_len -2 

def get_dataloader(data, config):
    len_dataset = len(data)
    train_pct = config['train_size']
    valid_pct = config['valid_size']
    
    train_dataset = AQDataset(
        data= data[:int(len_dataset * train_pct)], 
        config=config
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        drop_last=True
    )

    validation_dataset = AQDataset(
        data=data[int(len_dataset * train_pct): int(len_dataset * (train_pct+valid_pct))], 
        config=config
    )
    valid_dataloader = DataLoader(
        validation_dataset,
        batch_size=config['batch_size'],
        shuffle=False, 
        drop_last=True
    )

    test_dataset = AQDataset(
        data=data[int(len_dataset * (train_pct+valid_pct)): ],
        config=config
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False, 
        drop_last=True
    )
    return train_dataloader, valid_dataloader, test_dataloader

def get_data_array(config, target_station):
    file_path = config['data_dir']
    file_gauge = file_path + 'gauges_processed/'
    # target_station = config['data']['target_station']
    list_input_ft = config['input_features']
    scaler = MinMaxScaler()
    df = pd.read_csv(file_gauge  + f"{target_station}.csv")[list_input_ft]
    arr = df.iloc[:,:].astype(np.float32).values
    # import pdb; pdb.set_trace()
    scaler.fit(arr)
    transformed_data = scaler.transform(arr)
    return transformed_data,  scaler

if __name__ == '__main__':
    config_path = './config/magan.yml'
    with open(config_path, 'r') as f:
        config= yaml.safe_load(f)
    data, scaler = get_data_array(config) 

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(data, config)

    for it in train_dataloader:
        print(it['x'].shape)
        print(it['y'].shape)
        break