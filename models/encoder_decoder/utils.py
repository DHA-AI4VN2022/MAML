import numpy as np 
import pandas as pd 
import torch
from  torch.utils.data import Dataset, DataLoader 
import  yaml
# class AQDataset(Dataset):
#     def __init__(self, data, config):
#         super().__init__()
#         self.data = data 
#         self.config = config 
#         self.input_len =  self.config['input_len']
#         self.output_len = self.config['output_len']

#         self.input_feats =  self.config["input_features"]
#         self.target_feat = self.config['target_features']
    
#     def __getitem__(self, index: int):
#         x = self.data[
#             index: index + self.input_len, 
#             :
#         ]
#         y = self.data[
#             index + self.input_len: index + self.input_len + self.output_len, 
#             0
#         ]
#         return {
#             'x': torch.from_numpy(x),
#             'y': torch.from_numpy(y)
#         }

#     def __len__(self) -> int:
#         return self.data.shape[0]  - self.input_len - self.output_len -2 

# def get_dataloader(data, config):
#     len_dataset = len(data)
#     train_pct = config['train_size']
#     valid_pct = config['valid_size']
    
#     train_dataset = AQDataset(
#         data= data[:int(len_dataset * train_pct)], 
#         config=config
#     )
#     train_dataloader = DataLoader(
#         train_dataset, 
#         batch_size=config['batch_size'],
#         shuffle=False, 
#         drop_last=True
#     )

#     validation_dataset = AQDataset(
#         data=data[int(len_dataset * train_pct): int(len_dataset * (train_pct+valid_pct))], 
#         config=config
#     )
#     valid_dataloader = DataLoader(
#         validation_dataset,
#         batch_size=config['batch_size'],
#         shuffle=False, 
#         drop_last=True
#     )

#     test_dataset = AQDataset(
#         data=data[int(len_dataset * (train_pct+valid_pct)): ],
#         config=config
#     )
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=config['batch_size'],
#         shuffle=False, 
#         drop_last=True
#     )
#     return train_dataloader, valid_dataloader, test_dataloader

# def data_loader(config):
#     dataset_path = config["data"]["path"]
#     seq_len = config["data"]["seq_len"]
#     horizon = config["data"]["horizon"]
#     input_features = config["data"]["input_features"]
#     target_features = config["data"]["target_features"]
#     train_size = config["data"]["train_size"]
#     valid_size = config["data"]["valid_size"]
#     input_data = pd.read_csv(dataset_path, usecols=input_features)
#     output_data = pd.read_csv(dataset_path, usecols=target_features)
#     input_data = input_data.to_numpy()
#     output_data = output_data.to_numpy()
    
#     train_len = int(len(input_data) * train_size)
#     valid_len = int(len(input_data) * (train_size + valid_size))
#     X_train_data = input_data[:train_len]
#     X_valid_data = input_data[train_len:valid_len]
#     X_test_data = input_data[valid_len:]

#     y_train_data = output_data[:train_len]
#     y_valid_data = output_data[train_len:valid_len]
#     y_test_data = output_data[valid_len:]

#     X_train = np.array([X_train_data[i:i+seq_len] for i in range(0, len(X_train_data) - seq_len - horizon)])
#     X_valid = np.array([X_valid_data[i:i+seq_len] for i in range(0, len(X_valid_data) - seq_len - horizon)])
#     X_test = np.array([X_test_data[i:i+seq_len] for i in range(0, len(X_test_data) - seq_len - horizon)])

#     y_train = np.array([y_train_data[i+seq_len: i+seq_len+horizon] for i in range(0, len(y_train_data) - seq_len - horizon)])
#     y_valid = np.array([y_valid_data[i+seq_len: i+seq_len+horizon] for i in range(0, len(y_valid_data) - seq_len - horizon)])
#     y_test = np.array([y_test_data[i+seq_len: i+seq_len+horizon] for i in range(0, len(y_test_data) - seq_len - horizon)])

#     return torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(X_valid), torch.FloatTensor(y_valid), torch.FloatTensor(X_test), torch.FloatTensor(y_test)

from sklearn.preprocessing import MinMaxScaler

def get_data_array(config):
    file_path = config['data']['data_dir']

    file_gauge = file_path + 'gauges_processed/'

    target_station = config['data']['target_station']
    
    list_input_ft = config['data']['input_features']
    scaler = MinMaxScaler()

    df = pd.read_csv(file_gauge  + f"{target_station}.csv")[list_input_ft]
    arr = df.iloc[:,:].astype(np.float32).values

    scaler.fit(arr)
    transformed_data = scaler.transform(arr)
    return transformed_data,  scaler

if __name__ == '__main__':
    config_path = './config/encoder_decoder_lstm.yml'
    with open(config_path, 'r') as f:
        config= yaml.safe_load(f)
    data, scaler = get_data_array(config) 

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(data, config)

    for it in train_dataloader:
        print(it['x'].shape)
        print(it['y'].shape)
        break
    