import yaml
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np

from util.multi_loader import get_data_array

def get_dataloader(data, config, train_ratio):
    len_dataset = len(data)

    train_pct = config['train_size']
    valid_pct = config['valid_size']
    test_pct = config['test_size']
    # import pdb; pdb.set_trace()

    train_data = data[:int(len_dataset * train_pct * train_ratio)]
    valid_data = data[int(len_dataset * (1 - test_pct - valid_pct)): int(len_dataset * (1 - test_pct))]
    test_data = data[int(len_dataset * (1 - test_pct)):]
    # training set
    train_dataset = AQDataset(
        data= train_data, 
        config= config 
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size = config['batch_size'], shuffle = False, drop_last = True
    )
    # valid set
    valid_dataset = AQDataset(
        data= valid_data,
        config=config  
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size = config['batch_size'], shuffle = False, drop_last = True)

    # test set
    test_dataset = AQDataset(
        data = test_data,
        config=config
    )
    test_dataloader = DataLoader(test_dataset, batch_size = config['batch_size'], shuffle = False, drop_last = True)
    return train_dataloader, valid_dataloader, test_dataloader

class AQDataset(Dataset):
    def __init__(self, data, config):
        super().__init__()
        self.data = data
        self.input_len = config['input_len']
        self.output_len = config['output_len']
        self.lst_input_ft = config['input_features']
        self.target_ft = config['target_features']
        self.target_ft_idx = config['input_features'].index(config['target_features'][0])

    def __getitem__(self, index: int):
        x = self.data[index : index + self.input_len, : , :]
        x_tran = np.transpose(x, (1, 2, 0))
        y = self.data[index + self.input_len : index + self.input_len + self.output_len, : , self.target_ft_idx]
        y_tran = np.transpose(y, (1, 0))
        return {
            'x' : torch.from_numpy(x_tran),
            'y' : torch.from_numpy(y_tran)
        }
    def __len__(self) -> int:
        return self.data.shape[0] - self.input_len - self.output_len +1
