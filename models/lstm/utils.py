# import numpy as np 
# import pandas as pd 
# import torch
# from  torch.utils.data import Dataset, DataLoader 
# import  yaml
# from sklearn.preprocessing import MinMaxScaler

# def get_data_array(config, target_station):
#     file_path = config['data_dir']
#     file_gauge = file_path + 'gauges/'
#     list_input_ft = config['input_features']
#     scaler = MinMaxScaler()
#     df = pd.read_csv(file_gauge  + f"{target_station}.csv")[list_input_ft]
#     arr = df.iloc[:,:].astype(np.float32).values
#     scaler.fit(arr)
#     transformed_data = scaler.transform(arr)
#     return transformed_data,  scaler

# if __name__ == '__main__':
#     config_path = './config/lstm.yml'
#     with open(config_path, 'r') as f:
#         config= yaml.safe_load(f)
#     data, scaler = get_data_array(config) 

#     train_dataloader, valid_dataloader, test_dataloader = get_dataloader(data, config)

#     for it in train_dataloader:
#         print(it['x'].shape)
#         print(it['y'].shape)
#         break
    