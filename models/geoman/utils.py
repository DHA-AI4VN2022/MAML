import os
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from util.multi_loader import get_k_best_stations

DEVICE = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

def get_candidate_stations(config, target_station):
      # assumption: get nearest stations
    data_dir = config['data_dir']
    df =  pd.read_csv(data_dir + "location.csv")
    stations = df['location'].to_list()
    lat = df['latitude'].to_list()
    long = df['longitude'].to_list()
    coord = zip(lat, long)
    coord = dict(zip(stations, coord))
  
    # ts = config['data']['target_station']
    ts = target_station

    stations.remove(ts)
  
    dist = {}
    for station in stations:
        dist[station] = L2norm(coord[station], coord[ts])
    dist = {k:v for k, v in sorted(dist.items(), key = lambda item : item[1])}
    return list(dist.keys())[:config['data']['num_neighbours']]
  
def distance_matrix(config):
  path = config['data_dir'] + 'location.csv'
  df = pd.read_csv(path, index_col = 'location')
  res = df.loc[config['neighbours']].to_numpy(dtype=np.float32)
  dist_mat = [np.exp(L2norm(res[0], res[i])*-1) for i in range(len(res))]
  return np.array(dist_mat).astype('float32')

def get_data(args, config, target_station):
  file_path = config['data_dir']
  file_gauge = file_path + 'gauges/'
  file_location = file_path + 'location.csv'

  list_station = [stat.split('.csv')[0] for stat in os.listdir(file_gauge)]
  num_k_best_stats = args.num_input_station
  selection_strategy = args.station_selection_strategy

  res = {}
  ts = target_station

  data_dir = config['data_dir']
  df_ts = pd.read_csv(data_dir  + 'gauges/' + ts + '.csv')
  res['ext'] = df_ts[config['external_features']].to_numpy(dtype=np.float32)
  
  #get input features for encoder
  stations = get_k_best_stations(file_gauge, file_location, list_station, target_station, num_k_best_stats, selection_strategy)
  res['X'] = []
  for station in stations:
    df = pd.read_csv(data_dir  + 'gauges/' + station + '.csv')
    res['X'].append(df[config['input_features']])
  res['X'] = np.stack(res['X'], 1)

  config['neighbours'] = stations
  res['dist_mat'] = distance_matrix(config)
  config['input_dim'] = res['X'].shape[2]
  config['ext_dim'] = res['ext'].shape[1]
  return stations, res
    
class AQDataset(Dataset):
  def __init__(self, station, data, config, target_station):
    super(AQDataset, self).__init__()
    self.config = config
    self.data = data
    self.len = self.data['X'].shape[0] - \
      self.config['input_len'] - \
      self.config['output_len'] + 1

    self.station_list_name = station
    self.target_station_name = target_station
    self.target_station_idx = self.station_list_name.index(target_station)
      
    self.input_feats =  self.config["input_features"]
    self.target_feat = self.config['target_features'][0]
    self.target_idx = self.input_feats.index(self.target_feat)
    
  def __len__(self):
    return self.len
  
  def __getitem__(self, idx):
    l = idx + self.config['input_len']
    r = l + self.config['output_len']
    x = torch.from_numpy(self.data['X'][idx:l].astype('float32'))
    ext = torch.from_numpy(self.data['ext'][l:r].astype('float32')  )
    y = torch.from_numpy(self.data['X'][l:r, self.target_station_idx, self.target_idx].astype('float32'))
    G = torch.from_numpy(self.data['dist_mat'].astype('float32'))
    return {'x': x, 'y': y, 'G': G, 'ext': ext}

def get_dataloader(station, data, config, target_station, train_ratio):
  len_dataset = data['X'].shape[0]
  train_pct = config['train_size']
  val_pct = config['val_size']
  test_pct = config['test_size']

  train_dataset = AQDataset(station, 
                            {'X':data['X'][:int(len_dataset * train_pct * train_ratio)],
                              'ext':data['ext'][:int(len_dataset * train_pct * train_ratio)],
                              'dist_mat': data['dist_mat']},
                             config, target_station)  
  train_dataloader = DataLoader(train_dataset,
                                batch_size=config['batch_size'],
                                shuffle = True,
                                drop_last = True)

  val_dataset = AQDataset(station,
                            {'X':data['X'][int(len_dataset * (1 - test_pct - val_pct)): int(len_dataset * (1 - test_pct))],
                              'ext':data['ext'][int(len_dataset * (1 - test_pct - val_pct)): int(len_dataset * (1 - test_pct))],
                              'dist_mat': data['dist_mat']},
                             config, target_station)  
  val_dataloader = DataLoader(val_dataset,
                              batch_size=config['batch_size'],
                              shuffle = False,
                              drop_last = True)
  
  test_dataset = AQDataset(station,
                            {'X':data['X'][ int(len_dataset * (1 - test_pct)) : ],
                              'ext':data['ext'][ int(len_dataset * (1 - test_pct)) : ],
                              'dist_mat': data['dist_mat']},
                             config, target_station)  
  test_dataloader = DataLoader(test_dataset,
                                batch_size=config['batch_size'],
                                shuffle = False,
                                drop_last = True)                                                         
  return train_dataloader, val_dataloader, test_dataloader

def L2norm(x1, x2):
  return np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)

# def distance_matrix(config, stations):
#   path = config['data_dir'] + 'location.csv'
#   df = pd.read_csv(path, index_col = 'location')
#   res = df.loc[stations].to_numpy(dtype=np.float32)
#   dist_mat = [np.exp(L2norm(res[0], res[i])*-1) for i in range(len(res))]
#   return torch.FloatTensor(dist_mat).to(DEVICE) 

def scale_data(data, config):
  scaler = {}

  ori_shape_X = data['X'].shape
  ori_shape_ext = data['ext'].shape

  data['X'] = np.reshape(data['X'], (-1, config['input_dim']))
  data['ext'] = np.reshape(data['ext'], (-1, config['ext_dim']))
  
  scaler['X'] = MinMaxScaler(copy=False)
  scaler['ext'] = MinMaxScaler(copy=False)
  
  scaler['X'].fit_transform(data['X'])
  scaler['ext'].fit_transform(data['ext'])  

  data['X'] = np.reshape(data['X'], ori_shape_X)
  data['ext'] = np.reshape(data['ext'], ori_shape_ext)
  return scaler

def read_config():
  config_path = './config/geoman.yaml'
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
  return config

if __name__ == '__main__':
  config_path = './config/geoman.yml'
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
  res = get_data(config)
  # print(res['X'].shape, res['ext'].shape)
  scaler = scale_data(res, config)
  # print(res['X'].shape, res['ext'.shape])
  dist_mat = distance_matrix(config)
  # print(dist_mat.shape)
  dataloader = get_dataloader(res, config)
  for it in dataloader['train']:
      print(it['X'].shape)
      print(it['y'].shape)
      print(it['ext'].shape)
      break


# def get_data(config):
#   data_dir = config['data_dir']
#   read_dir = data_dir + 'gauges/'
#   res = {}
#   X_l = []
#   ext_l = []

#   path =  data_dir +  'poi/POI.csv'
#   df1 = pd.read_csv(path, index_col='POI')
 
  
#   for station in config['data']['stations']:
#     path = read_dir + station + '.csv'
#     df = pd.read_csv(path)
    
#     X = df[config['data']['input_features']].to_numpy(dtype=np.float32)
    
    
#     ext_lst = []
#     if config['data']['ext_mask'][0] == 1:
#       ext_lst += config['data']['ext_feat']
#     if config['data']['ext_mask'][1] == 1:
#       ext_lst += ['Hour', 'Day', 'Month']
#     ext = df[ext_lst].to_numpy(dtype=np.float32)
#     if config['data']['ext_mask'][2] == 1:
#       x = df1[station].to_numpy(dtype=np.float32)
#       x = np.expand_dims(x.ravel()[np.flatnonzero(x)], 0)
#       x = np.repeat(x, X.shape[0], 0)
#       ext = np.stack([ext, x], 0)
    
#     config['model']['ext_dim'] = ext.shape[1] 
#     config['model']['inp_dim'] = X.shape[1]
#     X_l.append(X)
#     ext_l.append(ext)
  
#   res['X'] = np.stack(X_l, 1)
#   res['ext'] = np.stack(ext_l, 1)
#   return res
