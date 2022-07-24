'''
    Dataloader for multiple stations 
    get_data_array: get minmax scaled data in numpy form, location numpy array, list of station names,  scaler
    get_distance: distance between 2 coordinations
'''
import yaml 
import os 
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import yaml
import os 
from util.loader import get_k_best_stations
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random 

def unscale_tensor(config, tensor, scaler):
    unscaled_tensor =  np.zeros(shape = tensor.shape)
    sequence = tensor
    sequence = sequence.cpu().detach().numpy()
    sequence = sequence.reshape(-1, 1)
    padded_sequence = np.zeros(shape = (sequence.shape[0], config['input_dim']) )
    padded_sequence[:,0] = sequence[:,0]
    unscaled_sequence = scaler.inverse_transform(padded_sequence)[:,0]

    return unscaled_sequence

def get_poi_data_array(config):
    file_path = config['data_dir']
    file_poi = file_path + 'poi/POI.csv'

    poi_df = pd.read_csv(file_poi)
    arr = poi_df.iloc[:,1:].astype(np.float32).values
    return arr 

def get_data_array(target_station, args,config):
    file_path = config['data_dir']

    file_gauge = file_path + 'gauges/'
    file_location = file_path + 'location.csv'
    list_station = [stat.split('.csv')[0] for stat in os.listdir(file_gauge)]

    # print(list_station)

    num_k_best_stats = args.num_input_station
    selection_strategy = args.station_selection_strategy

    list_k_stations = get_k_best_stations(file_gauge, file_location, list_station, target_station, num_k_best_stats, selection_strategy)
    list_input_ft = config['input_features']

    location_df = pd.read_csv(file_location)
    scaler = MinMaxScaler()
    location_ = []
    list_arr = []

    for stat in list_k_stations:
        row_stat = location_df[location_df['location'] == stat]  # name, lat, lon
        location_it = row_stat.values[:, [0, 2, 1]]
        location_.append(location_it)

        df = pd.read_csv(file_gauge + f"{stat}.csv")
        df_ = df[list_input_ft]
        arr = df_.iloc[:, :].astype(np.float32).values
        list_arr.append(arr)
    num_ft = list_arr[0].shape[-1]  # 14
    list_arr = np.concatenate(list_arr, axis=0)  # 1430 * 28, 9
    # transformed_data = np.transpose(np.array(list_arr), (1,0,2))

    scaler.fit(list_arr)
    transformed = scaler.transform(list_arr)

    transformed_data = transformed.reshape(len(list_k_stations), -1, num_ft)  # 33, 8642, 14
    transformed_data = np.transpose(transformed_data, (1, 0, 2))

    location_ = np.concatenate(location_, axis=0)
    return transformed_data, location_, list_k_stations, scaler

def get_k_best_stations(dir_gauge, dir_location, list_stations, target_station, k: int,selection_strategy='correlation'):
    df_aqme = {}
    if selection_strategy == 'correlation':
        for station in list_stations:
            if station != '':
                file_name = dir_gauge + "{}.csv".format(station)
                # if file_name.split('.')[0] != 'location':
                # import pdb; pdb.set_trace()
                df_aqme[station] = pd.read_csv(file_name)
        # Lấy chỉ số PM2.5 của tất cả các trạm và đánh giá hệ số tương quan
        pm2_5_all = pd.DataFrame()
        for station in list_stations:
            pm2_5_all[station] = df_aqme[station]['PM2.5']
        station_corr = pm2_5_all.corr(method='pearson')[target_station]
        # import pdb; pdb.set_trace()
        lst_k_best_stations = station_corr.sort_values(ascending=False).index[0: k].to_numpy().tolist()
    elif selection_strategy == 'random':
        lst_stations  = [ stat for stat in list_stations if stat not in ['', target_station]]
        stations = random.choices(lst_stations, k=k-1)
        lst_k_best_stations = [target_station] + stations
    elif selection_strategy == 'distance':
        lst_stations  = [ stat for stat in list_stations if stat != '']
        df_lat_lon = pd.read_csv(dir_location)
        lat = df_lat_lon['latitude'].to_list()
        long = df_lat_lon['longitude'].to_list()
        coord = zip(lat, long)
        coord = dict(zip(lst_stations, coord))
    
        ts = target_station
        lst_stations.remove(ts)
    
        dist = {}
        for station in lst_stations:
            dist[station] = get_distance(coord[station], coord[ts])
        dist = {u:v for u, v in sorted(dist.items(), key = lambda item : item[1])}
        best_stations=  list(dist.keys())[:k-1]
        lst_k_best_stations = [ts] + best_stations
    return lst_k_best_stations

def get_distance(coords1, coords2):
    from geopy import  distance
    return distance.geodesic(tuple(coords1), tuple(coords2)).km

if __name__ == '__main__':
    config_file = './config/lstm.yml'
    with open(config_file) as f:
      conf = yaml.safe_load(f)

    arr = get_poi_data_array(conf)
    print(arr.shape)