import pandas as pd 
import numpy as np 

from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import torch 
import torch.nn as nn 

def get_config(model_type):
    config_path = './config/'
    return config_path + model_type + '.yml'

def to_numeric(x):
    x_1 = x.apply(pd.to_numeric, errors="coerce")
    res = x_1.clip(lower=0)
    return res

def fill_na(x):
    res= x.ffill()
    res.fillna(0, inplace=True)
    return res 

def clipping(x):
    min_clip = 0.0
    max_clip = 0.95
    ans = x.transform(lambda x: np.clip(x, x.quantile(min_clip), x.quantile(max_clip) ) )
    return ans

def get_k_best_stations(base_dir, list_stations, target_station, k:int):
    # for path in glob.glob(os.path.join(folder, '*.csv')):
    df_aqme = {}
    for station in list_stations:
        if station != '':
            file_name = base_dir + "{}.csv".format(station)
            #if file_name.split('.')[0] != 'location':
            # import pdb; pdb.set_trace()
            df_aqme[station] = pd.read_csv(file_name)
    #Lấy chỉ số PM2.5 của tất cả các trạm và đánh giá hệ số tương quan
    pm2_5_all = pd.DataFrame()
    for station in list_stations:
        pm2_5_all[station] = df_aqme[station]['PM2.5']    
    station_corr = pm2_5_all.corr(method = 'pearson')[target_station]
    # import pdb; pdb.set_trace()
    lst_k_best_stations = station_corr.sort_values(by=target_station ,ascending=False).index[0: k].to_numpy().tolist()
    return lst_k_best_stations

def preprocess_pipeline(df, type):
    lst_cols = list(set(list(df.columns)) - set(['Hour','Day','Month','Year']))
    type_transformer = FunctionTransformer(to_numeric)
    fill_transformer = FunctionTransformer(fill_na)
    clipping_transformer = FunctionTransformer(clipping)
    num_pl = Pipeline(
        steps=[
            ("fill_na", fill_transformer),
            ("numeric_transform", type_transformer),
            ("clipping", clipping_transformer),
        ],
    )

    preprocessor = ColumnTransformer(transformers=[("num", num_pl, lst_cols)])
    res = preprocessor.fit_transform(df)
 
    trans_df = pd.DataFrame(res, columns=lst_cols)
    trans_df[['Hour','Day','Month','Year']] = df[['Hour','Day','Month','Year']]
    
    lst_meteo_cols = ['PM2.5', 'Mean','AQI','PM10', 'CO', 'NO2', 'O3', 'SO2', 'prec', 'lrad', 'shum', 'pres', 'temp', 'wind', 'srad']

    final_lst_cols  = ['Hour','Day','Month','Year'] + lst_meteo_cols
    trans_df = trans_df[final_lst_cols]
    trans_df.reset_index(drop=True, inplace=True)
    return trans_df

# if __name__=='__main__':
#     model = 'spatio_attention_embedded_rnn'

#     data_folder =  './data/Beijing/gauges/'

#     out_folder =  './data/Beijing/gauges_processed/'

#     lst_stations = os.listdir(data_folder)

#     for stat in lst_stations:
#         df = pd.read_csv(data_folder + stat, index_col=0)
#         trans_df = preprocess_pipeline(df, model)
#         trans_df.to_csv(out_folder + stat, index=False)


if __name__ =='__main__':
    list_stations = ['北部新区',"丰台花园",'东高村','古城','大兴','南三环','前门','榆垡','琉璃河','永乐店']
    base_dir = '/home/aiotlabws/Workspace/Project/hungvv/Air-Quality-Prediction-Benchmark/data/Beijing/gauges_processed/'
    target_station = '北部新区'

    print(get_k_best_stations(base_dir, list_stations, target_station, 5))