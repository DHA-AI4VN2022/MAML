# default 
python main.py --model=lstm --input_len=48 --output_len=1 --target_station=all
train until reach accuracy
python main.py --model=lstm --input_len=48 --output_len=1 --target_station=房山 --experimental_mode=stop_until_reach_accuracy
train change train_ratio
python main.py --model=lstm --input_len=48 --output_len=1 --train_ratio=0.2 --target_station=all --experimental_mode=train_ratio
python main.py --model=lstm --input_len=48 --output_len=1 --train_ratio=0.4 --target_station=all --experimental_mode=train_ratio
python main.py --model=lstm --input_len=48 --output_len=1 --train_ratio=0.6 --target_station=all --experimental_mode=train_ratio
python main.py --model=lstm --input_len=48 --output_len=1 --train_ratio=0.8 --target_station=all --experimental_mode=train_ratio
python main.py --model=lstm --input_len=48 --output_len=1 --train_ratio=1 --target_station=房山 --experimental_mode=train_ratio
train change prediction_length 
python main.py --model=lstm --input_len=24 --output_len=1  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=lstm --input_len=24 --output_len=3  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=lstm --input_len=24 --output_len=6  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=lstm --input_len=24 --output_len=12  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=lstm --input_len=24 --output_len=24  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=lstm --experimental_mode=change_features --list_features="PM2.5 temp" --input_len=48 --output_len=1 --target_station=all
python main.py --model=lstm --list_features="PM2.5 temp wind" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 SO2" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 SO2 NO2" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 SO2 NO2 CO" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 SO2 NO2 CO PM10" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 SO2 NO2 CO PM10 AQI" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 NO2" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 NO2 wind" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 PM10" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 PM10 wind" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 NO2 wind AQI" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 NO2 wind O3" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 NO2 wind pres" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 NO2 wind shum" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
python main.py --model=lstm --list_features="PM2.5 NO2 wind AQI shum" --input_len=48 --output_len=1 --target_station=all --experimental_mode=change_features
