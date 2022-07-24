# default 
python main.py --model=geoman --input_len=7 --output_len=1 --target_station=all
# train until reach accuracy
python main.py --model=geoman --input_len=7 --output_len=1 --target_station=房山 --experimental_mode=stop_until_reach_accuracy
# train change train_ratio
python main.py --model=geoman --input_len=7 --output_len=1 --train_ratio=0.2 --target_station=房山 --experimental_mode=train_ratio
python main.py --model=geoman --input_len=7 --output_len=1 --train_ratio=0.4 --target_station=房山 --experimental_mode=train_ratio
python main.py --model=geoman --input_len=7 --output_len=1 --train_ratio=0.6 --target_station=房山 --experimental_mode=train_ratio
python main.py --model=geoman --input_len=7 --output_len=1 --train_ratio=0.8 --target_station=房山 --experimental_mode=train_ratio
python main.py --model=geoman --input_len=7 --output_len=1 --train_ratio=1 --target_station=房山 --experimental_mode=train_ratio
#train change prediction_length 
python main.py --model=geoman --input_len=7 --output_len=1  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=geoman --input_len=7 --output_len=3  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=geoman --input_len=7 --output_len=6  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=geoman --input_len=7 --output_len=12  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=geoman --input_len=7 --output_len=24  --target_station=房山 --experimental_mode=change_prediction_length
# train change num_input_station
python main.py --model=geoman --input_len=7 --output_len=1  --target_station=房山 --num_input_station=1 --experimental_mode=change_num_input_station
python main.py --model=geoman --input_len=7 --output_len=1  --target_station=房山 --num_input_station=5 --experimental_mode=change_num_input_station
python main.py --model=geoman --input_len=7 --output_len=1  --target_station=房山 --num_input_station=10 --experimental_mode=change_num_input_station
python main.py --model=geoman --input_len=7 --output_len=1  --target_station=房山 --num_input_station=15 --experimental_mode=change_num_input_station
python main.py --model=geoman --input_len=7 --output_len=1  --target_station=房山 --num_input_station=20 --experimental_mode=change_num_input_station
python main.py --model=geoman --input_len=7 --output_len=1  --target_station=房山 --num_input_station=25 --experimental_mode=change_num_input_station
python main.py --model=geoman --input_len=7 --output_len=1  --target_station=房山 --num_input_station=30 --experimental_mode=change_num_input_station
# train change station selection method 
python main.py --model=geoman --input_len=7 --output_len=1  --target_station=房山 --num_input_station=5 --station_selection_strategy=distance --experimental_mode=change_station_selecting_strategy
python main.py --model=geoman --input_len=7 --output_len=1  --target_station=房山 --num_input_station=5 --station_selection_strategy=correlation --experimental_mode=change_station_selecting_strategy
python main.py --model=geoman --input_len=7 --output_len=1  --target_station=房山 --num_input_station=5 --station_selection_strategy=random --experimental_mode=change_station_selecting_strategy