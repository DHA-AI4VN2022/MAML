# default 
python main.py --model=encoder_decoder --input_len=24 --output_len=1 --target_station=all
# train until reach accuracy
python main.py --model=encoder_decoder --input_len=24 --output_len=1 --target_station=房山 --experimental_mode=stop_until_reach_accuracy
# train change train_ratio
python main.py --model=encoder_decoder --input_len=24 --output_len=1 --train_ratio=0.2 --target_station=房山 --experimental_mode=train_ratio
python main.py --model=encoder_decoder --input_len=24 --output_len=1 --train_ratio=0.4 --target_station=房山 --experimental_mode=train_ratio
python main.py --model=encoder_decoder --input_len=24 --output_len=1 --train_ratio=0.6 --target_station=房山 --experimental_mode=train_ratio
python main.py --model=encoder_decoder --input_len=24 --output_len=1 --train_ratio=0.8 --target_station=房山 --experimental_mode=train_ratio
python main.py --model=encoder_decoder --input_len=24 --output_len=1 --train_ratio=1 --target_station=房山 --experimental_mode=train_ratio
# train change prediction_length 
python main.py --model=encoder_decoder --input_len=24 --output_len=1  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=encoder_decoder --input_len=24 --output_len=3  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=encoder_decoder --input_len=24 --output_len=6  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=encoder_decoder --input_len=24 --output_len=12  --target_station=房山 --experimental_mode=change_prediction_length
python main.py --model=encoder_decoder --input_len=24 --output_len=24  --target_station=房山 --experimental_mode=change_prediction_length
