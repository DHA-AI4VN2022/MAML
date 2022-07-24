# default 
python main.py --model=daqff --input_len=48 --output_len=1 --multi_task=True
# train until reach accuracy
python main.py --model=daqff --input_len=48 --output_len=1 --multi_task=True  --experimental_mode=stop_until_reach_accuracy
# train change train_ratio
python main.py --model=daqff --input_len=48 --train_ratio=0.2 --multi_task=True --experimental_mode=train_ratio
python main.py --model=daqff --input_len=48 --train_ratio=0.4 --multi_task=True --experimental_mode=train_ratio
python main.py --model=daqff --input_len=48 --train_ratio=0.6 --multi_task=True --experimental_mode=train_ratio
python main.py --model=daqff --input_len=48 --train_ratio=0.8 --multi_task=True --experimental_mode=train_ratio
python main.py --model=daqff --input_len=48 --train_ratio=1 --multi_task=True --experimental_mode=train_ratio
#train change prediction_length 
python main.py --model=daqff --input_len=48 --output_len=1  --multi_task=True --experimental_mode=change_prediction_length
python main.py --model=daqff --input_len=48 --output_len=3  --multi_task=True --experimental_mode=change_prediction_length
python main.py --model=daqff --input_len=48 --output_len=6  --multi_task=True --experimental_mode=change_prediction_length
python main.py --model=daqff --input_len=48 --output_len=12 --multi_task=True  --experimental_mode=change_prediction_length
python main.py --model=daqff --input_len=48 --output_len=24 --multi_task=True  --experimental_mode=change_prediction_length
