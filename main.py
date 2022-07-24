from email.policy import default
import argparse
import torch 
import yaml
from util.helper import seed, model_mapping
from util.model import save_config, save_result
import pandas as pd 

# them tat ca cac tham so muon chinh vao day
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        default=52,
                        type=int,
                        help='Seed')
    parser.add_argument('--model',
                        default='spattrnn',
                        type=str)
    parser.add_argument('--batch_size',
                        type=int)
    parser.add_argument('--lr',
                        type=float)
    parser.add_argument('--lr_decay_ratio',
                        type=float)                        
    parser.add_argument('--hidden_dim',
                        type=int)
    parser.add_argument('--hidden_dim_2',
                        type=int)
    parser.add_argument('--dropout',
                        type=float)
    parser.add_argument('--input_len',
                        type=int)
    parser.add_argument('--output_len',
                        type=int)
    parser.add_argument('--train_ratio',
                        default=1, 
                        type=float)
    parser.add_argument('--target_station',
                        default='all',
                        type=str)
    parser.add_argument('--num_input_station',
                        default=5,
                        type=int)
    parser.add_argument('--station_selection_strategy',
                        default='distance',
                        type=str,
                        choices={"random", "correlation", "distance"})
    parser.add_argument('--data_splitting',
                        default='hold-out',
                        type=str,
                        choices={'hold-out', 'time-series-cross-validation', 'blocking-cross-validation'})
    parser.add_argument('--experimental_mode',
                        default='default',
                        choices={'default', 'change_features', 'stop_until_reach_accuracy','data_splitting', 'train_ratio', 'change_prediction_length', 'change_num_input_station', 'change_station_selecting_strategy'})
    parser.add_argument('--multi_task',
                        default=False,
                        type=bool)
    parser.add_argument('--list_features',
                        default="PM2.5",
                        type=str)
    return parser

if __name__=="__main__":
    parser = parse_args()
    args = parser.parse_args()

    seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    conf = model_mapping(args.model)
    with open(conf['config'], encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.target_station == 'all':
        target_station = config['target_station']
    else:
        target_station = [args.target_station]
        
    if args.experimental_mode=='change_features':
        features = [x for x in args.list_features.split(' ')]
        config['input_dim'] = len(features)
        config['input_features'] = features

    train_ratio = args.train_ratio
    input_length = args.input_len
    output_length = args.output_len
    
    if not args.multi_task:
        for station in target_station:
            print('Currently training station {} with train_ratio {} - input_length {} - output_length {}'.format(station, train_ratio, input_length, output_length))
            model = conf['model'](args, config, station, train_ratio, device)
            res_train = model.train()
            res_test = model.test()
            save_result(args, config, res_train,res_test)
            del model 
    else:
        print('Currently training  with train_ratio {} - input_length {} - output_length {}'.format( train_ratio, input_length, output_length))
        model = conf['model'](args, config, train_ratio, device)
        res_train = model.train()
        res_test = model.test()
        save_result(args, config, res_train,res_test)
        del model
    save_config(config)