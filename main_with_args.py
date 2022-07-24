from email.policy import default
import argparse
import torch 
import yaml
import wandb

from util.helper import seed, model_mapping
from util.wb_loader import get_wandb_instance

# them tat ca cac tham so muon chinh vao day
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        type=str,
                        default='train')
    parser.add_argument('--seed',
                        default=52,
                        type=int,
                        help='Seed')
    parser.add_argument('--model',
                        type=str)
    parser.add_argument('--batch_size',
                        type=int)
    parser.add_argument('--model_lr',
                        type=float)
    parser.add_argument('--task_lr',
                        type=float)
    parser.add_argument('--lr_decay',
                        type=float)
    parser.add_argument('--weight_decay',
                        type=float)
    parser.add_argument('--input_len',
                        type=int)
    parser.add_argument('--output_len',
                        type=int)
    parser.add_argument('--train_ratio',
                        type=float,
                        default=1)
    parser.add_argument('--experimental_mode',
                        default='default',
                        choices={'default', 'change_features', 'stop_until_reach_accuracy','data_splitting', 'train_ratio', 'change_prediction_length', 'change_num_input_station', 'change_station_selecting_strategy'})
    parser.add_argument('--multi_task',
                        default=True,
                        type=bool)

    parser.add_argument('--data_splitting',
                        default='hold-out',
                        type=str,
                        choices={'hold-out', 'time-series-cross-validation', 'blocking-cross-validation'})



    return parser

if __name__=="__main__":
    parser = parse_args()
    args = parser.parse_args()

    seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = model_mapping(args.model)
    with open(conf['config'], encoding = 'utf-8') as f:
        config = yaml.safe_load(f)



    # target_station = config['data']['target_station'][0]
    # config_wandb = {
    #     'epochs': config["train"]["epochs"],
    #     'patience': config['train']['patience'],
    #     'optimizer': config['train']['optimizer'],
    #     'input_len': config['model']['input_len'],
    #     'output_len': config['model']['output_len'],
    #     'train_size': config['data']['train_size'],
    #     'valid_size': config['data']['valid_size'],
    #     'batch_size': config['data']['batch_size'],
    #     'data_dir': config['data']['data_dir'],
    #     'input_features': config['data']['input_features'],
    #     'target_features': config['data']['target_features'],
    #     'num_layers': config['model']['num_layers'],
    #     'input_dim': config['model']['input_dim'],
    #     'output_dim': config['model']['output_dim'],
    #     'hidden_dim': config['model']['hidden_dim'],
    #     # 'hidden_dim_2': config['model']['hidden_dim_2'],
    #     'lr': config['train']['lr'],
    #     'lr_decay_ratio': config['train']['lr_decay_ratio'],
    #     'activation': config['model']['activation'],
    #     # 'rnn_type':  config['model']['rnn_type'],
    #     'dropout': config['train']['dropout'],
    #     'alpha': config['train']['alpha'],
    #     # 'nan_station': config['data']['nan_station'],
    #     'input_features': config['data']['input_features']
    # }
    run, config_wandb = get_wandb_instance(config, args)
    print(config_wandb)
    # test voi 1 tram 
    # for station in target_station:
    # station = target_station
    model = conf['model'](args, config_wandb, device)
    val_loss= model.train()
    test_loss = model.test()
    run.log({"val_loss": val_loss})

    # # test voi nhieu tram
    # model = conf['model'](args, config, )
    # if args.mode == 'train':
    #     model.train()
    # elif args.mode == 'test':
    #     model.test()
    # else:
    #     raise RuntimeError("Mode needs to be train/evaluate/test!")