import torch
import os
import numpy as np 
import pandas as pd 
import time
from tqdm import tqdm

from models.spattrnn.model import SpatioEmbeddedRNN
from models.spattrnn.utils import get_dataloader
from util.multi_loader import get_data_array, unscale_tensor
from util import model as model_utils
from util.early_stop import EarlyStopping, EarlyStoppingReachAccuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.loss import r2_loss
import os, psutil
from util.torch_util import weight_init

class SAERSupervisor():
    def __init__(self, args, config, target_station, train_ratio, device):
        # Config
        self.device = device 
        self._epochs = config["epochs"]
        self._lr = config["lr"]
        self._patience = config['patience']
        self._optimizer = config['optimizer']
        self._lr_decay = config['lr_decay_ratio']
        self.input_dim = config['input_dim']
        config['input_len'] = args.input_len
        config['output_len'] = args.output_len
        config['num_input_station'] = args.num_input_station
        self.target_station = target_station
        self.config = config

        # Data
        transformed_data,  location_, list_station, scaler = get_data_array(target_station, args, self.config)
        train_dataloader, valid_dataloader, test_dataloader = get_dataloader(station=list_station, 
                                                                                data=transformed_data, 
                                                                                location=location_, 
                                                                                config=config, 
                                                                                target_station=target_station,
                                                                                train_ratio=train_ratio)

        self._scaler = scaler 
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.list_station = list_station
        self.target_station_idx = list_station.index(target_station) 

        # print("Len of test dataloader")
        # print(len(self.valid_dataloader))
        
        # first_it = next(iter(train_dataloader))   
        # X_temporal, X_spatial, y, G = first_it['x_temporal'], first_it['x_spatial'], first_it['y'], first_it['G']

        self._base_dir = model_utils.generate_log_dir(args, config['base_dir'])
        self._weights_path = os.path.join(self._base_dir, "best.pth")
        
        # early stopping 
        self._es = EarlyStopping(
            patience=self.config['patience'],
            verbose=True,
            delta=0.0,
            path=self._weights_path            
        )

        self._es_until_reach_accuracy = EarlyStoppingReachAccuracy(
            patience=self.config['patience'],
            verbose=True,
            delta=0.0,
            path=self._weights_path   
        )

        # Model
        self._model = SpatioEmbeddedRNN(config, device).to(device)
        self._model.apply(weight_init)

        self.train_ratio = train_ratio
        self.stop_until_reach_accuracy = False
        if args.experimental_mode == 'stop_until_reach_accuracy':
            self.stop_until_reach_accuracy = True

    def train(self):
        model = self._model
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.config['lr_decay_ratio'], patience=5)
        # Train the model

        total_train_time = 0.0     
        val_losses = []
        train_losses = []
        train_r2_losses = []

        if self.stop_until_reach_accuracy:
            es = self._es_until_reach_accuracy
        else:
            es = self._es

        for epoch in range(self._epochs):
            if not es.early_stop:
                # train 
                model.train()
                epoch_train_loss = 0 
                epoch_train_r2 =  0 

                for data in tqdm(self.train_dataloader):
                    torch.cuda.synchronize()
                    train_it_start = int(round(time.time()*1000))
                    optimizer.zero_grad()
                    batch_loss = 0 
                    # import pdb; pdb.set_trace()
                    x = data['x'].to(self.device)
                    y = data['y'].to(self.device)
                    G = data['G'].to(self.device)

                    outputs = model(x, G, self.target_station_idx)
                    batch_loss = criterion(outputs, y)
                    batch_r2_loss = r2_loss(outputs, y)

                    batch_loss.backward()
                    optimizer.step()

                    torch.cuda.synchronize()
                    time_elapsed = int(round(time.time()*1000)) - train_it_start
                    total_train_time += time_elapsed
                    epoch_train_loss  += batch_loss.item()
                    epoch_train_r2 += batch_r2_loss.item()

                train_loss = epoch_train_loss / len(self.train_dataloader)
                train_r2_loss = epoch_train_r2 / len(self.train_dataloader)
                train_losses.append(train_loss)
                train_r2_losses.append(train_r2_loss)
                print("Epoch: %d, train_loss: %1.5f" % (epoch, train_loss))
                print("Epoch: %d, train_r2_loss: %1.5f" % (epoch, train_r2_loss))

                # validation 
                model.eval()
                epoch_val_loss = 0  
                with torch.no_grad():
                    for data in tqdm(self.valid_dataloader):
                        batch_loss =  0

                        x = data['x'].to(self.device)
                        y = data['y'].to(self.device)
                        G = data['G'].to(self.device)

                        torch.cuda.synchronize()
                        valid_start = int(round(time.time()*1000))

                        output = model(x, G, self.target_station_idx)

                        torch.cuda.synchronize()
                        batch_valid_time = int(round(time.time()*1000)) - valid_start
                        total_train_time += batch_valid_time

                        batch_loss = criterion(output, y)
                        epoch_val_loss += batch_loss.item()

                    val_loss = epoch_val_loss / len(self.valid_dataloader)

                val_losses.append(val_loss)
                scheduler.step(val_loss)
                print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
                if self.stop_until_reach_accuracy:
                    es(train_r2_loss, model)
                else:
                    es(val_loss, model)

        num_params = sum(p.numel() for p in model.parameters())
        process = psutil.Process(os.getpid())
        mem_used = process.memory_info().rss  / 1048576 # bytes -> mb  

        res = {
            'train_ratio': self.train_ratio,
            'val_losses': val_losses,
            'train_losses': train_losses,
            'train_r2_losses': train_r2_losses, 
            'train_time': total_train_time,
            'num_params': num_params,
            'mem_used': mem_used
        }
        return res
        
    def test(self):
        self._model.load_state_dict(torch.load(self._weights_path)["model_dict"])
        model = self._model
        model.eval()
        
        predict = [] 
        groundtruth = [] 
        lst_inference_time = []
        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                x = data['x'].to(self.device)
                y = data['y'].to(self.device)
                G = data['G'].to(self.device)

                torch.cuda.synchronize()
                inference_time_start = int(round(time.time()*1000))

                output= model(x, G, self.target_station_idx) 
                output, y = output.squeeze(), y.squeeze()
                output, y = unscale_tensor(self.config, output, self._scaler), unscale_tensor(self.config, y, self._scaler)
    
                torch.cuda.synchronize()
                total_inference_time = int(round(time.time()*1000)) - inference_time_start 
                lst_inference_time.append(total_inference_time)

                groundtruth += y.tolist()
                predict += output.tolist()

        test_time = np.sum(np.array(lst_inference_time))
        res = {
            'base_dir': self._base_dir,
            'target_station': self.target_station,
            'groundtruth': groundtruth,
            'predict': predict,
            'test_time': test_time
        }
        return res
