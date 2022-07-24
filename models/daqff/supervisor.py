from pkgutil import get_data
from models.daqff.model import Hybrid
import torch
import torch.nn as nn 
from models.daqff.utils import get_dataloader
from util import model as model_utils
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.early_stop import EarlyStopping
from util.multitask_loader import get_data_array, unscale_tensor
from util.loss import r2_loss
import os, psutil
from util.torch_util import weight_init
from util.early_stop import EarlyStopping, EarlyStoppingReachAccuracy
from util import model as model_utils
import time 

class DAQFFSupervisor():
    def __init__(self, args, config, train_ratio, device) :
        self.device = device

        self.learning_rate = config['lr']
        self.num_epochs =config['epochs']
        self._patience = config.get('patience')

        config['input_len'] = args.input_len
        config['output_len'] = args.output_len
        config['num_input_station'] = len(config['target_station'])
        
        self.config = config 

        stations = config['target_station']
        self.stations = stations 

        list_data, self._scaler = get_data_array(stations, args, config)
        self.train_dataloader, self.valid_dataloader, self.test_dataloader= get_dataloader( data=list_data, config=self.config, train_ratio=train_ratio)

        # Dir
        self._base_dir = model_utils.generate_log_dir(args)
        self._weights_path = os.path.join(self._base_dir,"best.pth")
        self._model = Hybrid(config=config, device=device).to(device)
        self._model.apply(weight_init)

        # Data 
        self.criterion = nn.MSELoss()
        self._es = EarlyStopping(
            patience=self._patience,
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

        self.train_ratio = train_ratio
        self.stop_until_reach_accuracy = False
        if args.experimental_mode == 'stop_until_reach_accuracy':
            self.stop_until_reach_accuracy = True

    def train(self):
        config = self.config
        model = self._model
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr'))
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.config['lr_decay_ratio'], patience=5, min_lr=1e-6)

        total_train_time  =0 

        train_losses= []
        train_r2_losses = [] 
        val_losses = []

        if self.stop_until_reach_accuracy:
            es = self._es_until_reach_accuracy
        else:
            es = self._es
        for epoch in range(config.get('epochs')):
            if not es.early_stop:
                # running_loss = 0.0
                model.train()

                epoch_train_loss = 0 
                epoch_train_r2 = 0

                for data in self.train_dataloader:
                    torch.cuda.synchronize()
                    optimizer.zero_grad()
                    train_it_start = int(round(time.time()*1000))
                    x,  y = data['x'].to(self.device), data['y'].to(self.device)
                    out = model(x)
                    # import pdb; pdb.set_trace()
                    batch_loss = self.criterion(out, y)
                    batch_r2_loss =  r2_loss(out, y)

                    batch_loss.backward()
                    optimizer.step()

                    torch.cuda.synchronize()
                    time_elapsed = int(round(time.time()*1000)) - train_it_start
                    total_train_time += time_elapsed

                    epoch_train_loss += batch_loss
                    epoch_train_r2 += batch_r2_loss 
                
                train_loss = epoch_train_loss / len(self.train_dataloader)
                train_r2_loss = epoch_train_r2 / len(self.train_dataloader)
                train_losses.append(train_loss)
                train_r2_losses.append(train_r2_loss)
                print("Epoch: %d, train_loss: %1.5f" % (epoch, train_loss))
                print("Epoch: %d, train_r2_loss: %1.5f" % (epoch, train_r2_loss))

                model.eval()
                epoch_val_loss = 0 
                with torch.no_grad():
                    for  data in self.valid_dataloader:
                        eval_loss = 0 

                        x = data['x'].to(self.device)
                        y = data['y'].to(self.device)
                        torch.cuda.synchronize()
                        valid_start = int(round(time.time()*1000))
                        output = model(x)
                        torch.cuda.synchronize()
                        batch_valid_time = int(round(time.time()*1000)) - valid_start
                        total_train_time += batch_valid_time

                        eval_loss = self.criterion(output, y)
                        epoch_val_loss += eval_loss.item()
                
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

        groundtruth = {}
        predict = {}
        lst_inference_time = []

        model.eval()
        with torch.no_grad():
            for data in self.test_dataloader:
                x, y_ = data['x'].to(self.device), data['y'].to(self.device)

                torch.cuda.synchronize()
                inference_time_start = int(round(time.time()*1000))

                output = model(x)
                output, y_ = output.squeeze(), y_.squeeze()
                y, output = unscale_tensor(self.config, y_, self._scaler), unscale_tensor(self.config, output, self._scaler)
                torch.cuda.synchronize()
                total_inference_time = int(round(time.time()*1000)) - inference_time_start 
                lst_inference_time.append(total_inference_time)
                
                for i in range(y.shape[0]):
                    if self.stations[i] not in groundtruth.keys():
                        groundtruth.update({self.stations[i]: []})
                        predict.update({self.stations[i]: []})
                    groundtruth[self.stations[i]] += y[i, :].tolist()
                    predict[self.stations[i]] += output[i,:].tolist()
        
        test_time = np.sum(np.array(lst_inference_time))
        res = {
            'base_dir': self._base_dir,
            'target_station': '',
            'groundtruth': groundtruth,
            'predict': predict,
            'test_time': test_time
        }
        return res

        # all_pred = torch.cat(list_predict, dim=0)
        # print(all_pred.shape)
        # all_label = torch.cat(list_label, dim=0)
        # print(all_label.shape)
        # predict = all_pred.cpu().detach().numpy()
        # label = all_label.cpu().detach().numpy()
            
        # for idx, station in enumerate(list_station):
        #     label_list = []
        #     predict_list = []
        #     for i in range (n_out):
        #         labeli = scaler.inverse_transform(np.tile(np.expand_dims(label[:, idx, i], axis=1), 15))[:,2]
        #         predicti = scaler.inverse_transform(np.tile(np.expand_dims(predict[:, idx, i], axis=1), 15))[:,2]
        #         label_list.append(np.expand_dims(labeli, axis=1))
        #         predict_list.append(np.expand_dims(predicti, axis=1))
        #     labels = np.concatenate(label_list, axis=1)
        #     predicts = np.concatenate(predict_list, axis=1)
        #     # labels = scaler.inverse_transform(np.tile(label[:, idx, :], 15))[:,2]
        #     # predicts =scaler.inverse_transform(np.tile(predict[:, idx, :], 15))[:,2]
        #     if station in target_station:
        #         save_results(labels, predicts, data_path, station)