import yaml
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.gc_lstm.utils import get_dataloader, save_results, save_checkpoint, load_checkpoint, \
    mean_absolute_percentage_error, root_mean_squared_error, plot_results
from util.multi_loader import get_data_array, unscale_tensor
from models.gc_lstm.model import GC_LSTM
from util.loss import r2_loss
import os, psutil
from util.torch_util import weight_init
from util.early_stop import EarlyStopping, EarlyStoppingReachAccuracy
from util import model as model_utils

class GCLSTMSupervisor():
    def __init__(self, args, config, target_station, train_ratio, device):

        # self.predictions_dir = self.config['predictions_dir']
        # self.plot_dir = self.config['plot_dir']
        # self.metrics_dir = self.config['metrics_dir']
        # self.model_dir = self.config['model_dir']
        self.device = device
        
        self.learning_rate = config['lr']
        self.num_epochs =config['epochs']
        self.patience = config['patience']
        config['input_len'] = args.input_len
        config['output_len'] = args.output_len
        config['num_input_station'] = args.num_input_station
        self.config = config
        self.target_station = target_station

        # Data 

        list_data,location,list_station,scaler = get_data_array(target_station, args, self.config)
        train_dataloader, valid_dataloader, test_dataloader= get_dataloader(station=list_station, data=list_data, location=location,config=self.config,target_station=target_station, train_ratio=train_ratio)
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self._scaler = scaler 

        self._base_dir = model_utils.generate_log_dir(args, config['base_dir'])
        self._weights_path = os.path.join(self._base_dir, "best.pth")

        # model
        self._model = GC_LSTM(self.config, device=device).to(device)
        self._model.apply(weight_init)

        self.criterion = nn.MSELoss()
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
        self.train_ratio = train_ratio
        self.stop_until_reach_accuracy = False
        if args.experimental_mode == 'stop_until_reach_accuracy':
            self.stop_until_reach_accuracy = True

    # The function to train the model
    def train(self):
        model = self._model

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.config['lr_decay_ratio'])

        # Declare the training hyper parameters
        patience = self.patience
        device = self.device
        criterion = self.criterion

        # Train the model
        total_train_time = 0.0

        val_losses = []
        train_losses = []
        train_r2_losses = []

        if self.stop_until_reach_accuracy:
            es = self._es_until_reach_accuracy
        else:
            es = self._es
            

        for epoch in range(self.num_epochs):
            if not es.early_stop:
                model.train()
                # train_gt = []
                # train_predict = []
                
                epoch_train_loss = 0 
                epoch_train_r2 = 0

                for data in tqdm(self.train_dataloader):
                    torch.cuda.synchronize()
                    optimizer.zero_grad()
                    train_it_start = int(round(time.time()*1000))

                    # Early stopping if there are no improvements after patience number of steps
                    # Perform a training step
                    src, laplacian, tgt = data['x'], data['G'], data['y']
                    x = src.to(device)
                    laplacian = laplacian.to(device)
                    target = tgt.to(device)
                    model.train()
                    
                    # forward prop
                    output = model(x, laplacian)
                    loss = criterion(output, target)
                    batch_r2_loss = r2_loss(output, target)
                    #backward prop
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    #Update the optimizer and scheduler
                    optimizer.step()

                    # losses.append(loss)

                    torch.cuda.synchronize()
                    time_elapsed = int(round(time.time()*1000)) - train_it_start
                    total_train_time += time_elapsed

                    epoch_train_loss += loss
                    epoch_train_r2 += batch_r2_loss 

                train_loss = epoch_train_loss / len(self.train_dataloader)
                train_r2_loss = epoch_train_r2 / len(self.train_dataloader)
                train_losses.append(train_loss)
                train_r2_losses.append(train_r2_loss)
                print("Epoch: %d, train_loss: %1.5f" % (epoch, train_loss))
                print("Epoch: %d, train_r2_loss: %1.5f" % (epoch, train_r2_loss))

                # validation 

                model.eval()
                epoch_val_loss = 0

                # Freeze gradients
                with torch.no_grad():
                    for data  in tqdm(self.valid_dataloader):
                        eval_loss = 0 
                        # x, y, laplacian = dev_dataset.get_dataset()
                        x = data['x'].to(device)
                        y = data['y'].to(device)
                        laplacian = data['G'].to(device)
                        
                        torch.cuda.synchronize()
                        valid_start = int(round(time.time()*1000))

                        output = model(x, laplacian)

                        torch.cuda.synchronize()
                        batch_valid_time = int(round(time.time()*1000)) - valid_start
                        total_train_time += batch_valid_time

                        eval_loss = criterion(output, y)
                        epoch_val_loss += eval_loss.item()
                    val_loss = epoch_val_loss / len(self.valid_dataloader)

                val_losses.append(val_loss)
                # update scheduler
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

    # This function is called to test the trained model on the test set and return the metrics
    def test(self):
        self._model.load_state_dict(torch.load(self._weights_path)["model_dict"])
        model = self._model

        model.eval()

        groundtruth = []
        predict = [] 
        lst_inference_time = []

        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                x = data['x'].to(self.device)
                y = data['y'].to(self.device)
                laplacian = data['G'].to(self.device)

                torch.cuda.synchronize()
                inference_time_start = int(round(time.time()*1000))
                
                output = model(x, laplacian)
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
       

            # x, y, laplacian = test_dataset.get_dataset()
            # x = x.to(device)
            # y = y.to(device)
            # laplacian = laplacian.to(device)
            # output = model(x, laplacian)
            # output, y = output.squeeze(), y.squeeze()
            # output, y = unscale_tensor(output, scaler), unscale_tensor(y, scaler)
        # total_time = round(time.time() - start_time, 5)
        # mae = mean_absolute_error(output, y)
        # mse = mean_squared_error(y, output)
        # rmse = root_mean_squared_error(y, output)
        # mape = mean_absolute_percentage_error(y, output)
        # r2 = r2_score(y, output)
        # print('MSE loss : ', mse, end=" ")
        # print('RMSE loss : ', rmse, end=" ")
        # print('MAE loss : ', mae, end=" ")
        # print('MAPE loss: ', mape, end=" ")
        # print('R2 score: ', r2, end=" ")
        # print()

        # return output, y, mae, mse, rmse, mape, r2, total_time

    # # The function to perform a training step (updating weights after a batch)
    # def training_step(self, model, optimizer, scheduler, src, laplacian, tgt, criterion):
    #     criterion = self.criterion
    #     device = self.device

    #     model.train()
    #     x = src.to(device)
    #     laplacian = laplacian.to(device)
    #     target = tgt.to(device)
    #     # forward prop
    #     output = model(x, laplacian)
    #     loss = criterion(output, target)

    #     r2_loss = r2_loss(output, target)
    #     #backward prop
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    #     #Update the optimizer and scheduler
    #     optimizer.step()
    #     scheduler.step()
    #     optimizer.zero_grad()

    #     return loss.item(), r2_loss.item()



    # # THis function is called to perform validation on the dev set
    # def validate(self, model, val_dataloader):
    #     criterion = self.criterion
    #     device  = self.device

    #     model.eval()
    #     eval_losses = []
    #     # Freeze gradients
    #     with torch.no_grad():
    #         for data  in tqdm(val_dataloader):
    #             batch_loss = 0 
    #             # x, y, laplacian = dev_dataset.get_dataset()
    #             x = data['x'].to(device)
    #             y = data['y'].to(device)
    #             laplacian = data['G'].to(device)
    #             output = model(x, laplacian)
    #             eval_loss = criterion(output, y)
    #     return eval_loss.item()



if __name__ == "__main__":
    config_file = "../../config/gc_lstm.yml"
    with open(config_file, encoding="utf8") as f:
        config = yaml.safe_load(f)

    supervisor = GCLSTMSupervisor(config)

    # Uncomment this line to test the training and testing function
    supervisor.train_and_test()
