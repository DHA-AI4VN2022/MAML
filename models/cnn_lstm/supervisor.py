import torch
from models.cnn_lstm.model import CNNLSTM
import os, psutil
from util import model as model_utils
from util.single_loader import get_data_array, get_dataloader
from util.early_stop import EarlyStopping, EarlyStoppingReachAccuracy
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.torch_util import weight_init
from util.loss import r2_loss
import time

class CNNLSTMSupervisor():
    def __init__(self, args, config, target_station, train_ratio, device):
        # Config
        self._epochs = config["epochs"]
        self._lr = config["lr"]

        self._patience = config['patience']
        self._optimizer = config['optimizer']
        self._lr_decay = config['lr_decay_ratio']
        self.optimizer = config['optimizer']

        self.input_size = config['input_dim']
        self.output_size = config['output_dim']

        self.config = config
        self.config['input_len'] = args.input_len
        self.config['output_len'] = args.output_len

        self.num_layers = config["num_layers"]
        self.batch_size = config["batch_size"]
        self.dropout = 0
        
        # Data
        self.target_station = target_station
        transformed_data, scaler = get_data_array(self.config, target_station)
        self._scaler = scaler 
        
        train_dataloader, valid_dataloader, test_dataloader = get_dataloader(transformed_data, args, self.config, train_ratio)
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self._base_dir = model_utils.generate_log_dir(args, config['base_dir'])
        self._weights_path = os.path.join(self._base_dir, "best.pth")

        # Model
        self.device = device
        self._model = CNNLSTM(config, device).to(self.device)
        
        self._model.apply(weight_init)

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
        
    def train(self):
        model = self._model
        criterion = torch.nn.MSELoss()
        if self.optimizer == 'adam':
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

        # Train the model
        for epoch in range(self._epochs):
            if not self._es.early_stop:
                model.train()
                # train 
                epoch_train_loss = 0 
                
                train_gt = []
                train_predict = [] 
                
                epoch_train_r2 = 0
                for data in tqdm(self.train_dataloader):
                    torch.cuda.synchronize()
                    train_it_start = int(round(time.time()*1000))

                    batch_loss = 0 
                    optimizer.zero_grad()
                    
                    x = data['x'].to(self.device)
                    y = data['y'].to(self.device) 
                
                    outputs = model(x)
                    # batch_loss = criterion(outputs.view(self.batch_size, -1, self.output_size),  y)
                    batch_loss = criterion(outputs, y)

                    batch_loss.backward()
                    optimizer.step()

                    torch.cuda.synchronize()
                    time_elapsed = int(round(time.time()*1000)) - train_it_start
                    total_train_time += time_elapsed

                    epoch_train_loss += batch_loss.item()

                    batch_r2_loss = r2_loss(outputs, y)
                    epoch_train_r2 += batch_r2_loss.item()
                
                train_loss = epoch_train_loss / len(self.train_dataloader)
                train_r2_loss = epoch_train_r2 / len(self.train_dataloader)

                train_losses.append(train_loss)
                train_r2_losses.append(train_r2_loss)

                print("Epoch: %d, train_loss: %1.5f" % (epoch, train_loss))
                print("Epoch: %d, train_r2_loss: %1.5f" % (epoch, train_r2_loss))

                #validation
                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for data in tqdm(self.valid_dataloader):
                        batch_loss =  0
                        x = data['x'].to(self.device)
                        y = data['y'].to(self.device)

                        torch.cuda.synchronize()
                        valid_start = int(round(time.time()*1000))

                        outputs = model(x)

                        torch.cuda.synchronize()
                        batch_valid_time = int(round(time.time()*1000)) - valid_start
                        total_train_time += batch_valid_time

                        # batch_loss = criterion(outputs.view(self.batch_size, -1, self.output_size),  y)
                        batch_loss = criterion(outputs, y)
                        epoch_val_loss += batch_loss.item()
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

    def test(self):
        self._model.load_state_dict(torch.load(self._weights_path)["model_dict"])
        model = self._model
        groundtruth = []
        predict = [] 
        lst_inference_time = []

        model.eval()
        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                x = data['x']
                y = data['y']

                torch.cuda.synchronize()
                inference_time_start = int(round(time.time()*1000))

                # output tensor
                outputs = model(x)

                torch.cuda.synchronize()
                total_inference_time = int(round(time.time()*1000)) - inference_time_start 
                lst_inference_time.append(total_inference_time)

                groundtruth_final = y.view(-1, 1).squeeze(-1)
                output_final = outputs.view(-1, 1).squeeze(-1)

                groundtruth += groundtruth_final.tolist()
                predict += output_final.tolist()
        
        predict_ = np.expand_dims(predict, 1)
        groundtruth_ = np.expand_dims(groundtruth, 1)
        predict_cpy_n_cols =  np.repeat(predict_, self.input_size, axis=1)
        groundtruth_cpy_n_cols = np.repeat(groundtruth_, self.input_size, axis=1)
        predicts = self._scaler.inverse_transform(predict_cpy_n_cols)[:, [0]]
        groundtruths = self._scaler.inverse_transform(groundtruth_cpy_n_cols)[:,[0]]

        final_predicts = predicts.squeeze(-1)
        final_groundtruths = groundtruths.squeeze(-1)
                
        test_time = np.sum(np.array(lst_inference_time))
        res = {
            'base_dir': self._base_dir,
            'target_station': self.target_station,
            'groundtruth': final_groundtruths,
            'predict': final_predicts,
            'test_time': test_time
        }
        return res
