import torch
import torch.nn as nn
import torch.optim as optim
from models.ac_lstm.model import AcLSTM
import os, psutil
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.early_stop import EarlyStopping, EarlyStoppingReachAccuracy
# from models.ac_lstm.utils import get_dataloader, preprocess, Generator
from models.ac_lstm.utils import get_dataloader
from util.multi_loader import get_data_array, unscale_tensor
from util import model as model_utils
import numpy as np
import time as time
from tqdm import tqdm
from util.torch_util import weight_init
from util.loss import r2_loss
from sklearn.metrics import r2_score

class AcLSTMSupervisor():
    def __init__(self, args, config, target_station, train_ratio, device):
        # Config
        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.batch_size = config['batch_size']
        self.patience= config['patience']
        self.device = device
        # self._index = station_idx
        self.target_station = target_station

        config['input_len'] = args.input_len
        config['output_len'] = args.output_len
        config['num_input_station'] = args.num_input_station
        config['input_dim'] = len(config['input_features'])
        self.config = config

        # Data
        list_data, location, list_station, scaler = get_data_array(target_station, args, self.config)
        train_dataloader, valid_dataloader, test_dataloader = get_dataloader(station=list_station, data=list_data, \
                                                                            location=location,config=self.config, \
                                                                            target_station=target_station, train_ratio=train_ratio)
        # self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self._scaler = preprocess(args, config, target_station, train_ratio)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self._scaler = scaler 

        self._base_dir = model_utils.generate_log_dir(args, config['base_dir'])
        self._weights_path = os.path.join(self._base_dir, "best.pth")

        self.criterion = nn.MSELoss()

        # Model
        self.model = AcLSTM(config, device).to(self.device)
        self.model.apply(weight_init)
        
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
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.config['lr_decay_ratio'],patience=20)

        # Declare the training hyper parameters
        device = self.device
        criterion = self.criterion

        # train_dataset = Generator(self.X_train, self.y_train)
        # train_dataloader = get_dataloader(train_dataset, self.batch_size, shuffle = True)
        # valid_dataset = Generator(self.X_valid, self.y_valid)
        # valid_dataloader = get_dataloader(valid_dataset, self.batch_size, shuffle = False)
        
        # model = self.model
        # criterion = nn.MSELoss()
        # optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.config['lr_decay_ratio'], patience = 3)

        # Train the model
        total_train_time = 0.0

        val_losses = []
        train_losses = []
        train_r2_losses = []

        if self.stop_until_reach_accuracy:
            es = self._es_until_reach_accuracy
        else:
            es = self._es
        
        for epoch in range(self.epochs):
            if not es.early_stop:
                
                #Train
                epoch_train_loss = 0
                epoch_train_r2 = 0
                model.train()

                for data in tqdm(self.train_dataloader):
                    # input, target = input.float().to(self.device), target.float().to(self.device)
                    torch.cuda.synchronize()
                    train_it_start = int(round(time.time()*1000))

                    x, y =  data['x'].to(device), data['y'].to(device)
                    out = model(x)

                    batch_loss = criterion(out, y)

                    batch_loss.backward() 
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    torch.cuda.synchronize()
                    time_elapsed = int(round(time.time()*1000)) - train_it_start
                    total_train_time += time_elapsed

                    epoch_train_loss += batch_loss.item()
                    batch_r2_loss = r2_score(out.cpu().detach().numpy(), y.cpu().detach().numpy())
                    epoch_train_r2 += batch_r2_loss.item() 

                train_loss = epoch_train_loss / len(self.train_dataloader)
                train_r2_loss = epoch_train_r2 / len(self.train_dataloader)

                train_losses.append(train_loss)
                train_r2_losses.append(train_r2_loss)

                print("Epoch: %d, train_loss: %1.5f" % (epoch, train_loss))
                print("Epoch: %d, train_r2_loss: %1.5f" % (epoch, train_r2_loss))

                #Validation
                epoch_val_loss = 0
                model.eval()

                with torch.no_grad():
                    for data in self.valid_dataloader:
                        eval_loss = 0
                        x = data['x'].to(device)
                        y = data['y'].to(device)
                        torch.cuda.synchronize()
                        valid_start = int(round(time.time()*1000))

                        # input, target = input.float().to(self.device), target.float().to(self.device)
                        out = model(x)

                        torch.cuda.synchronize()
                        batch_valid_time = int(round(time.time()*1000)) - valid_start
                        total_train_time += batch_valid_time

                        eval_loss = criterion(out, y)
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

    #Test model
    def test(self):
        # test_dataset = Generator(self.X_test, self.y_test)
        # test_dataloader = get_dataloader(test_dataset, self.batch_size, shuffle = False)

        self.model.load_state_dict(torch.load(self._weights_path)["model_dict"])
        self.model.eval()
        
        groundtruth = []
        predict = [] 
        lst_inference_time = []

        # test_dataset = Generator(self.X_test, self.y_test)
        # test_dataloader = get_dataloader(test_dataset, self.batch_size, shuffle = False)

        # list_predict = []
        # list_actual = []

        # torch.cuda.synchronize()
        # begin_time = int(round(time.time() * 1000))

        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                # input, target = input.float().to(self.device), target.float().to(self.device)
                x = data['x'].to(self.device)
                y = data['y'].to(self.device)

                torch.cuda.synchronize()
                inference_time_start = int(round(time.time()*1000))
                
                output = self.model(x)
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