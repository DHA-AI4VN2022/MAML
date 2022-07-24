import torch
from models.imda_vae.model import ImdaVAEModel, PredictionLayer
from models.imda_vae.utils import  loss_function
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

class ImdaVAESupervisor():
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
        self.hidden_size = config["hidden_dim"]
        self.hidden_size_2 = config['hidden_dim_2']
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
        self._weights_path_encoder = os.path.join(self._base_dir, "best_encoder.pth")
        self._weights_path_predictor = os.path.join(self._base_dir, "best_predictor.pth")

        # Model
        self.device = device
        self._model_encoder = ImdaVAEModel(config, device).to(self.device)
        self._model_encoder.apply(weight_init)
        
        self._model_prediction = PredictionLayer(config, device).to(self.device)
        self._model_prediction.apply(weight_init)

        self._es_encoder = EarlyStopping(
            patience=self.config['patience'],
            verbose=True,
            delta=0.0,
            path=self._weights_path_encoder            
        )

        self._es_predictor = EarlyStopping(
            patience=self.config['patience'],
            verbose=True,
            delta=0.0,
            path=self._weights_path_predictor            
        )
        self._es_until_reach_accuracy_predictor = EarlyStoppingReachAccuracy(
            patience=self.config['patience'],
            verbose=True,
            delta=0.0,
            path=self._weights_path_predictor
        )
        self.train_ratio = train_ratio
        self.stop_until_reach_accuracy = False
        if args.experimental_mode == 'stop_until_reach_accuracy':
            self.stop_until_reach_accuracy = True
        
    def train(self):
        model_encoder = self._model_encoder
        model_prediction = self._model_prediction

        criterion_unsupervised = loss_function

        optimizer_unsupervised = torch.optim.Adam(model_encoder.parameters(), lr=self._lr)
        optimizer_supervised = torch.optim.Adam(model_prediction.parameters(), lr=self._lr)
        
        scheduler_unsupervised = ReduceLROnPlateau(optimizer_unsupervised, 'min', factor=self.config['lr_decay_ratio'], patience=5)

        # Train the model
        total_train_time = 0.0

        val_losses_encoder = []
        train_losses_encoder = []

        es_encoder = self._es_encoder
        if self.stop_until_reach_accuracy:
            es_predictor = self._es_until_reach_accuracy_predictor
        else:
            es_predictor = self._es_predictor

        # Train the model
        for epoch in range(self._epochs):
            if not es_encoder.early_stop:
                model_encoder.train()
                # train 
                epoch_train_loss_encoder = 0 

                for data in tqdm(self.train_dataloader):
                    torch.cuda.synchronize()
                    train_it_start = int(round(time.time()*1000))

                    batch_loss_us = 0 
                    optimizer_unsupervised.zero_grad()
                    x = data['x'].to(self.device)
                    outputs, _,  mean, log_var= model_encoder(x)
                    # print('-------------------X-----------------------------')
                    # print(x)
                    # print(outputs)
                    # print('-------------------X-----------------------------')
                    batch_loss_us = criterion_unsupervised(x,outputs, mean, log_var, self.config)
                    batch_loss_us.backward()
                    optimizer_unsupervised.step()

                    torch.cuda.synchronize()
                    time_elapsed = int(round(time.time()*1000)) - train_it_start
                    total_train_time += time_elapsed

                    epoch_train_loss_encoder += batch_loss_us
                
                train_loss_encoder = epoch_train_loss_encoder / len(self.train_dataloader)

                train_losses_encoder.append(train_loss_encoder)
                print("Epoch: %d, train_loss_encoder: %1.5f" % (epoch, train_loss_encoder))

                #validation unsupervised
                model_encoder.eval()
                epoch_val_loss_encoder = 0
                with torch.no_grad():
                    for data in tqdm(self.valid_dataloader):
                        batch_loss_sup =  0
                        x = data['x'].to(self.device)

                        torch.cuda.synchronize()
                        valid_start = int(round(time.time()*1000))

                        outputs, _,  mean, log_var = model_encoder(x)

                        torch.cuda.synchronize()
                        batch_valid_time = int(round(time.time()*1000)) - valid_start
                        total_train_time += batch_valid_time

                        batch_loss_sup = criterion_unsupervised(x, outputs, mean, log_var, self.config)
                        epoch_val_loss_encoder += batch_loss_sup
                    val_loss_encoder = epoch_val_loss_encoder / len(self.valid_dataloader)
                
                es_encoder(val_loss_encoder, model_encoder)
                val_losses_encoder.append(val_loss_encoder)
                # update scheduler
                scheduler_unsupervised.step(val_loss_encoder)
                print('Epoch-{0} lr: {1}'.format(epoch, optimizer_unsupervised.param_groups[0]['lr']))

        # Freeze encoder, train predictor 
        for param in model_encoder.parameters():
            param.requires_grad = False

        criterion_supervised = torch.nn.MSELoss()
        scheduler_supervised = ReduceLROnPlateau(optimizer_supervised, 'min', factor=self.config['lr_decay_ratio'], patience=5)
        train_losses_predictor = []
        val_losses_predictor = []
        train_r2_losses_predictor = []

        for epoch in range(self._epochs):
            if not es_predictor.early_stop:
                model_prediction.train()
                model_encoder.eval()
                # train 
                epoch_train_loss_predictor = 0 
                epoch_train_r2_predictor = 0

                for data in tqdm(self.train_dataloader):
                    torch.cuda.synchronize()
                    train_it_start = int(round(time.time()*1000))

                    batch_loss_sup = 0 
                    optimizer_supervised.zero_grad()
                    x = data['x'].to(self.device)
                    y = data['y'].to(self.device) 

                    _, z, _, _ = model_encoder(x)
                    outputs = model_prediction(x, z)
                    batch_loss_sup = criterion_supervised(outputs,  y)
                    batch_loss_sup.backward()
                    optimizer_supervised.step()

                    torch.cuda.synchronize()
                    time_elapsed = int(round(time.time()*1000)) - train_it_start
                    total_train_time += time_elapsed

                    epoch_train_loss_predictor += batch_loss_sup.item()

                    batch_r2_loss = r2_loss(outputs, y)
                    epoch_train_r2_predictor += batch_r2_loss.item()
                
                train_loss_predictor = epoch_train_loss_predictor / len(self.train_dataloader)
                train_r2_loss_predictor = epoch_train_r2_predictor / len(self.train_dataloader)

                train_losses_predictor.append(train_loss_predictor)
                train_r2_losses_predictor.append(train_r2_loss_predictor)

                print("Epoch: %d, train_loss predictor: %1.5f" % (epoch, train_loss_predictor))
                print("Epoch: %d, train_r2_loss predictor: %1.5f" % (epoch, train_r2_loss_predictor))

                #validation
                model_prediction.eval()
                epoch_val_loss_preditor = 0
                with torch.no_grad():
                    for data in tqdm(self.valid_dataloader):
                        batch_loss_val =  0
                        x = data['x'].to(self.device)
                        y = data['y'].to(self.device)

                        torch.cuda.synchronize()
                        valid_start = int(round(time.time()*1000))

                        _, z, _, _ = model_encoder(x)
                        outputs = model_prediction(x, z)

                        torch.cuda.synchronize()
                        batch_valid_time = int(round(time.time()*1000)) - valid_start
                        total_train_time += batch_valid_time

                        batch_loss_val = criterion_supervised(outputs, y)
                        epoch_val_loss_preditor += batch_loss_val.item()
                    val_loss_predictor = epoch_val_loss_preditor / len(self.valid_dataloader)

                val_losses_predictor.append(val_loss_predictor)
                # update scheduler
                scheduler_supervised.step(val_loss_predictor)
                print('Epoch-{0} lr: {1}'.format(epoch, optimizer_supervised.param_groups[0]['lr']))
                if self.stop_until_reach_accuracy:
                    es_predictor(train_r2_loss_predictor, model_prediction)
                else:
                    es_predictor(val_loss_predictor, model_prediction)

        num_params_encoder = sum(p.numel() for p in model_encoder.parameters())
        num_params_predictor = sum(p.numel() for p in model_prediction.parameters())
        num_params = num_params_encoder + num_params_predictor
        process = psutil.Process(os.getpid())
        mem_used = process.memory_info().rss  / 1048576 # bytes -> mb  

        res = {
            'train_ratio': self.train_ratio,
            'val_losses': val_losses_predictor,
            'train_losses': train_losses_predictor,
            'train_r2_losses': train_r2_losses_predictor, 
            'train_time': total_train_time,
            'num_params': num_params,
            'mem_used': mem_used
        }
        return res

    def test(self):
        self._model_encoder.load_state_dict(torch.load(self._weights_path_encoder)["model_dict"])
        self._model_prediction.load_state_dict(torch.load(self._weights_path_predictor)["model_dict"])
        model_encoder = self._model_encoder
        model_predictor = self._model_prediction

        groundtruth = []
        predict = [] 
        lst_inference_time = []

        model_encoder.eval()
        model_predictor.eval()

        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                x = data['x'].to(self.device)
                y = data['y'].to(self.device)

                torch.cuda.synchronize()
                inference_time_start = int(round(time.time()*1000))

                # output tensor
                _, z, _, _ = model_encoder(x)
                outputs = model_predictor(x, z)

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
