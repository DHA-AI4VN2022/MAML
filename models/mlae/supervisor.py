import psutil
from matplotlib.pyplot import flag
import numpy as np
from numpy import argsort
from sklearn import config_context
import torch
import os
from models.mlae.utils import  get_data_array, MultiTaskPM25Dataset, get_dataloader, unscale_tensor
from util import model as model_utils
from torch.utils.data import DataLoader
from models.mlae.model import MLAE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.early_stop import EarlyStopping , EarlyStoppingReachAccuracy
from util import model as model_utils
from tqdm import tqdm
import torch.nn as nn
from util.torch_util import weight_init
import time
from util.metric import mae, mse, rmse, mape, nse, mdape, r2_score
from util.loss import r2_loss

import pandas as pd
from   torch.nn import functional as F


from maml.maml import MAML


class MultitaskLSTMAutoencSupervisor():
    def __init__(self, args, config, train_ratio, device=None):
        # Config
        self.args = args
        self.config = config
        self.ae_epochs = self.config['ae_epochs']
        self.model_epochs = self.config['model_epochs']

        self.ae_lr = float(self.config['ae_lr'])
        self.model_lr = float(self.config['model_lr'])
        self.task_lr = float(self.config['task_lr'])
        self.update_step = int(self.config['update_step'])

        self._lr_decay = float(self.config['lr_decay'])
        self._weight_decay = float(self.config['weight_decay'])

        self._batch_size = self.config['batch_size']
        self._patience = self.config['patience']

        if args.input_len != None:
            self.config['input_len'] = args.input_len


        if args.output_len != None:
            self.config['output_len'] = args.output_len

        self.ntime_step = self.config["output_len"]


        # self.target_station = target_station
        self.stations = config['target_station']
        self.train_ratio = train_ratio
        self.train_ratio = args.train_ratio
        self.device = device
        # Data
        (pm_array, meteo_array), location, list_k_stations, self._scaler = get_data_array( args=self.args, config=self.config)
        # print(meteo_array, meteo_array.shape)
        self.dataset = MultiTaskPM25Dataset(pm25_data=pm_array, meteo_data=meteo_array, config=self.config)
        self._train_loader, self._valid_loader, self._test_loader = get_dataloader(pm25_data=pm_array,
                                                           meteo_data=meteo_array,
                                                                                   args=self.args, config=self.config,
                                                                                   train_ratio=self.train_ratio)
        self._base_dir = os.path.join(self.config['base_dir'])
        self._weights_path = os.path.join(self._base_dir, "best.pth")

        # early stopping
        self._es = EarlyStopping(
            patience=self._patience,
            verbose=True,
            delta=0.0,
            path=self._weights_path
        )

        self._es_until_reach_accuracy = EarlyStoppingReachAccuracy(
            patience=self._patience,
            verbose=True,
            delta=0.0,
            path=self._weights_path
        )
        self.stop_until_reach_accuracy = False

        if args.experimental_mode == 'stop_until_reach_accuracy':
            self.stop_until_reach_accuracy = True

        # Model
        self.number_tasks = self.config["num_stations"]
        #         print(self.number_tasks)
        self._model = MLAE(self.config, self.number_tasks, self.device)

    def train(self):
        model = self._model
        # maml_learner = MAML(model = model, lr = self.task_lr,allow_unused=True)
        # for name, param in model.named_parameters():
        #     if name == "lstm_block.lstm_1.weight_ih_l0":
        #         print(name, param.shape)

        model.apply(weight_init)
        autoencoder = model.ae_block
        optimizer_model = torch.optim.Adam(model.parameters(), lr=self.model_lr, weight_decay=self._weight_decay)
        optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=self.ae_lr, weight_decay=self._weight_decay)

        scheduler_model = ReduceLROnPlateau(
            optimizer_model, 'min', factor=self._lr_decay, patience=5, min_lr=1e-10)

        scheduler_ae = ReduceLROnPlateau(
            optimizer_ae, 'min', factor=self._lr_decay, patience=5, min_lr=1e-10)

        criterion = torch.nn.MSELoss()

        num_batches = len(self._train_loader)
        ae_losses = []
        ae_val_losses = []
        val_losses = []
        train_losses = []
        train_r2_losses = []
        total_train_time = 0.0

        best_ae_loss = 999
        non_decreasing_epochs = 0
        # train the autoencoder first
        if self.stop_until_reach_accuracy:
            es = self._es_until_reach_accuracy
        else:
            es = self._es

        print("Training the Autoencoder")
        for epoch in (range(self.ae_epochs)):
            if non_decreasing_epochs > self._patience:
                print()
                print("Early stopping in AE enabled")
                break

            batch = 0
            autoencoder.train()
            loss_epoch = 0


            for data in tqdm(self._train_loader):
                x_pm, y_pm, x_meteo = data
                #                 print(x_meteo.shape)
                x_pm, y_pm, x_meteo = x_pm.to(self.device), y_pm.to(self.device), x_meteo.to(self.device)
                _, x_reconstruct = autoencoder(x_meteo)

                ae_loss = criterion(x_reconstruct, x_meteo)
                optimizer_ae.zero_grad()
                ae_loss.backward()
                optimizer_ae.step()
                nn.utils.clip_grad_norm_(autoencoder.parameters(), 1)

                ae_losses.append(ae_loss.item())

            autoencoder.eval()
            val_losses = []
            with torch.no_grad():
                for data in self._valid_loader:
                    x_pm, y_pm, x_meteo = data
                    #                     print(x_meteo.shape)
                    x_pm, y_pm, x_meteo = x_pm.to(self.device), y_pm.to(self.device), x_meteo.to(self.device)
                    _, x_reconstruct = autoencoder(x_meteo)

                    ae_loss = criterion(x_reconstruct, x_meteo)
                    ae_val_losses.append(ae_loss.item())

            val_loss = sum(ae_val_losses) / len(ae_val_losses)
            scheduler_ae.step(val_loss)
            print(f"Autoencoder epoch {epoch} \t Autoencoder Loss {val_loss}")
            if val_loss < best_ae_loss:
                best_ae_loss = val_loss
                non_decreasing_epochs = 0
                model_utils.save_checkpoint(model, optimizer_model, self._weights_path)
            #                 torch.save(model.state_dict(), self._weights_path)

            else:
                non_decreasing_epochs += 1

        print("Finished training the Autoencoder")

        # Training the full model
        model.load_state_dict(torch.load(self._weights_path)["model_dict"])

        for param in model.ae_block.parameters():
            param.requires_grad = False

        # torch.autograd.set_detect_anomaly(True)
        maml_learner = MAML(model = model, lr = self.task_lr,allow_unused=True)
        optimizer_model = torch.optim.Adam(maml_learner.parameters(), lr=self.model_lr, weight_decay=self._weight_decay)


        for epoch in range(self.model_epochs):
            if not es.early_stop:
                model.train()
                epoch_train_loss = 0
                epoch_train_r2 = 0

                for data in tqdm(self._train_loader):
                    x_pm, y_pm, x_meteo = data
                    x_pm, y_pm, x_meteo = x_pm.to(self.device), y_pm.to(self.device), x_meteo.to(self.device)

                    model.train()
                    torch.cuda.synchronize()
                    train_it_start = int(round(time.time() * 1000))

                    meta_train_loss = 0
                    for task in range(self.number_tasks):
                        task_learner =  maml_learner.clone()

                        # for _ in range():
                        #     # with torch.backends.cudnn.flags(enabled=False):
                        #     task_learner.module._apply(lambda x: x)
                        #
                        #     output = task_learner(x_pm, x_meteo)
                        #     task_loss = criterion(output[:,task,:],y_pm[:,task,:])
                        #     task_learner.adapt(task_loss,allow_unused=True, allow_nograd=True)
                        # Update the meta model
                        # with torch.backends.cudnn.flags(enabled=False):
                        output = task_learner(x_pm, x_meteo)
                        update_task_loss = criterion(output[:, task, :], y_pm[:, task, :])
                        task_learner.adapt(update_task_loss, allow_unused=True, allow_nograd=True)
                        meta_train_loss += update_task_loss

                    meta_train_loss = meta_train_loss / self.number_tasks
                    optimizer_model.zero_grad()
                    meta_train_loss.backward()
                    optimizer_model.step()
                    nn.utils.clip_grad_norm_(model.parameters(), 1)

                    # original_parameter = model.state_dict()
                    #
                    # def update_param(param_list, grad, lr):
                    #     new_params = []
                    #     for i in range(len(param_list)):
                    #         param = param_list[i]
                    #         param_grad = grad[i]
                    #         if param_grad != None:
                    #             updated_param = param - lr * param_grad
                    #         else:
                    #             updated_param = param
                    #         new_params.append(updated_param)
                    #     return new_params
                    #
                    # for i in range(self.number_tasks):
                    #     # print(i)
                    #     clone_model = clone_module(model)
                    #     clone_model.load_state_dict(original_parameter)
                    #     for k in range(self.update_step):
                    #         output = clone_model(x_pm,x_meteo)
                    #         # output, y = output.clone().detach(), y_pm.clone().detach()
                    #         y_task, output_task = y_pm[:,i,:], output[:,i,:]
                    #         loss = F.mse_loss(output_task, y_task)
                    #         grad = torch.autograd.grad(loss, clone_model.parameters() , allow_unused = True)
                    #         # for name, param in model.named_parameters():
                    #         #     if param.grad is None:
                    #         #         print(name)
                    #         param_list = []
                    #         with torch.no_grad():
                    #             for param in clone_model.parameters():
                    #                 param_list.append(param)
                    #             updated_param_list = update_param(param_list, grad, self.task_lr)

                            # print("Before")
                            # for name, param in model.named_parameters():
                            #     if name == "lstm_block.lstm_1.weight_ih_l0":
                            #         print(name, param.shape)
                            # param_index = 0
                            # # print(model.state_dict())
                            # for  name, param in clone_model.named_parameters():
                            #     param.data = updated_param_list[param_index]
                            #     param_index+=1
                            #
                            # output = clone_model(x_pm,x_meteo)
                            # # output, y = output.clone().detach(), y_pm.clone().detach()
                            # y_task, output_task = y_pm[:,i,:], output[:,i,:]
                            # loss_task = F.mse_loss(output_task, y_task)
                            # # print(loss_task)
                            # losses_task[k] += loss_task


                        # model.load_state_dict(original_parameter)
                        # losses_task_sum = losses_task[-1] / self.number_tasks
                        # optimizer_model.zero_grad()
                        # losses_task_sum.backward()
                        # optimizer_model.step()
                        # nn.utils.clip_grad_norm_(model.parameters(), 1)
                    torch.cuda.synchronize()
                    time_elapsed = int(round(time.time() * 1000)) - train_it_start
                    total_train_time += time_elapsed

                    epoch_train_loss += meta_train_loss.item()
                    batch_r2_loss = 0
                    epoch_train_r2 += batch_r2_loss

                    # # Create copies of models' parameters for each tasks
                    # # task_parameters = []
                    # # task_optimizers = []
                    # # for i in range(self.number_tasks):
                    # #     task_param = MLAE(self.config, self.number_tasks, self.device)
                    # #     task_param.load_state_dict(model.state_dict())
                    # #     task_parameters.append(task_param)
                    # #     task_optimizer = torch.optim.Adam(task_param.parameters(), lr=self.task_lr, weight_decay=self._weight_decay)
                    # #     task_optimizers.append(task_optimizer)
                    # #
                    # # Temporarily freeze the model's gradients
                    # x_pm, y_pm, x_meteo = data
                    # x_pm, y_pm, x_meteo = x_pm.to(self.device), y_pm.to(self.device), x_meteo.to(self.device)
                    #
                    # task_losses = []
                    # # print(y_pm.shape)
                    # for i in range(self.number_tasks):
                    #     output = task_parameters[i](x_pm,x_meteo)
                    #     y_task, output_task = y_pm[:,i,:], output[:,i,:]
                    #
                    #     # Update theta_i for each tasks
                    #     loss_task = criterion(output_task, y_task)
                    #     # print("Before", loss_task)
                    #     task_optimizers[i].zero_grad()
                    #     loss_task.backward()
                    #     task_optimizers[i].step()
                    #
                    #     #calculate the ith task loss after updating theta_i
                    #     output = task_parameters[i](x_pm,x_meteo)
                    #     y_task, output_task = y_pm[:,i,:], output[:,i,:]
                    #     loss_task_updated =  criterion(output_task, y_task)
                    #     task_losses.append(loss_task_updated)
                    #     # print("Updated", loss_task_updated)
                    #
                    # global_loss = sum(task_losses)
                    # optimizer_model.zero_grad()
                    # global_loss.backward()
                    # optimizer_model.step()
                    # nn.utils.clip_grad_norm_(model.parameters(), 1)
                    # print("Global", global_loss)
                    # torch.cuda.synchronize()
                    # time_elapsed = int(round(time.time() * 1000)) - train_it_start
                    # total_train_time += time_elapsed
                    #
                    # epoch_train_loss += global_loss.item()
                    # batch_r2_loss = 0
                    # epoch_train_r2 += batch_r2_loss

                train_loss = epoch_train_loss / len(self._train_loader)
                train_r2_loss = epoch_train_r2 / len(self._train_loader)
                train_losses.append(train_loss)
                train_r2_losses.append(train_r2_loss)

                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for data in self._valid_loader:
                        x_pm, y_pm, x_meteo = data
                        x_pm, y_pm, x_meteo = x_pm.to(self.device), y_pm.to(self.device), x_meteo.to(self.device)
                        #                 _,x_reconstruct = autoencoder(x_meteo)
                        output = model(x_pm, x_meteo)
                        #                     print(output.shape, y_pm.shape)
                        #                     loss = criterion(output, y_pm)
                        multi_task_loss = 0
                        for i in range(self.number_tasks):
                            station_output = output[:, i, :]
                            station_y = y_pm[:, i, ]
                            loss = criterion(station_output, station_y)
                            multi_task_loss += loss
                        multi_task_loss /= self.number_tasks
                        epoch_val_loss += multi_task_loss.item()
                    val_loss = epoch_val_loss / len(self._valid_loader)

                val_losses.append(val_loss)
                # model_losses.append(val_loss)
                scheduler_model.step(val_loss)
                print(f"Model epoch {epoch} \t Model Loss {val_loss}")
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

        return min(val_losses)

    def test(self):
        self._model.load_state_dict(torch.load(self._weights_path)["model_dict"])
        model = self._model

        predict = {}
        groundtruth = {}

        lst_inference_time = []
        model.eval()
        with torch.no_grad():
            for data in self._test_loader:
                x_pm, y_pm, x_meteo = data
                x_pm, y_pm, x_meteo = x_pm.to(self.device), y_pm.to(self.device), x_meteo.to(self.device)

                torch.cuda.synchronize()
                inference_time_start = int(round(time.time() * 1000))

                output = model(x_pm, x_meteo)
                # print(output.shape)
                torch.cuda.synchronize()
                total_inference_time = int(round(time.time() * 1000)) - inference_time_start
                lst_inference_time.append(total_inference_time)
                y, output = unscale_tensor(self.config, y_pm, self._scaler), unscale_tensor(self.config, output, self._scaler)
                # y, output = y[:,:,-1], output[:,:,-1]
                for i in range(self.number_tasks):
                    if self.stations[i] not in groundtruth.keys():
                        groundtruth.update({self.stations[i]: []})
                        predict.update({self.stations[i]: []})
                    groundtruth[self.stations[i]] += y[:,i].tolist()
                    predict[self.stations[i]] += output[:,i].tolist()

        test_time = np.sum(np.array(lst_inference_time))

        res = {
            'base_dir': self._base_dir,
            'target_station' : " ",
            'groundtruth': groundtruth,
            'predict': predict,
            'test_time': test_time
        }
        return res

if __name__ == '__main__':
    import yaml

    args = {'station_selection_strategy': 'correlation', "experimental_mode": 'stop_until_reach_accuracy'}
    with open("../../config/mlae.yml") as f:
        config = yaml.safe_load(f)
    print(config['output_len'])
    print(config['ae_lr'])

    target_station = "房山"
    train_ratio = 0.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    supervisor = MultitaskLSTMAutoencSupervisor(args, config, target_station, train_ratio, device=device)
    supervisor.train()
