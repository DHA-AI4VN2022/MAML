import torch
from models.magan.model import MAGAN
import os
from models.magan.utils import get_dataloader
from util import model as model_utils
from models.magan.utils import get_data_array
from util.early_stop import EarlyStopping
# them LRReduceOnPlateu
import numpy as np
from tqdm.notebook import tqdm
import torch.nn as nn
import time

class MAGANSupervisor():
    def __init__(self, args, config, target_station, device):
        self._epochs = config["epochs"]
        self._lr = config["lr"]
        self.window_size = config["window_size"]

        self._optimizer = config['optimizer']
        self.optimizer = config['optimizer']

        self.input_size = config['window_size']
        self.input_dim = config['driving_series']

        self._hidden_size = config['hidden_size']
        self._output_size = config['output_size']

        self._batch_size = config['batch_size']

        # Data
        self.target_station = target_station
        transformed_data, scaler = get_data_array(config, target_station)
        self._scaler = scaler 
        self.true_labels = torch.ones(self._batch_size,1).to(device)
        self.fake_labels = torch.zeros(self._batch_size,1).to(device)
        self.device = device

        train_dataloader, valid_dataloader, test_dataloader = get_dataloader(data=transformed_data, config=config)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        first_it = next(iter(train_dataloader))   
        X, y = first_it['x'], first_it['y']

        self._base_dir = model_utils.generate_log_dir(args)
        self._weights_path = os.path.join(self._base_dir, "best.pth")

        self.config = config

        # Model
        self._model = MAGAN(config, device).to(self.device)

    def train(self):
        val_losses = []
        clip = 5
        model = self._model
        criterion = torch.nn.MSELoss()
        es = EarlyStopping(
            # patience=patience,
            verbose=True,
            delta=0.0,
            path=self._weights_path          
        )
        total_train_time = 0.0
        if self.optimizer == 'adam':
            optimizer_marnn = torch.optim.Adam(model.marnn.parameters(), lr=self._lr)
            optimizer_dis = torch.optim.Adam(model.dis.parameters(), lr=self._lr)
            optimizer_marnn_t = torch.optim.Adam(model.marnn.parameters(), lr=self._lr)
        # Train the model
        for epoch in range(self._epochs):
            if not es.early_stop:
                epoch_trainD_loss = 0 
                epoch_trainG_loss = 0
                model.marnn.train()
                model.dis.train()
                torch.cuda.synchronize()
                train_it_start = int(round(time.time()*1000))
                for data in tqdm(self.train_dataloader):
                    x = data['x'].to(self.device)
                    y = data['y'].to(self.device)
                    target = data['target'].to(self.device)
                    model.marnn.zero_grad()
                    model.dis.zero_grad()
                    #train dis
                    y_fake = model.marnn(x,target).detach()
                    d_out_fake = model.dis(y_fake)
                    d_out_real = model.dis(y[:,0].view(self._batch_size,1))
                    d_loss_fake = criterion(d_out_fake, self.fake_labels)
                    d_loss_real = criterion(d_out_real, self.true_labels)
                    d_loss = d_loss_fake + d_loss_real
                    d_loss.backward()
                    optimizer_dis.step()
                    epoch_trainD_loss  += d_loss.item()
                    with torch.no_grad():
                        for param in model.dis.parameters():
                            param.clamp_(-0.01, 0.01)

                    #train MARNN
                    y_fake = model.marnn(x,target)
                    dis_out = model.dis(y_fake)
                    g_loss = criterion(dis_out, self.true_labels)
                    g_loss.backward()
                    nn.utils.clip_grad_norm_(model.marnn.parameters(), clip)
                    optimizer_marnn.step()
                    epoch_trainG_loss += g_loss.item()

                    #train again
                    model.marnn.zero_grad()
                    y_pred = model.marnn(x,target)
                    marnn_loss = criterion(y_pred,y[:,0].view(self._batch_size,1))
                    marnn_loss.backward()
                    # nn.utils.clip_grad_norm_(marnn.parameters(), clip)
                    optimizer_marnn_t.step()
                torch.cuda.synchronize()
                time_elapsed = int(round(time.time()*1000)) - train_it_start
                total_train_time += time_elapsed
                
                print('Train epoch {}: Discriminator train loss: {:.6f}\tMARNN train loss: {:.6f}\tTrain marnn Loss: {:.6f}'.format(epoch,epoch_trainD_loss,epoch_trainG_loss,marnn_loss.item()))
            #validation
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for data in tqdm(self.valid_dataloader):
                    batch_loss =  0

                    x = data['x'].to(self.device)
                    y = data['y'].to(self.device)
                    target = data['target'].to(self.device)
                    outputs = model(x,target)
                    batch_loss = criterion(outputs.squeeze(0), y[:,0].view(self._batch_size,1))
                    epoch_val_loss += batch_loss.item()
                val_loss = epoch_val_loss / len(self.valid_dataloader)
                val_losses.append(val_loss)
                print('Valid loss : {}'.format(val_loss))
        model_utils.save_checkpoint(model, optimizer_marnn, self._weights_path)
        return val_losses[-1], total_train_time
            
    def test(self):
        self._model.load_state_dict(torch.load(self._weights_path)["model_dict"])
        model = self._model
        model.eval
        groundtruth = []
        predict = [] 
        lst_inference_time = []
        with torch.no_grad():
            i = 0
            for data in tqdm(self.test_dataloader):
                torch.cuda.synchronize()
                inference_time_start = int(round(time.time()*1000))
                if (i%self._output_size) == 0:
                    x = data['x'].to(self.device)
                    y = data['y'].to(self.device)
                    target = data['target'].to(self.device)
                    next = data['next'].to(self.device)
                    for j in range(self._output_size):
                        output = model.marnn(x,target)
                        # import pdb; pdb.set_trace()
                        pred_targets = target[:,1:self.window_size]
                        target[:,0:self.window_size-1] = pred_targets
                        target[:,self.window_size-1] = output[:,0]
                        x_next = x[:,1:self.window_size,:]
                        x[:,0:self.window_size - 1,:] = x_next
                        x[:,self.window_size -1,:] = next[:,j,:]
                        x[:,self.window_size -1,0] = output[:,0]


                        groundtruth_final = y[:,j].view(self._batch_size,1).view(-1, 1).squeeze(-1)
                        output_final = output.view(-1, 1).squeeze(-1)

                        groundtruth += groundtruth_final.tolist()
                        predict += output_final.tolist()
                torch.cuda.synchronize()
                total_inference_time = int(round(time.time()*1000)) - inference_time_start 
                lst_inference_time.append(total_inference_time)
                i+=1
                
        predict_ = np.expand_dims(predict, 1)
        groundtruth_ = np.expand_dims(groundtruth, 1)
        predict_cpy_n_cols =  np.repeat(predict_, self.input_dim, axis=1)
        groundtruth_cpy_n_cols = np.repeat(groundtruth_, self.input_dim, axis=1)
        predicts = self._scaler.inverse_transform(predict_cpy_n_cols)[:, [0]]
        groundtruths = self._scaler.inverse_transform(groundtruth_cpy_n_cols)[:,[0]]

        final_predicts = predicts.squeeze(-1)
        final_groundtruths = groundtruths.squeeze(-1)
        num_params = sum(p.numel() for p in model.parameters())  
        # model_utils.save_results(final_groundtruths, final_predicts, self._base_dir)
        # model_utils.visualize(final_groundtruths, final_predicts, self._base_dir, "results.png")
        # model_utils.save_results(final_groundtruths, final_predicts, self._base_dir, self.target_station, self._output_size)
        model_utils.save_results(final_groundtruths, final_predicts, self._base_dir, self.target_station, num_params, np.mean(np.array(lst_inference_time)))
        model_utils.visualize(final_groundtruths, final_predicts, self._base_dir, "result_{}.png".format(self.target_station) )
