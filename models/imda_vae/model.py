import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
# from models.imda_vae.layers import ScaledDotProductAttention

class ImdaVAEEncoder(nn.Module):
    def __init__(self,config) -> None:
        super(ImdaVAEEncoder, self).__init__()

        self.fc_input = nn.Linear(config['input_dim'], config['hidden_dim'])
        self.fc_input_1 = nn.Linear(config['hidden_dim'], config['hidden_dim'])
        # self.attn = ScaledDotProductAttention()
        self.fc_mean = nn.Linear(config['hidden_dim'], config['latent_dim'])
        self.fc_var   = nn.Linear(config['hidden_dim'], config['latent_dim'])
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True
    def forward(self,x):
        h_ = self.LeakyReLU(self.fc_input(x))
        h_ = self.LeakyReLU(self.fc_input_1(h_))
        mean = self.fc_mean(h_)
        log_var = self.fc_var(h_)

        return mean, log_var 

class ImdaVAEDecoder(nn.Module):
    def __init__(self, config) -> None:
        super(ImdaVAEDecoder, self).__init__()
        self.fc_hidden = nn.Linear(config['latent_dim'], config['hidden_dim'])
        self.fc_hidden_2 = nn.Linear(config['hidden_dim'], config['hidden_dim'])
        self.fc_output = nn.Linear(config['hidden_dim'], config['input_dim'])
        self.LeakyReLU = nn.LeakyReLU(0.2)
    def forward(self, x):
        h     = self.LeakyReLU(self.fc_hidden(x))
        h     = self.LeakyReLU(self.fc_hidden_2(h))
        x_hat = torch.sigmoid(self.fc_output(h))
        return x_hat

class ImdaVAEModel(nn.Module):
    def __init__(self, config, device):
        super(ImdaVAEModel, self).__init__()
        self.Encoder = ImdaVAEEncoder(config)
        self.Decoder = ImdaVAEDecoder(config)
        self.device =device 
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        return x_hat, z, mean, log_var 

class PredictionLayer(nn.Module):
    def __init__(self, config, device):
        super(PredictionLayer, self).__init__()
        self.lstm = nn.LSTM(config['latent_dim'] + config['input_dim'], config['hidden_dim_2'], config['num_layers'], batch_first=True, dropout=config['dropout'])
        self.fc_out = nn.Linear(config['hidden_dim_2'], config['output_len'])

    def forward(self, x, z):
        input = torch.cat((x, z), -1)
        x_ , h = self.lstm(input)
        x_ = x_[:,-1,:]
        out = self.fc_out(x_)
        return out
        
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.ones(32, 48, 8).to(device)

    config = {
        "input_dim": 8,
        "output_dim": 1,
        "input_len": 48,
        "output_len": 5,
        "hidden_dim": 400,
        "latent_dim": 200,
        "hidden_dim_2": 64,
        "num_layers": 2,
        "dropout": 0.4
    }
    model = ImdaVAEModel(config, device ).to(device)
    model_predict = PredictionLayer(config).to(device)
    x_hat, z, mean, log_var  = model(x)
    output = model_predict(z)

    print(x_hat.shape)
    print(z.shape)
    print(output.shape)
