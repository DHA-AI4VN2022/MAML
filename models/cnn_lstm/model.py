import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, config, device):
        super(CNNLSTM, self).__init__()
        self.num_layers = config['num_layers']
        self.input_size = config['input_dim']
        self.output_len = config['output_len']
        self.dropout = config['dropout']
        self.batch_size = config['batch_size']
        self.hidden_dim = config['hidden_dim']

        self.conv1 = TimeDistributed(nn.Conv1d(self.input_size, 64, kernel_size=2))
        self.conv2 = TimeDistributed(nn.Conv1d(64, 64, kernel_size=2))
        self.maxpool = TimeDistributed(nn.MaxPool1d(2))
        self.flatten = TimeDistributed(nn.Flatten())
        self.dropout_layer = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(64, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_len)
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_maxpool = self.maxpool(x_conv2)
        x_flatten = self.flatten(x_maxpool)
        x_dropout = self.dropout_layer(x_flatten)     
        x_review = x_dropout.view(self.batch_size, 64, -1)
        out_lstm, (h_out, _) = self.lstm(torch.transpose(x_review, 1, 2))      # turn into lstm required input form 
        out = out_lstm[:, -1, :]
        out = self.relu(self.fc(out))
        return out.view(self.batch_size, -1)

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, time_steps, input_size) '''
        batch_size, time_steps, input_size = x.size()
        c_in = torch.transpose(x,2,1)
        c_out = self.module(c_in)

        if len(c_out.size()) > 2 : 
            return torch.transpose(c_out, 2,1)
        else :
            return c_out