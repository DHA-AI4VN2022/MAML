import torch
import torch.nn as nn
import pdb

class AcLSTM(nn.Module):
    def __init__(self, config, device): #input_seq_len, output_seq_len, hidden_size, num_layers, kernel_size, dropout):
        super(AcLSTM, self).__init__()

        self.input_seq_len = config["input_len"]
        self.horizon = config["output_len"]
        self.hidden_size = config['hidden_size']
        self.hidden_size_2 = config['hidden_size_2']
        self.num_layers =  config['num_layers']
        self.kernel_size = config['kernel_size']
        # self.dropout = nn.Dropout(config['dropout'])
        self.input_size = len(config['input_features']) + config['num_input_station'] - 1
        self.output_size = len(config['target_features'])

        self.conv1 = nn.Conv1d(self.input_size, self.hidden_size_2, kernel_size = self.kernel_size, stride = 1, padding = int((self.kernel_size-1)/2))
        self.lstm = nn.LSTM(self.hidden_size_2, self.hidden_size, self.num_layers, batch_first=True, dropout = config['dropout'])
        #self.attention = nn.MultiheadAttention(embed_dim = self.hidden_size, num_heads = 1, batch_first = True, dropout = self.dropout)
        self.attention = nn.MultiheadAttention(embed_dim = self.hidden_size, num_heads = 1, dropout = config['dropout'])
        self.fc = nn.Linear(self.input_seq_len, self.horizon)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc3 = nn.Linear(self.hidden_size, self.horizon)
        
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

        self.device = device    

    def forward(self, input):
        ## x = (batch_size, input_seq_len, input_size(channel))
        ## --> x(batch_size, input_size, input_seq_len)
        #pdb.set_trace()
        x = input.permute(0,2,1)
        ## --> x(batch_size, 128, input_seq_len)
        x = torch.tanh(self.conv1(x))
        # x = self.dropout(x)

        ## h = (num_layers,batch_size,Hout)
        ## c = (num_layers,batch_size,Hcell)
        # h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(self.device)

        ## x = (batch_size, in_seq_len, 128)
        x = x.permute(0,2,1)

        ## x: (batch_size, in_seq_len, hidden_size)
        # x, (h_out, c_out) = self.lstm(x)
        x, (h_out, c_out) = self.lstm(x)
        # x = self.dropout(x)

        # q = self.leakyRelu(self.fc1(x))
        # k = self.leakyRelu(self.fc1(x))
        # v = self.leakyRelu(self.fc1(x))
        # # import pdb; pdb.set_trace()

        # attn_output, attn_output_weights = self.attention(q,k,v)

        # ## x: (batch_size, out_seq_len, hidden_size)
        # x = attn_output.permute(0,2,1)
        # x = self.leakyRelu((self.fc(x)).permute(0,2,1))
        # x = self.dropout(x)
        # ## x: (batch_size, out_seq_len, output_size)
        # out = self.leakyRelu(self.fc2(x))
        # return out
        # import pdb; pdb.set_trace()
        # x  = self.sigmoid(self.fc(x.permute(0,2,1))).permute(0,2,1)
        out = self.sigmoid(self.fc3(x[:,-1,:]))
        return out
        