import torch
import torch.nn as nn
from torch.nn import Parameter
import math
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, config, device):  # ,m,T,n
        super(Encoder, self).__init__()
        self.hidden_size = config['hidden_size']
        self.window_size = config['input_len']
        self.driving_series = config['input_dim']
        self.device = device

        self.input_lstm = nn.LSTM(self.driving_series, self.hidden_size, 1, batch_first=True).to(self.device)

        #     self.We = nn.Linear(self.hidden_size * 2, self.window_size, bias = False).to(self.device)
        #     self.Ue = nn.Linear(self.window_size, self.window_size, bias = False).to(self.device)
        #     self.Ve = nn.Linear(self.window_size, 1, bias = False).to(self.device)
        self.We = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False).to(self.device)
        self.Ue = nn.Linear(self.driving_series, self.hidden_size, bias=False).to(self.device)
        self.Ve = nn.Linear(self.hidden_size, 1, bias=False).to(self.device)

        self.LSTM1 = nn.LSTM(self.driving_series, self.hidden_size, 1, batch_first=True).to(self.device)

        # self attention
        #     self.Wg = nn.Linear(in_features = self.driving_series, out_features  = self.hidden_size).to(self.device)
        #     self.Wa = nn.Linear(in_features = self.hidden_size, out_features  = self.driving_series).to(self.device)
        self.Wg = nn.Linear(in_features=self.driving_series, out_features=self.hidden_size).to(self.device)
        self.Wa = nn.Linear(in_features=self.hidden_size, out_features=self.driving_series).to(self.device)

        self.LSTM2 = nn.LSTM(self.driving_series, self.hidden_size, 1, batch_first=True).to(self.device)

        # function
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        batch_size, T, n = x.size()
        h, s = self.init_hidden(batch_size)
        h1 = (h, s)
        h2 = (h, s)
        input_attention = torch.zeros(batch_size, T, n).to(self.device)

        # Calculate input attention
        #       output, (_,s) = self.input_lstm(x)
        # Initialize the hidden state and cell state (1, batch_size, hidden_size)
        (h, s) = (h, s)
        for t in range(T):
            # x_t: input at time step t (batch_size, input_features)
            x_t = x[:, t, :].unsqueeze(1)
            # Concatenate the hidden state and cell state [h,s] (batch_size, hidden_size * 2)
            h_s = torch.cat((h.squeeze(0), s.squeeze(0)), -1)
            h_s = h_s.unsqueeze(1).repeat(1, self.window_size, 1)
            # Calculate the attention weights
            e_t = self.Ve(self.tanh(self.We(h_s) + self.Ue(x)))
            a_t = self.softmax(e_t.squeeze(-1)).unsqueeze(1)
            # Calculate input attention
            input_attention[:, t, :] = (a_t @ x).squeeze(1)
            # Pass through an lstm layer to get the current hidden state and cell state
            _, (h, s) = self.input_lstm(x_t, (h, s))

        # Calculate self attention
        a = []
        for d in range(T):
            # input at time step t (batch_size, input_features)
            xt = x[:, d, :]
            g_t = self.tanh(self.Wg(xt))
            a_t = self.sigmoid(self.Wa(g_t))
            a.append(a_t)
        a = torch.stack(a, dim=2)
        self_attention = a * x.transpose(2, 1)

        # Pass input attention and self attention through 2 lstms
        output_1, h1 = self.LSTM1(input_attention, h1)
        output_2, h2 = self.LSTM2(self_attention.transpose(2, 1), h2)

        # Concatenate 2 lstm outputs as latent space Z (batch_size, time_steps,hidden_size * 2)
        z = torch.cat((output_1, output_2), dim=-1)
        return z

    def init_hidden(self, batch_size):
        h = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        return h, c

class TimeDistributedCNN(nn.Module):

    def __init__(self, module, batch_first=False):
        super(TimeDistributedCNN, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, time_steps, num_channels, input_size) '''
        batch_size, time_steps, num_channels, input_size = x.size()
        c_in = x.view(batch_size * time_steps, num_channels, input_size)
        c_out = self.module(c_in)
#         print(c_out.shape)
        r_in = c_out.view(batch_size, time_steps, c_out.shape[1], c_out.shape[2])
        if self.batch_first is False:
            r_in = r_in.permute(1, 0, 2)

        return r_in


# generator
class Generator(nn.Module):
    def __init__(self, config, device):  # p,T,n,m,k,w
        super(Generator, self).__init__()
        self.hidden_size = config['hidden_size']
        self.num_filter = config['num_filter']
        #     self.batch_size = config['batch_size']
        self.window_size = config['input_len']
        self.output_size = config['output_len']
        self.hidden_size = config['hidden_size']
        self.kernel_size = config['kernel_size']
        self.device = device
        # Parameter define
        #     self.Wd = nn.Linear(self.hidden_size * 2, self.hidden_size, bias = False).to(self.device)
        #     self.Ud = nn.Linear(self.hidden_size, self.hidden_size, bias = False).to(self.device)
        #     self.vd = nn.Linear(self.hidden_size, 1, bias = False).to(self.device)
        #     self.output_linear = nn.Linear((self.hidden_size) + 1, 1).to(self.device)
        self.Wd = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False).to(self.device)
        self.Ud = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        self.vd = nn.Linear(self.hidden_size, 1, bias=False).to(self.device)
        self.output_linear = nn.Linear((self.hidden_size) + 1, 1).to(self.device)

        # function
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.conv = nn.Conv1d(1, self.num_filter, self.kernel_size, padding='same').to(self.device)
        self.distributed_conv = TimeDistributedCNN(self.conv, batch_first=True)
        self.LSTM = nn.LSTM(1, self.hidden_size, 1, batch_first=True).to(self.device)
        self.conv_linear = nn.Linear(self.num_filter * self.hidden_size * 2, self.hidden_size).to(self.device)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, Z, y_real):
        batch_size = Z.shape[0]
        # Pass the latent space through time distributed conv
        Z = Z.unsqueeze(2)
        H = self.relu(self.distributed_conv(Z))
        # H shape (batch_size,time_steps, num_filters, hidden_size * 2)
        # Reshape H to (batch_size, time_steps, hidden_size)

        H = H.view(H.shape[0], H.shape[1], H.shape[2] * H.shape[3])
        H = self.relu(self.conv_linear(H))

        # Attention and predict y
        d, s = self.init_hidden(batch_size)
        previous_y = y_real.unsqueeze(-1)
        predictions = []
        for t in range(self.output_size):
            # Concat previous hidden state and cell state
            d_s = torch.cat((d, s), 2).squeeze(0)
            d_s = d_s.unsqueeze(1).repeat(1, self.window_size, 1)
            # Calculate attention
            l_t = self.vd(self.tanh(self.Wd(d_s) + self.Ud(H)))
            b_t = self.softmax(l_t.squeeze(-1))
            b_t = b_t.unsqueeze(1)
            # Calculate context vector
            c_t = (b_t @ H).squeeze(1)
            # Calculate prediction at time step t
            y_t = self.output_linear(torch.cat((c_t, previous_y), dim=1))
            predictions.append(y_t)
            y_t = y_t.unsqueeze(-1)
            # Pass through an LSTM layer to get current hidden state, cell state
            _, (d, s) = self.LSTM(y_t, (d, s))
        predictions = torch.cat(predictions, dim=1)
        return predictions

    def init_hidden(self, batch_size):
        h = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        return h, c


# discriminator
class Discriminator(nn.Module):
    def __init__(self, config, device):
        super(Discriminator, self).__init__()
        self.output_size = config['output_len']
        #     self.batch_size = config['batch_size']
        self.device = device
        self.kernel_size = config['kernel_size']
        self.conv1 = nn.Sequential(nn.Conv1d(1, 32, self.kernel_size, padding='same').to(self.device), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, self.kernel_size, padding='same').to(self.device), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, self.kernel_size, padding='same').to(self.device), nn.LeakyReLU())
        self.linear = nn.Linear(128 * self.output_size, 1).to(self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.linear(x.view(batch_size, 128 * self.output_size))
        return self.sigmoid(x)


