import torch.nn as nn
import torch

class ChebConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, K, device):
        super(ChebConvLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = K
        self.device = device
        self.weights = nn.Parameter(torch.FloatTensor(K, self.input_dim, self.output_dim)).to(device)
        self.init_weights()

    def forward(self, x, laplacian):
        cheb_x = []
        x0 = x
        cheb_x.append(x0)

        if self.K > 1:
            x1 = torch.bmm(laplacian, cheb_x[0])
            cheb_x.append(x1)
            for k in range(2, self.K):
                x = 2 * torch.bmm(laplacian, cheb_x[k - 1]) - cheb_x[k - 2]
                cheb_x.append(x)

        chebyshevs = torch.stack(cheb_x, dim=0)
        if chebyshevs.is_sparse:
            chebyshevs = chebyshevs.to_dense()

        output = torch.einsum('hlij,hjk->lik', chebyshevs, self.weights)
        return output

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weights)


class ChebConvNet(nn.Module):
    def __init__(self, config, device):
        super(ChebConvNet, self).__init__()
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.hidden_dim = config['hidden_dim']
        self.K = config['k']
        self.num_layers = config['num_conv_layers']
        self.device = device
        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config['dropout'])
        self.softmax = nn.Softmax(dim=1)
        assert self.num_layers >= 1, "Number of layers have to be >= 1"

        if self.num_layers == 1:
            self.convs.append(
                ChebConvLayer(self.input_dim, self.output_dim, K=self.K, device=self.device).to(self.device))
        elif self.num_layers >= 2:
            self.convs.append(
                ChebConvLayer(self.input_dim, self.hidden_dim, K=self.K, device=self.device).to(self.device))
            for i in range(self.num_layers - 2):
                self.convs.append(ChebConvLayer(self.hidden_dim, self.hidden_dim, device=self.device).to(self.device))
            self.convs.append(
                ChebConvLayer(self.hidden_dim, self.output_dim, K=self.K, device=self.device).to(self.device))

    def forward(self, x, laplacian):
        for i in range(self.num_layers - 1):
            x = self.dropout(x)
            x = self.convs[i](x, laplacian)
            x = self.relu(x)

        x = self.dropout(x)
        x = self.convs[-1](x, laplacian)
        x = self.softmax(x)
        return x


class LSTM(nn.Module):

    def __init__(self, input_size, config, device):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = config['hidden_lstm_dim']
        self.num_layers = config['num_lstm_layers']
        self.bidirect = config['bidirect'] 
        self.output_size = config['output_len']
        
        self.D = 2 if self.bidirect else 1
        self.device = device
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirect
        )
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)
        self.relu = nn.ReLU()
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        h_0 = torch.zeros(self.D * self.num_layers, x.shape[0], self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.D * self.num_layers, x.shape[0], self.hidden_size).to(self.device)
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn[0].view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc(out)  # Final Output
        return out