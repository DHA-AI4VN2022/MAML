import torch.nn as nn
import torch
from models.gc_lstm.layers import ChebConvNet, ChebConvLayer, LSTM

class GC_LSTM(nn.Module):
    """ The main model for prediction.
    Input: graph signal X: (batch_size, input_time_steps,num_stations, num_features)
              Laplacian: (batch_size,input_time_steps,num_stations,num_stations)
       Output: The predicted PM 2.5 index: (batch_size, output_time_steps)"""
    def __init__(self, config, device):
        super(GC_LSTM, self).__init__()
        self.num_stations = config['num_input_station']
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.hidden_dim = config['hidden_dim']
        self.K = config['k']
        self.num_conv_layers = config['num_conv_layers']
        self.num_input_steps = config['input_len']
        self.dropout = config['dropout']
        self.bidirect = config['bidirect']
        self.device = device
        self.gcns = nn.ModuleList()

        # Declare a graph convolution layer for each input time step t
        for i in range(self.num_input_steps):
            self.gcns.append(ChebConvNet(config, device=self.device))

        self.lstm = LSTM(input_size=self.num_stations * (self.input_dim + self.output_dim)
                         , config=config, device=self.device)

    # x: (batch_size, input_time_steps, num_station, num_features)
    def forward(self, x, laplacian):
        assert self.num_input_steps == x.shape[1]
        lstm_inputs = []
        for t in range(self.num_input_steps):
            # batch_wise operation xt: (batch_size,num_time_steps, N, num_features)
            xt = x[:, t, :, :]
            laplacian_t = laplacian[:, t, :, :]
            # Passing the input graph signal Xt through corresponding graph convolution layers
            ht = self.gcns[t](xt, laplacian_t)
            # Concatenate the graph signal Xt and the graph convolution output Ht to be LSTM inputs
            lstm_input = torch.cat((xt, ht), dim=2)
            # Flatten the concatenated vector
            lstm_input = lstm_input.view(lstm_input.shape[0], lstm_input.shape[1] * lstm_input.shape[2])
            # Stack all lstm inputs at different time steps
            lstm_inputs.append(lstm_input)
        lstm_inputs = torch.stack(lstm_inputs, dim=1)
        # Pass through lstm  + fc layer
        output = self.lstm(lstm_inputs)

        return output

if __name__ == '__main__':
    import yaml
    config_file = "../../config/gc_lstm.yml"
    with open(config_file, encoding="utf8") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GC_LSTM(config, device=device).to(device)
    demo_input = torch.rand(32, 24, 10, 15).to(device)
    demo_laplacian = torch.rand(32, 24, 10, 10).to(device)
    output = model(demo_input, demo_laplacian)
    print("Output shape: ", output.shape)
