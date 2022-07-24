import torch.nn as nn

class VanillaLSTM(nn.Module):
    def __init__(self, config, device):
        super(VanillaLSTM, self).__init__()
        self.num_layers = config['num_layers']
        self.input_dim = config['input_dim']
        self.output_len = config['output_len']
        self.hidden_dim = config['hidden_dim']
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_len)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        out, (h_out, _) = self.lstm(x)
        out = out[:,-1, :]
        out = self.fc(out)
        return out