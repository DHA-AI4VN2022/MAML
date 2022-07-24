import torch 
import torch.nn as nn 
import  yaml
from models.spattrnn.layers import GCN, GAT, TemporalGCN
# from layers import GCN, GAT, TemporalGCN

class SpatioEmbeddedRNN(nn.Module):
  def __init__(self, config, device):
    super(SpatioEmbeddedRNN, self).__init__()
    self.config = config
    self.device = device 
    self.batch_size = config['batch_size']
    self.input_len = config['input_len']
    self.output_len = config['output_len']
    self.input_dim = config['input_dim']
    self.output_dim = config['output_dim']
    self.hidden_dim = config['hidden_dim']
    self.hidden_dim_2 = config['hidden_dim_2']

    self.activation = config['activation']
    self.bias = True
    rnn_type = config['rnn_type']
    dropout_p = config['dropout']

    # self.batch_size = config['data']['batch_size']
    # self.input_len = config['model']['input_len']
    # self.output_len = config['model']['output_len']
    # self.input_dim = config['model']['input_dim']
    # self.output_dim = config['model']['output_dim']
    # self.hidden_dim = config['model']['hidden_dim']
    # self.hidden_dim_2 = config['model']['hidden_dim_2']

    # self.activation = config['model']['activation']
    # self.bias = True
    # rnn_type = config['model']['rnn_type']
    # dropout_p = config['train']['dropout']
    
    self.rnn_type = rnn_type
    self.fc = nn.Linear(self.input_dim, self.hidden_dim)
    self.fc_2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
    self.fc_3 = nn.Linear(self.hidden_dim, self.hidden_dim_2)
    self.fc_4 = nn.Linear(self.hidden_dim_2, self.output_len)
    self.fc_5 = nn.Linear(self.hidden_dim, self.hidden_dim)

    self.bn = nn.BatchNorm2d(self.input_dim)  
    if rnn_type == 'gru':
      self.rnn = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True, dropout=dropout_p)
    elif rnn_type == 'lstm':
      self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, dropout=dropout_p)
    # self.gcn = GCN(self.input_dim, self.hidden_dim)
    self.temporal_gcn = TemporalGCN(self.input_dim, self.hidden_dim, self.batch_size, config=config)
    self.relu = nn.ReLU()

  def forward(self, x, adj, target_station_idx):
    # import pdb; pdb.set_trace()
    # x_spatial = self.bn(x_spatial.permute(0,3,1,2)).permute(0,2,3,1)
    x = x.to(self.device)
    adj =  adj.to(self.device)

    x_temporal = x[:, :, target_station_idx, :]
    x_temporal = x_temporal.squeeze(2)
    
    x_spatial = x[:,:,:,:] # lay timestep cuoi cung 
    h_spatial = torch.zeros(self.batch_size,x_spatial.shape[2], self.hidden_dim).to(self.device) #(b, 207, 32)
    c_spatial = torch.zeros(self.batch_size,x_spatial.shape[2], self.hidden_dim).to(self.device) #(b, 207, 32)
    adj = adj[:, -1, :,:].squeeze(1)
    
    for i in range(x_spatial.shape[1]):
      x_ = x_spatial[:,i,:,:].squeeze(1)  
      h_spatial, c_spatial = self.temporal_gcn(x_, adj, H=h_spatial, C=c_spatial) # batch_size, num_nodes, hidden_dim = 32,33, 64
      
    x_spatial = self.fc_5(self.relu(h_spatial)) # 32, 33, 64
    x_spatial = x_spatial[:, target_station_idx,:].squeeze(1) # 32, 64

    # h_temporal = torch.zeros(1, self.batch_size, self.hidden_dim)
    x_temporal, h_temporal = self.rnn(x_temporal) # x_temporal = 32,24,1 ; h_temporal = 1, 32, 64 
    x_temporal = x_temporal[:, -1,:].squeeze(1) # 32,64

    h_concated = torch.cat((x_temporal, x_spatial), dim=-1)
    h_concated = self.relu(self.fc_2(h_concated.squeeze(0)))
    out_ = self.relu(self.fc_3(h_concated))
    out = self.fc_4(out_)
    return out 

if __name__ == '__main__':
    config_file = '../../config/spatio_attention_embedded_rnn.yml'
    with open(config_file) as f:
        config = yaml.safe_load(f)
    model = SpatioEmbeddedRNN(config)
    print(model.forward( torch.rand(32,7,33,8), torch.rand(32,7,33,33), target_station_idx=5 ))
    