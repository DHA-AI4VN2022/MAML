import torch 
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self, config):
    super(Encoder,self).__init__()
    self.input_size = config['input_dim']
    self.hidden_size = config['hidden_dim']
    self.num_layers = config['num_layers']
    self.dropout = config['dropout']
    self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
  
  def forward(self, x):
    # x = (batch_size, input_seq_len, input_size)
    # (h0, c0) : default filled with 0, h0, c0 : (num_layers, batch_size, hidden_size)
    out, state = self.lstm(x)      
    # state: (num_layers, batch_size, hidden_size)
    # out: (batch_size, input_seq_len, hidden_size)
    
    return out, state   

class Decoder(nn.Module):
  def __init__(self, config):
    super(Decoder, self).__init__()
    self.output_size = config['output_dim']
    self.hidden_size = config['hidden_dim']
    self.num_layers = config['num_layers']
    self.dropout = config['dropout']
    
    self.lstm = nn.LSTM(input_size=self.output_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
    self.fc = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, x, state):
    # hidden = h_out
    # cell = c_out
    # x: (batch_size, 1, output_size)   -> 1 input at a time, seq_len=1, initialize to 0
    # state: (num_layers, batch_size, hidden_size)

    output, hidden_state = self.lstm(x, state)   # state: (num_layers, batch_size, hidden_size)
    # output: (batch_size, seq_length=1, hidden_size)
    out = self.fc(output)   
    # out: (batch_size, seq_length=1, output_size)
    return out, hidden_state

class EncoderDecoder(nn.Module):
  def __init__(self, config, device):
    super(EncoderDecoder, self).__init__()
    self.input_size = config['input_dim']
    self.output_size = config['output_dim']
    self.input_seq_len = config['input_len']
    self.output_seq_len = config['output_len']
    self.hidden_size = config['hidden_dim']
    self.num_layers = config['num_layers']
    self.dropout = config['dropout']
    self.batch_size = config['batch_size']
    self.lr_decay_ratio = config['lr_decay_ratio']

    self.encoder = Encoder(config)
    self.decoder = Decoder(config)
    self.device = device
  
  def forward(self, x):
    outputs = torch.zeros(self.batch_size, self.output_seq_len, self.output_size).to(self.device)
      
    encoder_output, encoder_hidden = self.encoder(x)

    decoder_input = torch.zeros(self.batch_size, 1, self.output_size).to(self.device)
    decoder_hidden = encoder_hidden

    for t in range(self.output_seq_len):
      decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
      # import pdb; pdb.set_trace()
      outputs[:,t,:] = decoder_output.squeeze(1)
      decoder_input = decoder_output
    outputs = outputs.squeeze(-1)
    return outputs