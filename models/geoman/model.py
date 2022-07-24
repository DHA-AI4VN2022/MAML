import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as tf
from models.geoman.layers import Linear
import numpy as np

def param(shape, init):
  res = nn.Parameter(torch.FloatTensor(*shape))
  if init == 0:
    nn.init.zeros_(res)
  elif init == 1:
    nn.init.ones_(res)
  elif init == 2:
    nn.init.xavier_uniform_(res)
  return res

def input_transform(x, ext, device):
  # x: (32, 24, 10, 15) bs, n_steps_encoder, sensors, n_input_encoder
  # ext: (32, 6, 3) bs, n_steps_decoder, n_input_ext
  # encoder_inputs 
      # local_inputs: 32, 15, 24 -> (32, 15) * 24
      # global_target_inputs: 32, 10, 24 -> (32, 10) * 24 
      # global_total_inputs: 32, 10, 15, 24 -> 32 * (10, 15, 24) 
  # decoder_inputs: 32, 1, 6 -> (32,1)* 6 
  # encoder_attention_states: A tuple consisting of
  #             1) local_attention_states: 3D tensor [batch_size, n_input_encoder, n_steps_encoder]
  #             2) global_attention_states: 4D tensor [batch_size, n_sensors, n_input_encoder, n_steps_encoder]
  batch_size = x.size(0)
  n_steps_encoder = x.size(1)
  n_sensors = x.size(2)
  n_input_encoder = x.size(3)
  n_steps_decoder = ext.size(1)
  n_input_ext = ext.size(2)

  _local_inputs  = x[:,:,0,:].permute(1,0,2) # 32, 24, 15 -> 24, 32, 15 
  _local_inputs = _local_inputs.contiguous().view(-1, n_input_encoder).to(device) # 24 * 32, 15
  _local_inputs = torch.split(_local_inputs, batch_size, 0) # 24 * [32,15]

  _global_inputs = x[:,:,:,0].permute(1, 0, 2) # 32, 24, 10 -> 24, 32, 10
  _global_inputs = _global_inputs.contiguous().view(-1, n_sensors).to(device)
  _global_inputs = torch.split(_global_inputs, batch_size, 0)

  encoder_inputs = (_local_inputs, _global_inputs)

  local_attn_states = x[:,:,0,:].permute(0,2,1).to(device) # 32, 24, 15 ->  32, 15, 24
  global_attn_states = x.permute(0,2,3,1).to(device) # 32, 24, 10, 15 -> 32, 10, 15, 24
  encoder_attention_states = (local_attn_states, global_attn_states)

  _decoder_inputs = [param([batch_size, 1], 1).to(device) for i in range(n_steps_decoder)]
  _external_inputs = ext.permute(1, 0, 2) # 32, 6, 3 -> 6, 32, 3
  _external_inputs = _external_inputs.contiguous().view(-1, n_input_ext).to(device)
  _external_inputs = torch.split(_external_inputs, batch_size, 0)
  return encoder_attention_states, encoder_inputs, _external_inputs, _decoder_inputs

class SpatialAttention(nn.Module):
  """ Spatial attention in GeoMAN
      Args:
        encoder_inputs: A tuple consisting of
          1) local_inputs: the inputs of local spatial attention, i.e., a list of 2D tensors with the shape of
            [batch_size, n_inputs_encoder]
          2) global_inputs: the inputs of local spatial attention, i.e., a list of 2D tensors with the shape of
            [batch_size, n_sensors]
        attention_states: A tuple consisting of
          1) local_attention_states: 3D tensor [batch_size, n_input_encoder, n_steps_encoder]
          2) global_attention_states: 4D tensor [batch_size, n_sensors, n_input_encoder, n_steps_encoder]
        cell: core_rnn_cell.RNNCell defining the cell function and size.
        s_attn_flag: 0: only local. 1: only global. 2: local + global.
        output_size: Size of the output vectors; if None, we use cell.output_size.
        loop_function: the loop function we use.
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "spatial_attention".
      Return:
        A tuple of the form (outputs, state), where:
      Raises:
        ValueError: when num_heads is not positive, there are no inputs, shapes
          of attention_states are not set, or input size cannot be inferred from the
          input.
    """
  def __init__(self, config=None, s_attn_flag=2, device=None) -> None:
    super(SpatialAttention, self).__init__()
    self.config = config
    self.device =device 

    # decide whether to use local/global attention
    # s_attn_flag: 0: only local. 1: only global. 2: local + global
    self.local_flag = True
    self.global_flag = True
    if s_attn_flag == 0:
      self.global_flag = False
    elif s_attn_flag == 1:
      self.local_flag = False

    self.batch_size = self.config['batch_size']

    # Define config for local attention vector
    #1) local_attention_states: 3D tensor [batch_size, n_input_encoder, n_steps_encoder]
    # self.local_attn_length = local_attention_states.data.size(1) # n_input_encoder
    # self.local_attn_size = local_attention_states.data.size(2) # n_steps_encoder
    self.local_attn_length = len(self.config['input_features'])# n_input_encoder
    self.local_attn_size = self.config['input_len']  # n_steps_encoder
    self.local_attention_vec_size = self.local_attn_size
    # vector query of local
    self.conv_local_u = nn.Conv2d(self.local_attn_size, self.local_attention_vec_size, (1,1), (1, 1), device=self.device)
    self.linear_local = nn.Linear(self.config['n_hidden_encoder'], self.local_attention_vec_size, bias=True, device=self.device)

    self.local_v = nn.Parameter(torch.FloatTensor(self.local_attention_vec_size)).to(self.device)                   
    init.normal_(self.local_v)


    # Define config for global attention vector 
    # 2) global_attention_states: 4D tensor [batch_size, n_sensors, n_input_encoder, n_steps_encoder]
    # self.global_attn_length = global_attention_states.data.size(1) # n_input_encoder
    # self.global_n_input = global_attention_states.data.size(2)
    # self.global_attn_size = global_attention_states.data.size(3) # n_steps_encoder
    self.global_attn_length = self.config['num_station'] # n_input_encoder
    self.global_n_input = len(self.config['input_features'])
    self.global_attn_size = self.config['input_len'] 
    self.global_attention_vec_size = self.global_attn_size
    
    # Size of query vectors for attention.
    self.conv_global_k = nn.Conv2d(self.global_attn_size, self.global_attention_vec_size, (1,self.global_n_input), (1, 1), device=self.device)
    self.linear_global = nn.Linear( self.config['n_hidden_encoder'], self.global_attention_vec_size, bias=True, device=self.device)

    self.global_v = nn.Parameter(torch.FloatTensor(self.global_attention_vec_size)).to(self.device)
    init.normal_(self.global_v)                        
    

  def local_attention(self, query, local_attention_states ):
    local_hidden = local_attention_states.contiguous().view(-1, self.local_attn_size, self.local_attn_length, 1)
    local_hidden_features = self.conv_local_u(local_hidden)
    # Calc Wl[ht-1; st-1]
    y = self.linear_local(query)
    y  = y.view(-1, 1, 1, self.local_attention_vec_size)
    # Attention mask is a softmax of v_lT * tanh(...)
    s = torch.sum(self.local_v * torch.tanh(local_hidden_features + y), dim=[1, 3])
    # Now calculate the attention-weighted vector, i.e., alpha in eq.[2]
    a = tf.softmax(s, dim=1)
    return a 

  def global_attention(self, query, distmat, global_attention_states):
    # A trick: to calculate Wg * Xl *ug by a 1-by-1 convolution
    global_hidden = global_attention_states.contiguous().view(-1, self.global_attn_size, self.global_attn_length, self.global_n_input)
    global_hidden_features = self.conv_global_k(global_hidden)
    # Calc Wl[ht-1; st-1]
    distmat = distmat.to(self.device)
    y = self.linear_global(query)
    y = y.view(-1, 1, 1, self.global_attention_vec_size)

    g = torch.sum(self.global_v * torch.tanh(global_hidden_features + y), dim=[1,3])
    lamda = self.config['lamda']
    g = (1- lamda) * g +  lamda * distmat
    # g[:, 0] =  0 # 0 mean target station
    a = tf.softmax(g, dim=1)
    return a

  def forward(self, encoder_inputs, attention_states, cell, distmat):
    local_inputs = encoder_inputs[0]
    global_inputs = encoder_inputs[1]

    local_attention_states = attention_states[0].to(self.device)
    global_attention_states = attention_states[1].to(self.device)
        
    outputs = []
    attn_weights = []
    i = 0

    local_attn = nn.Parameter(torch.FloatTensor(self.batch_size, self.local_attn_length)).to(self.device)
    init.xavier_uniform_(local_attn)
    global_attn = nn.Parameter(torch.FloatTensor(self.batch_size, self.global_attn_length)).to(self.device)
    init.xavier_uniform_(global_attn)

    for local_inp, global_inp in zip(local_inputs, global_inputs):
      local_inp = local_inp.to(self.device)
      global_inp = global_inp.to(self.device)
      
      if self.local_flag and self.global_flag:
        # multiply attention weights with the original input
        local_x = local_attn * local_inp
        global_x = global_attn * global_inp

        #concat local with global
        cated_x = torch.cat([local_x, global_x], 1)
        cell_output, state = cell(cated_x)
        local_attn = self.local_attention(state,local_attention_states)
        global_attn = self.global_attention(state, distmat, global_attention_states)
        attn_weights.append((local_attn, global_attn))
      elif self.local_flag:
        local_x = local_attn * local_inp
        cell_output, state = cell(local_x)
        local_attn = self.local_attention(state, local_attention_states)
        attn_weights.append(local_attn)
      elif self.global_flag:
        global_x = global_attn * global_inp
        cell_output, state = cell(global_x)
        global_attn = self.global_attention(state, global_attention_states)
        attn_weights.append(global_attn)

      # Attention output projection
      output = cell_output
      outputs.append(output)
      i += 1
    return outputs, state, attn_weights

class TemporalAttention(nn.Module):
  """ Temporal attention in GeoMAN
    Args:
      decoder_inputs: A list (length: n_steps_decoder) of 2D Tensors [batch_size, n_input_decoder].
      external_inputs: A list (length: n_steps_decoder) of 2D Tensors [batch_size, n_external_input].
      encoder_state: 2D Tensor [batch_size, cell.state_size].
      attention_states: 3D Tensor [batch_size, n_step_encoder, n_hidden_encoder].
      cell: core_rnn_cell.RNNCell defining the cell function and size.
      output_size: Size of the output vectors; if None, we use cell.output_size.
      external_flag: whether to use external factors
    Return:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as the inputs of decoder of 2D Tensors of
                  shape [batch_size x output_size]
        state: The state of each decoder cell the final time-step.
  """
  def __init__(self, config, external_flag, output_size=64, device=None) -> None:
    super(TemporalAttention, self).__init__()
    self.config = config 
    self.device = device 

    self.batch_size = self.config['batch_size']
    self.attn_length = self.config['input_len']
    self.attn_size = self.config['n_hidden_decoder']
    self.output_size = output_size
    self.external_flag = external_flag

    self.attention_vec_size = self.attn_size

    # Calc Wd H0
    self.w_conv = nn.Conv2d(self.attn_size, self.attention_vec_size, (1,1), (1,1), device=self.device)
    self.v = nn.Parameter(torch.FloatTensor(self.attention_vec_size)).to(self.device)
    init.normal_(self.v )       
    
    self.linear_attention  =  nn.Linear( self.config['n_hidden_decoder'], self.attention_vec_size, bias=True, device=self.device)
    if self.external_flag:
      self.linear_concat = nn.Linear(self.config['n_input_decoder'] + len(self.config['external_features']) + self.attn_size, self.config['n_input_decoder'], bias=True, device=self.device)
    else:
      self.linear_concat = nn.Linear(self.config['n_input_decoder'] + self.attn_size, self.config['n_input_decoder'], bias=True, device=self.device)
    self.linear_output = nn.Linear(self.config['n_hidden_decoder'] + self.attn_size, self.output_size, bias=True, device=self.device)


  def attention(self, query, attention_states):
    hidden = attention_states.view(-1, self.attn_size, self.attn_length, 1) # need to reshape before
    # A trick: to calculate W_d * h_o by a 1-by-1 convolution
    # See at eq.[6] in the paper
    # Size of query vectors for attention.
    hidden_features = self.w_conv(hidden)  # Wd H0
    #v = Variable(torch.zeros(attention_vec_size)) # v_l 

    # Calc Wd' [dt-1; st-1]
    y = self.linear_attention(query)
    y = y.view(-1, 1, 1, self.attention_vec_size) 

    # Attention mask is a softmax of v_d^{\top} * tanh(...).
    s = torch.sum(self.v * torch.tanh(hidden_features + y), dim=[1, 3])
    # Now calculate the attention-weighted vector, i.e., gamma in eq.[7]
    a = tf.softmax(s, dim=1)
    # eq. [8]
    # Calc weighted context vector c'
    d = torch.sum(a.view(-1, 1, self.attn_length, 1)* hidden, dim=[2, 3])
    return d.view(-1, self.attn_size)

  def forward(self, decoder_inputs, external_inputs, encoder_state, attention_states, cell):    
    i = 0
    outputs = []
    prev = None

    for (inp, ext_inp) in zip(decoder_inputs, external_inputs):
      inp = inp.to(self.device)
      ext_inp = ext_inp.to(self.device)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.data.size(1)
      # we map the concatenation to shape [batch_size, input_size]

      attn = nn.Parameter(torch.FloatTensor(self.batch_size, self.attn_size)).to(self.device)
      init.xavier_uniform_(attn)    
      if self.external_flag:
        x = torch.cat([inp, ext_inp, attn], dim=1)
        x = self.linear_concat(x)
      else:
        x = torch.cat([inp, attn], dim=1)
        x = self.linear_concat(x)

      # Run RNN
      cell_output, state = cell(x)
      # define attention vector
      
      attn = self.attention(state, attention_states) # c't

      # Attention output 
      output_ = torch.cat([cell_output, attn], dim=1) # [d't; c't]
      output = self.linear_output(output_)
      outputs.append(output)
      i+=1
    return outputs, state

class GeoMan2(nn.Module):
  def __init__(self, config, device):
    super(GeoMan2, self).__init__()
    self.config = config 
    self.device= device

    self.linear_out = nn.Linear(self.config['output_size'], self.config['n_output_decoder'], bias=True, device=self.device)
    # self.w_out = nn.Parameter(torch.FloatTensor(config['n_hidden_decoder'], config['n_output_decoder'])).to(self.device)
    # self.b_out = nn.Parameter(torch.FloatTensor(config['n_output_decoder'])).to(self.device)

    self.encoder_cell = nn.LSTMCell(len(config['input_features'])+ config['num_station'] , config['n_hidden_encoder'], bias=True, device=self.device)
    self.decoder_cell = nn.LSTMCell(config['n_input_decoder'], config['n_hidden_decoder'], bias=True,  device=self.device)

    # init.xavier_uniform_(self.w_out)
    # init.normal_(self.b_out)                    

    self.spatial_attention = SpatialAttention(config=config, device=device)      
    self.temporal_attention = TemporalAttention(config=config, external_flag=self.config['ext_flag'], device=device)

  def forward(self, x, ext, distmat):
    encoder_attention_states, encoder_inputs, _external_inputs, decoder_inputs \
            = input_transform(x, ext, self.device)
    encoder_outputs, encoder_state, attn_weights \
            = self.spatial_attention(encoder_inputs,
                encoder_attention_states,
                self.encoder_cell,
                distmat)
    # Calculate concatenation of encoder outputs to put attention on.
    top_states = [e.view(-1, 1, 64) for e in encoder_outputs]
    attention_states = torch.cat(top_states, 1)

    decoder_outputs, states = self.temporal_attention(decoder_inputs,
                                                      _external_inputs,
                                                      encoder_state,
                                                      attention_states,
                                                      self.decoder_cell)

    preds = [self.linear_out(i) for i in decoder_outputs]
    outs =  torch.cat(preds, dim=1)
    return outs

  # def spatial_attention(self, encoder_inputs, attention_states, cell, distmat, s_attn_flag=2, output_size=64):
  #   """ Spatial attention in GeoMAN
  #       Args:
  #         encoder_inputs: A tuple consisting of
  #           1) local_inputs: the inputs of local spatial attention, i.e., a list of 2D tensors with the shape of
  #             [batch_size, n_inputs_encoder]
  #           2) global_inputs: the inputs of local spatial attention, i.e., a list of 2D tensors with the shape of
  #             [batch_size, n_sensors]
  #         attention_states: A tuple consisting of
  #           1) local_attention_states: 3D tensor [batch_size, n_input_encoder, n_steps_encoder]
  #           2) global_attention_states: 4D tensor [batch_size, n_sensors, n_input_encoder, n_steps_encoder]
  #         cell: core_rnn_cell.RNNCell defining the cell function and size.
  #         s_attn_flag: 0: only local. 1: only global. 2: local + global.
  #         output_size: Size of the output vectors; if None, we use cell.output_size.
  #         loop_function: the loop function we use.
  #         dtype: The dtype to use for the RNN initial state (default: tf.float32).
  #         scope: VariableScope for the created subgraph; default: "spatial_attention".
  #       Return:
  #         A tuple of the form (outputs, state), where:
  #       Raises:
  #         ValueError: when num_heads is not positive, there are no inputs, shapes
  #           of attention_states are not set, or input size cannot be inferred from the
  #           input.
  #   """
  #   if not encoder_inputs:
  #     raise ValueError(
  #       "Must provide at least 1 input to attention encoder.")
  #   local_inputs = encoder_inputs[0]
  #   global_inputs = encoder_inputs[1]

  #   local_attention_states = attention_states[0].to(self.device)
  #   global_attention_states = attention_states[1].to(self.device)

  #   batch_size = local_inputs[0].size(0)

  #   # decide whether to use local/global attention
  #   # s_attn_flag: 0: only local. 1: only global. 2: local + global
  #   local_flag = True
  #   global_flag = True
  #   if s_attn_flag == 0:
  #       global_flag = False
  #   elif s_attn_flag == 1:
  #       local_flag = False

  #   if local_flag:
  #     local_attn_length = local_attention_states.data.size(1) # n_input_encoder
  #     local_attn_size = local_attention_states.data.size(2) # n_steps_encoder
    
  #     # calc Ul * xi,k using 1-by-1 convolution
  #     local_hidden = local_attention_states.contiguous().view(-1, local_attn_size, local_attn_length, 1)
  #     # Size of query vectors for attention.
  #     local_attention_vec_size = local_attn_size
  #     # vector query of local
  #     local_u = nn.Conv2d(local_attn_size, local_attention_vec_size, (1,1), (1, 1), device=self.device)
  #     #print(local_hidden.size())
  #     local_hidden_features = local_u(local_hidden.float())

  #     # Weight v
  #     local_v = nn.Parameter(torch.FloatTensor(local_attention_vec_size)).to(self.device)                   
  #     local_attn = nn.Parameter(torch.FloatTensor(batch_size, local_attn_length)).to(self.device)
  #     init.normal_(local_v)
  #     init.xavier_uniform_(local_attn)

  #     def local_attention(query):
  #       # Calc Wl[ht-1; st-1]
  #       y = Linear(query, local_attention_vec_size, True, device=self.device)
  #       y  = y.view(-1, 1, 1, local_attention_vec_size)
  #       # Attention mask is a softmax of v_lT * tanh(...)
  #       s = torch.sum(local_v * torch.tanh(local_hidden_features + y), dim=[1, 3])
  #       # Now calculate the attention-weighted vector, i.e., alpha in eq.[2]
  #       a = tf.softmax(s, dim=1)
  #       return a 

  #   if global_flag: # batchsize, num_feat, num_stats, seq_len
  #     global_attn_length = global_attention_states.data.size(1) # n_input_encoder
  #     global_n_input = global_attention_states.data.size(2)
  #     global_attn_size = global_attention_states.data.size(3) # n_steps_encoder

  #     # A trick: to calculate Wg * Xl *ug by a 1-by-1 convolution
  #     global_hidden = global_attention_states.contiguous().view(-1, global_attn_size, global_attn_length, global_n_input)
  #     # Size of query vectors for attention.
  #     global_attention_vec_size = global_attn_size
  #     global_k = nn.Conv2d(global_attn_size, global_attention_vec_size, (1,global_n_input), (1, 1), device=self.device)
  #     global_hidden_features = global_k(global_hidden.float())

  #     global_v = nn.Parameter(torch.FloatTensor(global_attention_vec_size)).to(self.device)
  #     global_attn = nn.Parameter(torch.FloatTensor(batch_size, global_attn_length)).to(self.device)
  #     init.normal_(global_v)                        
  #     init.xavier_uniform_(global_attn)

  #     def global_attention(query, distmat):
  #       # Calc Wl[ht-1; st-1]
  #       distmat = distmat.to(self.device)
  #       y = Linear(query, global_attention_vec_size, True, device=self.device)
  #       y = y.view(-1, 1, 1, global_attention_vec_size)

  #       g = torch.sum(global_v * torch.tanh(global_hidden_features + y), dim=[1,3])
  #       lamda = self.config['lamda']
  #       g = (1- lamda) * g +  lamda * distmat
  #       # g[:, 0] =  0 # 0 mean target station
  #       a = tf.softmax(g, dim=1)
  #       return a

  #   outputs = []
  #   attn_weights = []
  #   i = 0

  #   for local_inp, global_inp in zip(local_inputs, global_inputs):
  #     local_inp = local_inp.to(self.device)
  #     global_inp = global_inp.to(self.device)
      
  #     if local_flag and global_flag:
  #       # multiply attention weights with the original input
  #       local_x = local_attn * local_inp.float()
  #       global_x = global_attn * global_inp.float()

  #       #concat local with global
  #       cated_x = torch.cat([local_x, global_x], 1)
  #       cell_output, state = cell(cated_x)
  #       local_attn = local_attention([state])
  #       global_attn = global_attention([state], distmat)
  #       attn_weights.append((local_attn, global_attn))
  #     elif local_flag:
  #       local_x = local_attn * local_inp
  #       cell_output, state = cell(local_x).to(self.device)
  #       local_attn = local_attention([state])
  #       attn_weights.append(local_attn)
  #     elif global_flag:
  #       global_x = global_attn * global_inp
  #       cell_output, state = cell(global_x).to(self.device)
  #       global_attn = global_attention([state])
  #       attn_weights.append(global_attn)

  #     # Attention output projection
  #     output = cell_output
  #     outputs.append(output)
  #     i += 1
  #   return outputs, state, attn_weights

  # def temporal_attention(self, decoder_inputs, external_inputs, encoder_state, attention_states,
  #                       cell, external_flag, output_size=64):
  #   """ Temporal attention in GeoMAN
  #     Args:
  #       decoder_inputs: A list (length: n_steps_decoder) of 2D Tensors [batch_size, n_input_decoder].
  #       external_inputs: A list (length: n_steps_decoder) of 2D Tensors [batch_size, n_external_input].
  #       encoder_state: 2D Tensor [batch_size, cell.state_size].
  #       attention_states: 3D Tensor [batch_size, n_step_encoder, n_hidden_encoder].
  #       cell: core_rnn_cell.RNNCell defining the cell function and size.
  #       output_size: Size of the output vectors; if None, we use cell.output_size.
  #       external_flag: whether to use external factors
  #     Return:
  #       A tuple of the form (outputs, state), where:
  #         outputs: A list of the same length as the inputs of decoder of 2D Tensors of
  #                   shape [batch_size x output_size]
  #         state: The state of each decoder cell the final time-step.
  #   """
  #   # Needed for reshaping.
  #   batch_size = decoder_inputs[0].data.size(0)
  #   attn_length = attention_states.data.size(1)
  #   attn_size = attention_states.data.size(2)

  #   # A trick: to calculate W_d * h_o by a 1-by-1 convolution
  #   # See at eq.[6] in the paper
  #   hidden = attention_states.view(-1, attn_size, attn_length, 1) # need to reshape before
  #   # Size of query vectors for attention.
  #   attention_vec_size = attn_size

  #   # Calc Wd H0
  #   w_conv = nn.Conv2d(attn_size, attention_vec_size, (1,1), (1,1), device=self.device)
  #   hidden_features = w_conv(hidden)  # Wd H0
  #   #v = Variable(torch.zeros(attention_vec_size)) # v_l

  #   v = nn.Parameter(torch.FloatTensor(attention_vec_size)).to(self.device)
  #   init.normal_(v)       

  #   def attention(query):
  #     # Calc Wd' [dt-1; st-1]
  #     y = Linear(query, attention_vec_size, True, device=self.device)
  #     y = y.view(-1, 1, 1, attention_vec_size) 

  #     # Attention mask is a softmax of v_d^{\top} * tanh(...).
  #     s = torch.sum(v * torch.tanh(hidden_features + y), dim=[1, 3])
  #     # Now calculate the attention-weighted vector, i.e., gamma in eq.[7]
  #     a = tf.softmax(s, dim=1)
  #     # eq. [8]
  #     # Calc weighted context vector c'
  #     d = torch.sum(a.view(-1, 1, attn_length, 1)* hidden, dim=[2, 3])
  #     return d.view(-1, attn_size)
    
  #   # define attention vector
  #   attn = nn.Parameter(torch.FloatTensor(batch_size, attn_size)).to(self.device)
  #   init.xavier_uniform_(attn)    
                      
  #   i = 0
  #   outputs = []
  #   prev = None

  #   for (inp, ext_inp) in zip(decoder_inputs, external_inputs):
  #     inp = inp.to(self.device)
  #     ext_inp = ext_inp.to(self.device)
  #     # Merge input and previous attentions into one vector of the right size.
  #     input_size = inp.data.size(1)
  #     # we map the concatenation to shape [batch_size, input_size]
  #     if external_flag:
  #       x = Linear([inp.float()] + [ext_inp.float()] + [attn.float()], input_size, True, device=self.device)
  #     else:
  #       x = Linear([inp.float()] + [attn.float()], input_size, True, device=self.device)
  #     # Run RNN
  #     cell_output, state = cell(x)
  #     import pdb; pdb.set_trace()
  #     attn = attention([state])

  #     # Attention output 
  #     output = Linear([cell_output] + [attn], output_size, True, device=self.device)
  #     outputs.append(output)
  #     i+=1
  #   return outputs, state

if __name__ == '__main__':
    # encoder_inputs 
      # local_inputs: 32, 19, 12 -> (32, 19) * 12 
      # global_target_inputs: 32, 35, 12 -> (32, 35) * 12 
      # global_total_inputs: 32, 35, 19, 12 -> 32 * (35, 19, 12) 
    # decoder_inputs: 32, 1, 6 -> (32,1)* 6 
    # external_inputs: 32, 83, 6 -> (32,83) * 6
    # encoder_attention_states
      # local_attention_states: 32, 19, 12 -> 32 * (19, 12)
      # global_attention_states: 32, 35, 19, 12 -> 32 * (35, 19, 12)
    import yaml
    config_file = "../../config/geoman.yml"
    with open(config_file, encoding="utf8") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoMan2(config, device=device).to(device)
    # batchsize, num_feat, num_stats, seq_len
    input = torch.rand(32, 24, 35, 19).to(device)
    ext = torch.rand(32, 6, 3)
    distmat = torch.rand(35)
    output = model(input, ext, distmat)

# DEVICE = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

# class ENC_LOC_ATTN(nn.Module):
#   def __init__(self, config):
#     super(ENC_LOC_ATTN, self).__init__()
#     self.v = param([config['input_len']], 1)
#     self.W = param([config['input_len'], 2 * config['hid_dim']], 2)
#     self.conv = nn.Conv1d(config['input_len'], config['input_len'], 1)
#   def cal_state(self, X): 
#     self.state = self.conv(X)
#     self.state.transpose_(1, 2)
#   def forward(self, X, inp):
#     tmp = (self.W @ inp).transpose(1, 2)
#     e = torch.tanh(tmp + self.state) @ self.v
#     return X * nn.functional.softmax(e, dim=1)

# class ENC_GLO_ATTN(nn.Module):
#   def __init__(self, config):
#     super(ENC_GLO_ATTN, self).__init__()
#     self.v = param([config['input_len']], 1)
#     self.W1 = param([config['input_len'], 2*config['hid_dim']], 2)
#     self.conv = nn.Conv1d(config['input_len'], config['input_len'], 1)
#     self.W2 = param([config['input_len'], config['input_dim']], 2)
#     self.u = param([config['input_len']], 1)
#     self.lamda = config['lamda']
    
#   def cal_state(self, target, glo_feat):
#     tmp = self.conv(target).transpose(1, 2)
#     self.state = tmp + self.W2 @ glo_feat @ self.u 
  
#   def forward(self, target, dist_mat, inp):
#     tmp = (self.W1 @ inp).transpose(1, 2) 
#     g = torch.tanh(tmp + self.state) @ self.v
#     g = self.lamda * g + (1-self.lamda) * dist_mat
#     g[:, 0] = 0# 0 mean target station
#     return target * nn.functional.softmax(g, dim=1)

# class TEM_ATTN(nn.Module):
#   def __init__(self, config):
#     super(TEM_ATTN, self).__init__()
#     self.v = param([config['hid_dim']], 1)
#     self.W = param([config['hid_dim'], 2*config['hid_dim']], 2)
#     self.conv = nn.Conv1d(config['hid_dim'], config['hid_dim'], 1)
#   def cal_state(self, X):
#     self.state = self.conv(X)
#     self.state.transpose_(1, 2)
#   def forward(self, inp, enc_hid):
#     tmp = (self.W @ inp.unsqueeze(-1)).transpose(1, 2)
#     u = torch.tanh(tmp + self.state) @ self.v
#     tmp = enc_hid * (nn.functional.softmax(u, dim=1).unsqueeze(-1))
#     return torch.sum(tmp, 1)

# class Linear(nn.Module):
#   def __init__(self, config):
#     super(Linear, self).__init__()
#     self.linear = nn.Linear(2 * config['hid_dim'], config['hid_dim'])
#     self.v = param([config['hid_dim']], 1)
#     self.b = param([1], 0)

#   def forward(self, X):
#     return self.linear(X) @ self.v + self.b 

# class GeoMan(nn.Module):
#   def __init__(self, config):
#     super(GeoMan, self).__init__()
#     self.enc_len, self.dec_len = config['input_len'], config['output_len']
#     self.enc_loc_attn = ENC_LOC_ATTN(config)
#     self.enc_glo_attn = ENC_GLO_ATTN(config)
#     self.tem_attn = TEM_ATTN(config)
#     self.linear = Linear(config)
#     self.enc_lstm = nn.LSTMCell(config['num_station']+config['input_dim'], config['hid_dim'])
#     self.dec_lstm = nn.LSTMCell(config['ext_dim']+config['hid_dim']+1, config['hid_dim'])
#     self.linear = Linear(config)
#     self.config = config
   
#   def forward(self, X, ext, dist_mat):
#     import pdb; pdb.set_trace()
#     loc_feat = X[:, :, 0, :] # 32, 48, 5, 7 - bs, nneigh, ts, fts
#     self.enc_loc_attn.cal_state(loc_feat)
#     target = X[:, :, :, 0]
#     glo_feat = X.permute(0, 2, 3, 1)
#     self.enc_glo_attn.cal_state(target, glo_feat)
#     batch_size, hid_dim = self.config['batch_size'], self.config['hid_dim']
#     h = torch.ones([batch_size, hid_dim], device=DEVICE)
#     c = torch.ones([batch_size, hid_dim], device=DEVICE)
#     X = torch.transpose(X, 0, 1)
#     enc_hid = []
#     for i in range(self.enc_len):
#       inp = torch.concat([h, c], 1).unsqueeze(-1)
#       loc_feat = X[i, :, 0, :]
#       x_local = self.enc_loc_attn(loc_feat, inp) #
#       target = X[i, :, :, 0]
#       x_global = self.enc_glo_attn(target, dist_mat, inp) #
#       x = torch.concat([x_local, x_global], 1)
#       h, c = self.enc_lstm(x, (h, c))
#       enc_hid.append(h.unsqueeze(0))
    
#     enc_hid = torch.concat(enc_hid, 0)
#     tmp = enc_hid.permute(1, 2, 0)
#     self.tem_attn.cal_state(tmp)
    
#     d = torch.ones((batch_size, hid_dim), device=DEVICE)
#     s = torch.ones((batch_size, hid_dim), device=DEVICE)
#     enc_hid = enc_hid.transpose(0, 1)
#     y = torch.ones((batch_size, 1), device=DEVICE)
#     ext = torch.transpose(ext, 0, 1)
    
#     res = []
#     for i in range(self.dec_len):
#       inp = torch.concat([d, s], 1)
#       c = self.tem_attn(inp, enc_hid)

#       inp = torch.concat([y, ext[i], c], 1)
#       d, s = self.dec_lstm(inp, [d, s])
      
#       inp = torch.concat([d, c], 1)
#       y = self.linear(inp).unsqueeze(-1)
#       res.append(y)
#     return torch.concat(res, 1)
