import torch
import torch.nn as nn
from torch.nn import Parameter
import math
from torch.autograd import Variable
import torch.nn.functional as F

#encoder
class Encoder(nn.Module):
  def __init__(self,config,device): #,m,T,n
    super(Encoder,self).__init__()
    self.hidden_size = config['hidden_size']
    self.batch_size = config['batch_size']
    self.window_size = config['window_size']
    self.driving_series = config['driving_series']
    self.device = device
    #input attention
    self.Ve = Parameter(torch.Tensor(self.window_size))
    self.We = Parameter(torch.Tensor(self.window_size,2*self.hidden_size))
    self.Ue = Parameter(torch.Tensor(self.window_size,self.window_size))
    self.LSTM1 = nn.LSTM(self.window_size,self.hidden_size,1,batch_first = True)

    #self attention
    self.Wg = Parameter(torch.Tensor(self.hidden_size,self.batch_size))
    self.Wa = Parameter(torch.Tensor(self.batch_size,self.hidden_size))
    self.bg = Parameter(torch.Tensor(self.hidden_size,self.driving_series))
    self.ba = Parameter(torch.Tensor(self.batch_size,self.driving_series))
    self.LSTM2 = nn.LSTM(self.window_size,self.hidden_size,1,batch_first = True)

    #function
    self.tanh = nn.Tanh()
    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.Softmax(dim=1)

    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)
  
  def forward(self,x,h,s):
      batch, T,n = x.size()
      h1 = (h,s)
      h2 = (h,s)
      input_attention = Variable(torch.zeros(batch,n,T)).to(self.device)
      self_attention = Variable(torch.zeros(batch,T,n)).to(self.device)
    # for i in range(batch):
      for j in range(n):
        e_k = self.Ve.T @ self.tanh(self.We@ (torch.cat((h,s),2).transpose(2,1)) + self.Ue@x[:,:,j].T)
        a_k = self.softmax(e_k)
        # print(a_k.shape)
        input_attention[:,j,:] = a_k @ x[:,:,j]
      for d in range(T):
        g_t = self.tanh(self.Wg @ x[:,d,:] + self.bg)
        a_t = self.sigmoid(self.Wa @ g_t + self.ba)
        # print(a_t.shape)
        # print(self_attention[:,d,:].shape)
        self_attention[:,d,:] = a_t * x[:,d,:]
      _,h1 = self.LSTM1(input_attention, h1)
      _,h2 = self.LSTM2(self_attention.transpose(2,1),h2)
      return torch.cat((h1[0],h2[0]),0), h1
    
  def init_hidden(self):
    h = torch.zeros(1,self.batch_size,self.hidden_size).to(self.device)
    c = torch.zeros(1,self.batch_size,self.hidden_size).to(self.device)
    return h,c

#generator
class Generator(nn.Module):
  def __init__(self,config,device): #p,T,n,m,k,w
    super(Generator,self).__init__()
    self.hidden_size = config['hidden_size']
    self.num_filter = config['num_filter']
    self.batch_size = config['batch_size']
    self.window_size = config['window_size']
    self.output_size = 1
    self.hidden_size = config['hidden_size']
    self.kernel_size = config['kernel_size']
    self.device = device
    #Parameter define
    self.vd = Parameter(torch.Tensor(self.hidden_size))
    self.Wd = Parameter(torch.Tensor(self.hidden_size,2*self.hidden_size))
    self.Ud = Parameter(torch.Tensor(self.hidden_size,self.hidden_size))
    self.w = Parameter(torch.Tensor(self.output_size,self.hidden_size + self.window_size))
    self.b = Parameter(torch.Tensor(self.output_size,1))

    #function
    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax(dim=1)
    self.relu = nn.ReLU()

    self.conv = nn.Conv1d(2,self.window_size,self.kernel_size)
    self.LSTM = nn.LSTM(self.output_size,self.hidden_size,1,batch_first = True)

    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self,Z,d,s,y_real):
    # print(Z.shape)
    H = self.relu(self.conv(Z.transpose(1,0)))
    ct = Variable(torch.zeros(self.batch_size,self.hidden_size)).to(self.device)
    for i in range(self.window_size):
      l = self.vd.T @ self.tanh(self.Wd @ (torch.cat((d,s),2).transpose(2,1)) + self.Ud @ (H[:,i,:].T))
      B = self.softmax(l)
      ct += B@H[:,i,:]
      y_fake = self.w @ torch.cat((y_real,ct),1).T + self.b
      _,(d,s) =  self.LSTM(y_fake.view(self.batch_size,1,self.output_size),(d,s))
    return y_fake.view(self.batch_size,self.output_size),(d,s)

  def init_hidden(self):
    h = torch.zeros(1,self.batch_size,self.hidden_size).to(self.device)
    c = torch.zeros(1,self.batch_size,self.hidden_size).to(self.device)
    return h,c

#discriminator
class Discriminator(nn.Module):
  def __init__(self,config):
    super(Discriminator,self).__init__()
    self.output_size = 1
    self.batch_size = config['batch_size']
    self.conv1 = nn.Conv1d(self.output_size,128,1)
    self.conv2 = nn.Conv1d(128,256,1)
    self.conv3 = nn.Conv1d(256,512,1)
    self.linear = nn.Linear(512,self.output_size)
    self.sigmoid = nn.Sigmoid()
  def forward(self,x):
    x = x.view(1,self.output_size,self.batch_size)
    x = self.conv3(self.conv2(self.conv1(x)))
    x = self.linear(x.view(1,self.batch_size,512))
    return self.sigmoid(x)

class MARNN(nn.Module):
  def __init__(self,config,device):
    super(MARNN,self).__init__()
    self.encode = Encoder(config,device)
    self.decode = Generator(config,device)

  def forward(self,x,y_real):
    he,se = self.encode.init_hidden()
    hd,sd = self.decode.init_hidden()
    Z,_ = self.encode(x,he,se)
    y_fake,h = self.decode(Z,hd,sd,y_real)
    return y_fake

class MAGAN(nn.Module):
  def __init__(self,config,device):
    super(MAGAN,self).__init__()
    self.marnn = MARNN(config,device)
    self.dis = Discriminator(config)

  def forward(self,x,y_real):
    out_f = self.marnn(x,y_real)
    out = self.dis(out_f)
    return out