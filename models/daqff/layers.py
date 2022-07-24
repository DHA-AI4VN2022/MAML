import torch 
import torch.nn as nn 
import torch.nn.functional as F

class GCN(nn.Module):
  def __init__(self, in_features, out_features, activation="relu", bias=True, config=None):
    super(GCN, self).__init__()
    self.fc = nn.Linear(in_features, out_features)
    self.input_dim = in_features
    self.output_dim = out_features
    if activation=='relu':
      self.activation = nn.ReLU() 
    elif activation=='tanh':
      self.activation = nn.Tanh()
    # import pdb; pdb.set_trace()
    if bias:
      self.bias = nn.Parameter(torch.FloatTensor(out_features))
      self.bias.data.fill_(0.0)
    else:
      self.bias.register_parameter("bias", None)
    #init weight
    for m in self.modules():
      self.weights_init(m)
  
  def weights_init(self, m):
    if isinstance(m,nn.Linear):
      torch.nn.init.xavier_uniform_(m.weight.data)

      if m.bias is not None:
        m.bias.data.fill_(0.0)

  def forward(self, x, adj, sparse=False):
    seq_fts = self.fc(x) # (32, 20, 64)
    if sparse:
      out = torch.unsqueeze(
        torch.bmm(adj, torch.squeeze(adj, torch.squeeze(seq_fts, 0)))
      )
    else:
      out = torch.bmm(adj, seq_fts) #  (32, 20,20) * (32, 20, 64) -> (32, 20, 64)
    if self.bias is not None:
      out += self.bias 
    return out

class GAT(nn.Module):
    """
    GAT layer https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, config, concat=True):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = config['alpha']
        self.dropout = config['dropout']
        self.batch_size = config['batch_size']

        # self.alpha = config['train']['alpha']
        # self.dropout = config['train']['dropout']
        # self.batch_size = config['data']['batch_size']
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(self.batch_size ,in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(self.batch_size, 2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, sparse=False):
        # import pdb; pdb.set_trace()
        # if h.shape[0] != adj.shape[1]:
        Wh = torch.bmm(h, self.W) # h.shape: (B, N, in_features), Wh.shape: (in_ft, out_features)
        # print(Wh.shape)

        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:, :self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[:, self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.permute(0,2,1)
        return self.leakyrelu(e)

class TemporalGCN(torch.nn.Module):
    """Modified temporalgcn.py from 
    https://github.com/benedekrozemberczki/pytorch_geometric_temporal/
    """
    r"""An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        batch_size (int): Size of the batch.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: int, out_channels: int, batch_size: int, improved: bool = False, cached: bool = False, 
                 add_self_loops: bool = True, config=None):
        super(TemporalGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self.config = config
        self._create_parameters_and_layers()

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_f = GAT(in_features=self.in_channels,  out_features=self.out_channels, config=self.config )
        self.linear_wf = torch.nn.Linear(self.out_channels, self.out_channels)
        self.linear_uf = torch.nn.Linear(self.out_channels, self.out_channels)

    def _create_input_gate_parameters_and_layers(self):
        self.conv_i = GAT(in_features=self.in_channels,  out_features=self.out_channels, config=self.config)
        self.linear_wi = torch.nn.Linear(self.out_channels, self.out_channels)
        self.linear_ui = torch.nn.Linear(self.out_channels, self.out_channels)
    
    def _create_output_gate_parameters_and_layers(self):
        self.conv_o = GAT(in_features=self.in_channels,  out_features=self.out_channels, config=self.config)
        self.linear_wo = torch.nn.Linear(self.out_channels, self.out_channels)
        self.linear_uo = torch.nn.Linear(self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_c = GAT(in_features=self.in_channels,  out_features=self.out_channels, config=self.config)
        self.linear_wc = torch.nn.Linear(self.out_channels, self.out_channels)
        self.linear_uc = torch.nn.Linear(self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_forget_gate_parameters_and_layers()
        self._create_input_gate_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H, C):
        if H is None:
            H = torch.zeros(self.batch_size,X.shape[1], self.out_channels).to(X.device)
        if C is None:
            C = torch.zeros(self.batch_size,X.shape[1], self.out_channels).to(X.device)
        return H, C

    def _calculate_forget_gate(self, X, adj, H):
        conv = self.conv_f(X, adj)
        Fo = self.linear_wf(conv) + self.linear_uf(H)
        Fo = torch.sigmoid(Fo)
        return Fo

    def _calculate_input_gate(self, X, adj, H):
        conv = self.conv_i(X, adj)
        In = self.linear_wi(conv) + self.linear_ui(H)
        In = torch.sigmoid(In)
        return In

    def _calculate_output_gate(self, X, adj, H):
        conv = self.conv_i(X, adj)
        Ou = self.linear_wo(conv) + self.linear_uo(H)
        Ou = torch.sigmoid(Ou)
        return Ou

    def _calculate_cell_state(self, X, adj, H):
        conv = self.conv_i(X, adj)
        Ce = torch.tanh(self.linear_wo(conv) + self.linear_uo(H))
        return Ce

    def _calculate_candidate_state(self, C, Fo, In, Ce):
        # C: Ct-1
        # H: Ht-1
        # Fo: ft 
        # In: it 
        # Ce: gt 
        # Ct = ft * ct-1 + it * gt
        C_tilde = Fo * C + In * Ce
        return C_tilde

    def _calculate_hidden_state(self, Ou, Ca):
        Z = Ou * torch.tanh(Ca)
        return Z

    def forward(self,X: torch.FloatTensor, adj: torch.LongTensor,
                H: torch.FloatTensor = None, C: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H, C = self._set_hidden_state(X, H, C)
        Fo = self._calculate_forget_gate(X, adj, H)
        In = self._calculate_input_gate(X, adj, H)
        Ce = self._calculate_cell_state(X, adj, H)
        Ou = self._calculate_output_gate(X, adj, H)
        Ca = self._calculate_candidate_state(C, Fo, In, Ce)
        Z = self._calculate_hidden_state(Ou, Ca) # (b, 207, 32)
        return Z, Ca


if __name__ == '__main__':
    config = {
    'alpha': 0.1,
    'dropout': 0.4,
    'batch_size': 32,
    }
    gcn = TemporalGCN(14,64,batch_size=32, config=config)
    H, C = gcn.forward(torch.rand(32,20,14), torch.rand(32,20,20))
    print(H.shape)
    print(C.shape)