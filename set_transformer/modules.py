import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()

        self.dim_V = dim_V
        self.num_heads = num_heads

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):

        Q = self.fc_q(Q)
        
        K, V = self.fc_k(K), self.fc_v(K)
       # print(f"Q shape = {Q.shape}; K shape = {K.shape}; V shape = {V.shape}")

        dim_split = self.dim_V // self.num_heads
       # print(f"dim_split = {dim_split}")
        dim = 2 #2
        Q_ = torch.cat(Q.split(dim_split, dim), 0)
        K_ = torch.cat(K.split(dim_split, dim), 0)
        V_ = torch.cat(V.split(dim_split, dim), 0)

      #  print(f"Q_ shape = {Q_.shape}; K_ shape = {K_.shape}; V_ shape = {V_.shape}")
       # print(f"self.dim_V = {self.dim_V}")

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
      #  print(f"O shape = {O.shape}")
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
       # print("Running SAB")
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X) #this is what is passes to MAB forward

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False): #dim_out is the the dimensionality of the embeddings output default is 128
        super(ISAB, self).__init__()
     #   print(f"dim_in = {dim_in}; dim_out = {dim_out}")
       # print("Running ISAB")
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out)).to(device) #The first dimension is for batching ?
     #   print(f"self.I shape = {self.I.shape}")
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        #print(f"forw in ISAB X shape = {X.shape}")
        mtr = self.I.repeat(X.size(0), 1, 1)
        
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X).to(device)
     #   print(f"H shape = {H.shape}")
     #   print("_______<")
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
       # print("Running PMA")
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
       # print(f"self.S shape = {self.S.shape}")
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
