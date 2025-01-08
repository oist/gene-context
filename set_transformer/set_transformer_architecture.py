import torch.nn as nn

from set_transformer.modules import *



num_inds = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=num_inds, dim_hidden=64, num_heads=4, ln=True):  #num_inds is the number of inducing points m
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
                nn.Dropout(0.3))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Dropout(0.3),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):

        return self.dec(self.enc(X))    