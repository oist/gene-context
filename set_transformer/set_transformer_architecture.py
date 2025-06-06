import torch.nn as nn

from set_transformer.modules import *



num_inds = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        


class SetTransformer(nn.Module): #768 512
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=num_inds, dim_hidden=128, num_heads=4, ln=True):  #num_inds is the number of inducing points m dim_hidden=512 ln=True
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
              #  ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
                nn.Dropout(0.3))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Dropout(0.3),
                nn.Linear(dim_hidden, dim_output))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization (also known as Glorot initialization)
                torch.nn.init.xavier_uniform_(m.weight)  # Initialize weights using Xavier uniform
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)  # Initialize biases to zero

    def forward(self, X):
        return self.dec(self.enc(X))    


class SetTransformerWrapper(nn.Module):
    def __init__(self, model, output_class):
        super(SetTransformerWrapper, self).__init__()
        self.model = model
        self.output_class = output_class

    def forward(self, x):
        logits = self.model(x)  # Get model outputs (logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits to probabilities
        return probs[:, self.output_class]  # Select the class output as probability
        