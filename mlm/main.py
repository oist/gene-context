import os
import torch
import torch.nn as nn
from collections import defaultdict
from utils.utils import process_aerob_dataset
from torch.nn.utils.rnn import pad_sequence


from set_transformer.modules import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

num_inds = 10 

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False): #dim_out is the the dimensionality of the embeddings output default is 128
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out)).to(device) #The first dimension is for batching ?
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        mtr = self.I.repeat(X.size(0), 1, 1)
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask).to(device)
        return self.mab1(X, H, mask)


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

    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        dim = 2 
        Q_ = torch.cat(Q.split(dim_split, dim), 0)
        K_ = torch.cat(K.split(dim_split, dim), 0)
        V_ = torch.cat(V.split(dim_split, dim), 0)
        attn_scores = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)

        # Apply the mask to the attention scores
        if mask is not None:
            # Mask should have the shape (batch_size, seq_len) and needs to be expanded for multi-heads
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))  # Mask padding with -inf

        A = torch.softmax(attn_scores, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, mask)

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask)       

class SetTransformer(nn.Module): #768 512
    def __init__(self, num_COGs, embedding_dim, num_outputs, dim_output,
            num_inds=num_inds, dim_hidden=32, num_heads=4, ln=True):  #num_inds is the number of inducing points m dim_hidden=512 ln=True
        super(SetTransformer, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_COGs, embedding_dim=embedding_dim, padding_idx=0)

        self.enc = nn.Sequential(
                ISAB(embedding_dim, dim_hidden, num_heads, num_inds, ln=ln),
                nn.Dropout(0.3))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Dropout(0.3),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X, mask):
        X_emb = self.embedding(X)  # Convert tokenized COGs to embeddings
        X_enc = self.enc(X_emb, mask)  # Pass through encoder
        output = self.dec(X_enc, mask)  # Pass through decoder
        return self.dec[1:](output)  # Apply remaining layers
       # return output


# Train 

def train(num_COGs, padded_sequences, attention_masks):
    num_epochs = 10
    loss_function = torch.nn.BCEWithLogitsLoss()
    

    embedding_dim=128
    dim_output=1
    dim_hidden=64
    num_heads=4

    net = SetTransformer(num_COGs=num_COGs, embedding_dim=embedding_dim, num_outputs=K, dim_output=dim_output, num_inds=num_inds, dim_hidden=dim_hidden, num_heads=num_heads)#.cuda()
    net = net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=Parameters.learning_rate, weight_decay=0.01)


    train_dataset = COGDataset(padded_sequences, labels, attention_masks)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs): 
        epoch_loss = 0
        for batch_idx, (cog_sequences, labels, attention_masks) in enumerate(train_loader):
            cog_sequences, labels, attention_masks = cog_sequences.to(device), labels.to(device), attention_masks.to(device)


if __name__ == '__main__':
    # 1. Process input parameters
    # Parameters = process_args()

    data_filename_train = "data_aerob/all_gene_annotations.added_incompleteness_and_contamination.training.tsv"
    y_filename = "data_aerob/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv"

        # 2. Process train and test datasets
    X_train, X_train_column_names, y_train, d_gtdb_train = process_aerob_dataset(data_filename_train, y_filename, device, remove_noise=True)

    # print(X_train.detach().cpu().numpy())

    X_train = X_train.detach().cpu().numpy()

    cog_vocab = defaultdict()
    cog_ind = 1
    for cog_name in X_train_column_names:
        if cog_name not in cog_vocab.keys():
            cog_vocab[cog_name] = cog_ind
            cog_ind += 1
    
    cog_tokens = []
    for sample in X_train:
        #cog_tokens.append([])
        sequence = [cog_vocab[X_train_column_names[i]] for i in range(len(sample)) if sample[i] > 0]
        cog_tokens.append(torch.tensor(sequence))

    padded_sequences = pad_sequence(cog_tokens, batch_first=True, padding_value=0)
    print(f"padded_sequences.shape = {padded_sequences.shape}")

    attention_masks = (padded_sequences != 0).long()
    print(attention_masks)




    embedding_dim = 128  # Size of embedding vectors
    num_COGs = len(cog_vocab) + 1  # +1 for padding token

    print(f"num_COGs = {num_COGs}")

    embedding_layer = nn.Embedding(num_COGs, embedding_dim, padding_idx=0)
    embedded = embedding_layer(padded_sequences)
    print(embedded.shape) 
   