import os
import torch
import torch.optim as optim
import torch.nn as nn
from collections import defaultdict
from utils.utils import process_aerob_dataset
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import TensorDataset, DataLoader


from set_transformer.modules import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

num_inds = 9

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
        print(f"dim_Q = {dim_Q}")
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
        # Apply linear transformations to Q, K, V
        Q = self.fc_q(Q)  # Shape: [batch_size, seq_len, dim_q]
        K, V = self.fc_k(K), self.fc_v(K)  # Shape: [batch_size, seq_len, dim_k], [batch_size, seq_len, dim_v]

        print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
        
        dim_split = self.dim_V // self.num_heads  # Dimensionality per head
        dim = 2  # Sequence length dimension
        
        # Split Q, K, V into multiple heads (Shape: [batch_size * num_heads, seq_len, dim_per_head])
        Q_ = torch.cat(Q.split(dim_split, dim), 0)  # Shape: [batch_size * num_heads, seq_len, dim_q / num_heads]
        K_ = torch.cat(K.split(dim_split, dim), 0)  # Shape: [batch_size * num_heads, seq_len, dim_k / num_heads]
        V_ = torch.cat(V.split(dim_split, dim), 0)  # Shape: [batch_size * num_heads, seq_len, dim_v / num_heads]

        print(f"Q_ shape: {Q_.shape}, K_ shape: {K_.shape}, V_ shape: {V_.shape}")
        KT=  K_.transpose(1,2)
        print(f"dim K_.transpose(1,2) shape = {KT.shape}")
        
        # Compute attention scores: Q_.bmm(K_.transpose(1, 2))]
       # A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        attn_scores = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        print(f"Q_ shape  = {Q_.shape}")
        print(f"attn_scores shape BEF = {attn_scores.shape}")
        
        # Apply the mask to the attention scores if provided
        if mask is not None:
            print(f"mask shape = {mask.shape}")
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))  # Mask padding with -inf
        
        print(f"attn_scores shape AFT = {attn_scores.shape}")    

        # Apply softmax to the attention scores
        A = torch.softmax(attn_scores, dim=-1)  # Softmax over the sequence length dimension (seq_len)
        print(f"A shape: {A.shape}, V_ shape: {V_.shape}")
        
        # Perform the batch matrix multiplication
        result = A.bmm(V_)  # result shape will be [batch_size * num_heads, seq_len, feature_dim]
        print(f"A.bmm(V_) = {result.shape}")
        
        # Reshape result to the original batch_size and sequence length
        result = result.view(Q.size(0), self.num_heads, Q.size(1), result.size(-1))  # Shape: [batch_size, num_heads, seq_len, feature_dim]
        
        # Combine the result (using the `split` and `cat` to gather the heads)
        O = torch.cat((Q_ + result).split(Q.size(0), 0), 2)  # Shape: [batch_size, seq_len, dim_v]

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

        # self.enc = nn.Sequential(
        #         ISAB(embedding_dim, dim_hidden, num_heads, num_inds, ln=ln),
        #         nn.Dropout(0.3))
        # self.dec = nn.Sequential(
        #         PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        #         SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #         nn.Dropout(0.3),
        #         nn.Linear(dim_hidden, dim_output))

        # Directly define layers instead of Sequential
        self.isab = ISAB(embedding_dim, dim_hidden, num_heads, num_inds, ln=ln)
        self.dropout_enc = nn.Dropout(0.3)

        self.pma = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.sab = SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        self.dropout_dec = nn.Dropout(0.3)
        self.fc = nn.Linear(dim_hidden, dim_output)

    def forward(self, X, mask):
        # X_emb = self.embedding(X)  # Convert tokenized COGs to embeddings
        # X_enc = self.enc(X_emb, mask)  # Pass through encoder
        # logits = self.dec(X_enc, mask)  # Pass through decoder
        # return logits# self.dec[1:](output)  # Apply remaining layers
       # return output
        X_emb = self.embedding(X)  # Convert tokenized COGs to embeddings

        # Explicitly pass the mask
        X_enc = self.isab(X_emb, mask)
        X_enc = self.dropout_enc(X_enc)

        # Apply decoder layers with mask where required
        X_dec = self.pma(X_enc, mask)
        X_dec = self.sab(X_dec)  # If `SAB` needs a mask, pass it
        X_dec = self.dropout_dec(X_dec)
        output = self.fc(X_dec)

        return output


MASK_TOKEN_ID = 0  # Using 0 as the mask token (since padding_idx=0 in nn.Embedding)
MASK_PROB = 0.15   # Masking 15% of tokens

def mask_tokens(inputs, mask_token_id=MASK_TOKEN_ID, mask_prob=MASK_PROB):
    """ Randomly mask some tokens for masked language modeling. """
    labels = inputs.clone()  # Create a copy of inputs for the labels
    rand = torch.rand(inputs.shape)  # Generate random numbers
    
    # Create a mask for selecting tokens to replace with [MASK]
    mask = (rand < mask_prob) & (inputs != 0)  # Do not mask padding tokens (0)

    # Replace the selected tokens with [MASK]
    inputs[mask] = mask_token_id
    labels[~mask] = -100  # Ignore non-masked tokens for loss computation
    
    return inputs, labels


# Train 

def train(num_COGs, padded_sequences, attention_masks):
    # Initialize the model
    embedding_dim=128
    dim_output=1
    dim_hidden=64
    num_heads=4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    K=1
    model = SetTransformer(num_COGs=num_COGs, embedding_dim=embedding_dim, num_outputs=K, dim_output=dim_output, num_inds=num_inds, dim_hidden=dim_hidden, num_heads=num_heads)#.cuda()
    model = model.to(device)
    learning_rate = 0.0001
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100) 


    dataset = TensorDataset(padded_sequences, attention_masks)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs): 
        epoch_loss = 0
        for batch in dataloader:
            inputs, masks = batch  # Unpack batch
            inputs, labels = mask_tokens(inputs)  # Apply MLM masking
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(inputs, masks)  # Forward pass

            # Compute loss: Only for masked positions
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))  
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")


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

    train(num_COGs, padded_sequences, attention_masks)
   