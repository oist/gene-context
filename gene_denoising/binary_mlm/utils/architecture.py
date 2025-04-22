import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

class BinaryMLMModel(nn.Module):
    """
    Transformer-based masked language model (MLM) for binary (presence/absence) sequences.
    
    Based on an ESM-style architecture:
      - Learned token and absolute positional embeddings.
      - A global transformer encoder (no locality bias).
      - Xavier uniform initialization and LayerNorm in transformer layers.
      - An MLM head projecting to 2 classes: 0 (absence) and 1 (presence).
      - Mask token id is 2.
      
    This version uses gradient checkpointing to reduce memory consumption.
    """
    def __init__(self, 
                 vocab_size=3,   # tokens: 0, 1, and mask token (2)
                 embed_dim=512, 
                 num_layers=6, 
                 num_heads=8, 
                 dropout=0.1, 
                 max_seq_len=10000):  # adjust max_seq_len as needed
        super(BinaryMLMModel, self).__init__()
        self.embed_dim = embed_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout,
            activation='gelu'
        )
        # Build the transformer encoder.
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlm_head = nn.Linear(embed_dim, 2)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.position_embedding.weight)
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
                
    def forward(self, input_ids, labels=None):
        batch_size, seq_length = input_ids.shape
        # Create position indices on the same device as input_ids.
        positions = torch.arange(0, seq_length, device=input_ids.device) \
                        .unsqueeze(0).expand(batch_size, seq_length)
        
        token_embeds = self.token_embedding(input_ids)      # [B, L, D]
        pos_embeds = self.position_embedding(positions)       # [B, L, D]
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # Transformer expects input shape [L, B, D]
        x = x.transpose(0, 1)
        # Use gradient checkpointing to reduce memory usage.
        # Break the transformer layers into segments (here, one segment per layer).
        num_layers = len(self.transformer.layers)
        x = checkpoint_sequential(self.transformer.layers, num_layers, x)
        x = x.transpose(0, 1)  # back to [B, L, D]
        
        logits = self.mlm_head(x)  # [B, L, 2]
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1), ignore_index=-100)
            
        return logits, loss