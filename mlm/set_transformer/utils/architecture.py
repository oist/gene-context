import torch
import torch.nn as nn

class SAB(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super(SAB, self).__init__()
        # Multihead Attention block.
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        # LayerNorm applied after adding the attention output (residual connection).
        self.ln1 = nn.LayerNorm(dim)
        # Feed-forward network.
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        # Second LayerNorm applied after adding the feed-forward output (another residual connection).
        self.ln2 = nn.LayerNorm(dim)
    
    def forward(self, X, mask=None):
        # --- Self-Attention with Residual Connection ---
        attn_out, _ = self.mha(X, X, X, key_padding_mask=mask)
        X = self.ln1(X + attn_out)
        
        # --- Feed-Forward Network with Residual Connection ---
        ff_out = self.ff(X)
        out = self.ln2(X + ff_out)
        return out

class PMA(nn.Module):
    def __init__(self, dim, num_seeds, num_heads, dropout=0.0):
        super(PMA, self).__init__()
        # Learnable seed vectors that act as queries for pooling.
        self.num_seeds = num_seeds
        self.seed = nn.Parameter(torch.randn(num_seeds, dim))
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.ln = nn.LayerNorm(dim)
    
    def forward(self, X, mask=None):
        batch_size = X.shape[1]
        # Expand the seed for each example in the batch.
        S = self.seed.unsqueeze(1).expand(-1, batch_size, -1)  # (num_seeds, batch_size, dim)
        # Use S as queries and X as keys/values.
        pooled, _ = self.mha(S, X, X, key_padding_mask=mask)
        # Add skip connection: add the original seed S to the pooled output.
        pooled = pooled + S
        # Normalize the pooled output.
        pooled = self.ln(pooled)
        return pooled
    

class GenomeSetTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_sab=2, dropout=0.1):
        """
        The model architecture:
          - Embedding layer for COG tokens (with an extra pad token).
          - Linear layer to project binary counts into the same dimension.
          - Several SAB blocks (with residual connections) to model interactions.
          - PMA module to pool the set into a fixed-size representation, with a residual skip from the seed.
          - A local skip connection: we average the token features before heavy pooling.
          - The decoder receives the concatenation of global (pooled) and local features.
        """
        super(GenomeSetTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.pad_idx = vocab_size  # Reserve an extra token for padding.
        self.cog_embedding = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=d_model, padding_idx=self.pad_idx)
        # Project the binary count (0/1) into the same embedding space.
        self.count_linear = nn.Linear(1, d_model)
        # Stack of SAB blocks.
        self.sab_blocks = nn.ModuleList([SAB(dim=d_model, num_heads=num_heads, dropout=dropout) for _ in range(num_sab)])
        # PMA module for global pooling.
        self.pma = PMA(dim=d_model, num_seeds=1, num_heads=num_heads, dropout=dropout)
        # Decoder now expects concatenated features (global + local), so input dim is 2*d_model.
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )
    
    def forward(self, tokens, mask):
        """
        tokens: Tensor of shape (B, N, 2)
          - tokens[:,:,0]: COG indices.
          - tokens[:,:,1]: Noisy count values (later thresholded to binary).
        mask: Boolean tensor of shape (B, N) indicating padded positions.
        """
        B, N, _ = tokens.size()
        # Convert first column to long for embedding lookup.
        cog_ids = tokens[:, :, 0].long()
        # Threshold the noisy counts: 1 if > 0, else 0.
        binary_counts = (tokens[:, :, 1].float() > 0).float().unsqueeze(-1)
        # Get embeddings and project counts.
        emb_cog = self.cog_embedding(cog_ids)
        emb_count = self.count_linear(binary_counts)
        # Sum both representations.
        X = emb_cog + emb_count  # Shape: (B, N, d_model)
        
        # --- Local Skip Connection ---
        # Compute a local summary of token features by averaging over the token dimension.
        local_features = X.mean(dim=1)  # Shape: (B, d_model)
        
        # Prepare for attention: transpose to (N, B, d_model).
        X = X.transpose(0, 1)
        for sab in self.sab_blocks:
            X = sab(X, mask=mask)
        # Global pooling via PMA.
        pooled = self.pma(X, mask=mask)  # (num_seeds, B, d_model)
        pooled = pooled.squeeze(0)       # (B, d_model)
        
        # Concatenate the global pooled features with the local features.
        combined = torch.cat([pooled, local_features], dim=1)  # (B, 2*d_model)
        logits = self.decoder(combined)    # (B, vocab_size)
        probs = torch.sigmoid(logits)      # Probabilities in [0,1]
        return probs    