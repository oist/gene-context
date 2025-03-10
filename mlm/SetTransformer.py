#!/usr/bin/env python
# coding: utf-8
"""
Complete script for training a Set Transformer model to reconstruct full genome COG profiles.
This script:
  - Loads eggNOG and metadata from CSV/TSV files.
  - Processes the data into a fixed-length genome × COG matrix.
  - Splits the data by taxonomy.
  - Simulates false negatives (dropped genes) and false positives (spurious genes).
  - Defines a Set Transformer model to reconstruct the true binary COG profile.
  
The encoding and loss are as follows:
  - Each genome is represented as a binary vector y \in {0,1}^V (V = number of COGs).
  - The observed input tokens are pairs [COG_index, noisy_count]. Here noisy_count is
    thresholded to binary (i.e. 1 if > 0, else 0).
  - The reconstruction uses BCE loss:
      L = - ( y * log(p) + (1-y) * log(1-p) ) averaged over all V COGs.
  
Tested with Python 3.7.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

############################################
# 1. Data Processing Functions
############################################

def process_eggnog_and_metadata(eggnog_csv, ar_metadata_tsv, bac_metadata_tsv):
    # Load and filter eggNOG data for COG/arCOG entries.
    print_to_file("Loading eggNOG CSV data...")
    df_eggnog = pd.read_csv(eggnog_csv)
    print_to_file("  Total eggNOG records loaded: {}".format(len(df_eggnog)))
    
    print_to_file("Filtering eggNOG records for COG/arCOG entries...")
    cog_mask = df_eggnog['eggNOG_OGs'].str.startswith("COG") | df_eggnog['eggNOG_OGs'].str.startswith("arCOG")
    df_eggnog = df_eggnog[cog_mask]
    print_to_file("  Records after filtering: {}".format(len(df_eggnog)))
    
    print_to_file("Pivoting eggNOG data to build a gene count table per accession...")
    df_pivot = df_eggnog.pivot_table(
        index='acc', 
        columns='eggNOG_OGs', 
        values='count', 
        aggfunc='sum', 
        fill_value=0
    )
    df_pivot = df_pivot.reset_index().rename(columns={'acc': 'accession'})
    print_to_file("  Pivoted gene count table shape: {}".format(df_pivot.shape))
    
    # Load metadata for archaeal and bacterial genomes.
    print_to_file("Loading archaeal metadata...")
    df_ar = pd.read_csv(ar_metadata_tsv, sep="\t", low_memory=False)
    print_to_file("  Archaeal metadata records: {}".format(len(df_ar)))
    
    print_to_file("Loading bacterial metadata...")
    df_bac = pd.read_csv(bac_metadata_tsv, sep="\t", low_memory=False)
    print_to_file("  Bacterial metadata records: {}".format(len(df_bac)))
    
    meta_cols_needed = ['accession', 'gtdb_taxonomy', 
                        'checkm_completeness', 'checkm_contamination', 
                        'coding_bases', 'genome_size', 'gc_percentage']
    print_to_file("Selecting key metadata columns...")
    df_ar = df_ar[meta_cols_needed]
    df_bac = df_bac[meta_cols_needed]
    
    print_to_file("Combining archaeal and bacterial metadata...")
    df_meta = pd.concat([df_ar, df_bac], ignore_index=True)
    
    print_to_file("Merging gene count table with metadata...")
    merged_df = pd.merge(df_pivot, df_meta, how='left', on='accession')
    print_to_file("  Merged data shape: {}".format(merged_df.shape))
    missing_meta = merged_df['gtdb_taxonomy'].isna().sum()
    if missing_meta > 0:
        print_to_file("  WARNING: {} accessions are missing GTDB taxonomy metadata.".format(missing_meta))
    
    taxonomy_levels = ["domain", "phylum", "class", "order", "family", "group", "species"]
    print_to_file("Splitting 'gtdb_taxonomy' into taxonomy columns:")
    print_to_file("  Taxonomy levels: {}".format(taxonomy_levels))
    
    def split_taxonomy(tax_str):
        if pd.isna(tax_str):
            return [None] * len(taxonomy_levels)
        parts = tax_str.split(';')
        split_parts = parts[:len(taxonomy_levels)]
        if len(split_parts) < len(taxonomy_levels):
            split_parts += [None] * (len(taxonomy_levels) - len(split_parts))
        return split_parts

    taxonomy_splits = merged_df['gtdb_taxonomy'].apply(split_taxonomy)
    df_tax = pd.DataFrame(taxonomy_splits.tolist(), columns=taxonomy_levels)
    merged_df = pd.concat([merged_df, df_tax], axis=1)
    print_to_file("  After adding taxonomy columns, data shape is: {}".format(merged_df.shape))
    
    pivot_cog_columns = [col for col in df_pivot.columns if col != "accession"]
    cog_columns = pivot_cog_columns
    
    global_vocab = sorted(cog_columns)
    cog2idx = {cog: i for i, cog in enumerate(global_vocab)}
    print_to_file("Created global vocabulary ({} tokens) and cog2idx mapping.".format(len(global_vocab)))
    print_to_file("Data processing complete.")
    return merged_df, global_vocab, cog2idx

def subsample_and_split_by_taxonomy(data, subsample_fraction=0.1, taxonomic_level="group", test_fraction=0.2, random_state=None):
    print_to_file("Starting subsample_and_split_by_taxonomy...")
    print_to_file("  Total genomes before subsampling: {}".format(data.shape[0]))
    
    print_to_file("  Subsampling {}% of the genomes...".format(subsample_fraction * 100))
    subsampled = data.sample(frac=subsample_fraction, random_state=random_state).reset_index(drop=True)
    print_to_file("  Genomes after subsampling: {}".format(subsampled.shape[0]))
    
    if taxonomic_level not in subsampled.columns:
        raise ValueError("Taxonomic level '{}' not found in data columns.".format(taxonomic_level))
    
    unique_groups = subsampled[taxonomic_level].dropna().unique()
    print_to_file("  Found {} unique groups at taxonomic level '{}'.".format(len(unique_groups), taxonomic_level))
    
    rng = np.random.default_rng(random_state)
    shuffled_groups = list(unique_groups)
    rng.shuffle(shuffled_groups)
    
    total_genomes = subsampled.shape[0]
    target_test_count = test_fraction * total_genomes
    print_to_file("  Target test set size: ~{} genomes".format(int(target_test_count)))
    
    test_groups = []
    current_test_count = 0
    for group in shuffled_groups:
        group_count = subsampled[subsampled[taxonomic_level] == group].shape[0]
        if current_test_count < target_test_count:
            test_groups.append(group)
            current_test_count += group_count
        else:
            break
    
    test_set = subsampled[subsampled[taxonomic_level].isin(test_groups)].reset_index(drop=True)
    train_set = subsampled[~subsampled[taxonomic_level].isin(test_groups)].reset_index(drop=True)
    
    print_to_file("  Test set: {} genomes from {} groups; Train set: {} genomes".format(
        test_set.shape[0], len(test_groups), train_set.shape[0]))
    
    train_groups = set(train_set[taxonomic_level].unique())
    test_groups_set = set(test_set[taxonomic_level].unique())
    if train_groups.intersection(test_groups_set):
        print_to_file("  WARNING: Overlap detected in taxonomic groups between train and test sets!")
    else:
        print_to_file("  No overlap in taxonomic groups between train and test sets.")
    
    print_to_file("Subsample and split complete.\n")
    return train_set, test_set

############################################
# 2. Dataset and Collate Function
############################################

class GenomeDataset(Dataset):
    def __init__(self, df, global_vocab, cog2idx,
                 false_negative_rate=0.3, false_positive_rate=0.005,
                 count_noise_std=0.0, random_state=None):
        """
        Each genome’s gene profile is encoded as a binary vector (1: present, 0: absent).
        To simulate experimental noise:
          - True positives (genes present) are dropped with probability false_negative_rate.
          - False positives are added from the absent genes, with count determined by a Poisson
            process (with rate false_positive_rate * vocab_size).
        Additionally, count noise is added by multiplying the true count with a noise factor.
        """
        self.df = df.reset_index(drop=True)
        self.global_vocab = global_vocab
        self.cog2idx = cog2idx
        self.vocab_size = len(global_vocab)
        self.false_negative_rate = false_negative_rate
        self.false_positive_rate = false_positive_rate
        self.count_noise_std = count_noise_std
        self.rng = np.random.default_rng(random_state)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get the gene counts and convert to binary target (presence/absence)
        row = self.df.iloc[idx]
        counts = row[self.global_vocab].values.astype(np.float32)
        target = (counts > 0).astype(np.float32)
        
        observed_indices = []
        observed_counts = []
        # For each gene in the target, decide whether to keep it (simulate false negatives)
        for cog_idx, present in enumerate(target):
            if present:
                # With probability false_negative_rate, drop this gene.
                if self.rng.random() < self.false_negative_rate:
                    continue
                original_count = counts[cog_idx]
                # Add noise to the count (even though later we threshold to binary)
                noise = 1;#self.rng.normal(loc=1.0, scale=self.count_noise_std)
                noisy_count = max(original_count * noise, 0.0)
                observed_indices.append(cog_idx)
                observed_counts.append(noisy_count)
        
        # Simulate false positives: add a small number of genes that are truly absent.
        num_false_positives = self.rng.poisson(lam=self.false_positive_rate * self.vocab_size)
        absent_indices = np.where(target == 0)[0]
        if len(absent_indices) > 0 and num_false_positives > 0:
            false_pos = self.rng.choice(absent_indices, size=min(num_false_positives, len(absent_indices)), replace=False)
            for fp in false_pos:
                noisy_count = 1;#abs(self.rng.normal(loc=1.0, scale=self.count_noise_std))
                observed_indices.append(fp)
                observed_counts.append(noisy_count)
        
        # Build token array: each token = [COG_index, noisy_count]
        if len(observed_indices) == 0:
            tokens = np.empty((0, 2), dtype=np.float32)
        else:
            tokens = np.stack([np.array(observed_indices, dtype=np.int64),
                               np.array(observed_counts, dtype=np.float32)], axis=-1)
        
        sample = {
            'tokens': tokens,
            'target': target  # Ground truth binary vector (V,)
        }
        return sample

def collate_genomes(batch, pad_idx):
    """
    Pads the list of token arrays to the same length for batching.
    Also stacks the ground truth target vectors.
    """
    batch_tokens = [sample['tokens'] for sample in batch]
    targets = [sample['target'] for sample in batch]
    batch_size = len(batch_tokens)
    max_len = max(tokens.shape[0] for tokens in batch_tokens)
    # Initialize a padded array for tokens; using 0 for the count and pad_idx for the COG id.
    tokens_padded = np.full((batch_size, max_len, 2), fill_value=0, dtype=np.float32)
    # mask==True indicates a padded position.
    mask = np.ones((batch_size, max_len), dtype=bool)
    
    for i, tokens in enumerate(batch_tokens):
        length = tokens.shape[0]
        if length > 0:
            tokens_padded[i, :length, :] = tokens
            mask[i, :length] = False
        if length < max_len:
            tokens_padded[i, length:, 0] = pad_idx  # Fill COG id with pad token index.
    tokens_padded = torch.tensor(tokens_padded)
    mask = torch.tensor(mask)
    targets = torch.tensor(np.stack(targets, axis=0), dtype=torch.float32)
    return tokens_padded, mask, targets


# In[4]:


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


# In[12]:


############################################
# 4. Training and Validation (with Noise and FN Evaluation)
############################################

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_metrics_extended(model, dataloader, device, threshold=0.5):
    """
    Evaluate the model over the given dataloader and compute:
      - Per-genome Accuracy, Precision, Recall, F1 based on thresholded predictions.
      - Average Genome Size Difference (absolute difference between number of positives in target and prediction).
      - FP Noise Removed Fraction: Among genes observed in the noisy input that are false positives
        (observed but truly absent), the fraction correctly predicted as absent.
      - FN Noise Recovered Fraction: Among genes that were missed in the noisy input (false negatives)
        (not observed but truly present), the fraction correctly predicted as present.
    """
    model.eval()
    all_preds = []         # predicted probabilities, shape (N, V)
    all_targets = []       # ground truth binary labels, shape (N, V)
    abs_genome_size_diff_list = []
    fp_removed_list = []   # List to store per-genome FP noise removal fraction
    fn_recovered_list = [] # List to store per-genome FN noise recovery fraction
    
    for tokens, mask, targets in tqdm(dataloader, desc="Evaluating (extended)"):
        tokens = tokens.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            preds = model(tokens, mask)  # shape: (B, vocab_size)
        # Store predictions and targets for overall metrics.
        all_preds.append(preds.cpu().detach().numpy())
        all_targets.append(targets.cpu().detach().numpy())
        
        # Convert predictions to binary using the threshold.
        preds_np = (preds.cpu().detach().numpy() >= threshold).astype(int)
        targets_np = targets.cpu().detach().numpy().astype(int)
        batch_size = targets_np.shape[0]
        
        # Compute genome size difference per sample.
        for i in range(batch_size):
            true_size = np.sum(targets_np[i])
            pred_size = np.sum(preds_np[i])
            abs_diff = np.abs(true_size / pred_size)
            abs_genome_size_diff_list.append(abs_diff)
        
        # --- Compute noise correction metrics ---
        # For each sample in the batch, reconstruct the observed input (before noise correction)
        # from the tokens using the provided mask.
        tokens_np = tokens.cpu().detach().numpy()  # Shape: (B, N, 2)
        mask_np = mask.cpu().detach().numpy()        # Shape: (B, N)
        for i in range(batch_size):
            # Initialize an empty observed vector for this genome.
            observed = np.zeros(model.vocab_size, dtype=int)
            # For valid (unpadded) tokens, mark the gene as observed.
            valid_tokens = tokens_np[i][mask_np[i] == False]
            for token in valid_tokens:
                cog_id = int(token[0])
                # Make sure not to include pad tokens.
                if cog_id < model.vocab_size:
                    observed[cog_id] = 1
            # FP noise: observed but should be 0 in the target.
            fp_noise = (observed == 1) & (targets_np[i] == 0)
            # FN noise: not observed but should be 1 in the target.
            fn_noise = (observed == 0) & (targets_np[i] == 1)
            pred_binary = preds_np[i]
            # Compute fraction of FP noise that were correctly removed (predicted 0).
            if np.sum(fp_noise) > 0:
                fp_removed = np.sum((pred_binary == 0) & fp_noise) / np.sum(fp_noise)
                fp_removed_list.append(fp_removed)
            # Compute fraction of FN noise that were recovered (predicted 1).
            if np.sum(fn_noise) > 0:
                fn_recovered = np.sum((pred_binary == 1) & fn_noise) / np.sum(fn_noise)
                fn_recovered_list.append(fn_recovered)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    binary_preds = (all_preds >= threshold).astype(int)
    N, V = all_targets.shape
    
    per_sample_acc, per_sample_prec, per_sample_rec, per_sample_f1 = [], [], [], []
    
    # Compute per-sample classification metrics.
    for i in range(N):
        TP = np.sum((all_targets[i] == 1) & (binary_preds[i] == 1))
        TN = np.sum((all_targets[i] == 0) & (binary_preds[i] == 0))
        FP = np.sum((all_targets[i] == 0) & (binary_preds[i] == 1))
        FN = np.sum((all_targets[i] == 1) & (binary_preds[i] == 0))
        acc = (TP + TN) / V
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_sample_acc.append(acc)
        per_sample_prec.append(prec)
        per_sample_rec.append(rec)
        per_sample_f1.append(f1)
    
    # Average the noise correction metrics over samples that had any noise.
    avg_fp_removed = np.mean(fp_removed_list) if fp_removed_list else float('nan')
    avg_fn_recovered = np.mean(fn_recovered_list) if fn_recovered_list else float('nan')
    
    metrics = {
        "avg_accuracy": np.mean(per_sample_acc),
        "avg_precision": np.mean(per_sample_prec),
        "avg_recall": np.mean(per_sample_rec),
        "avg_f1": np.mean(per_sample_f1),
        "avg_genome_size_diff": np.mean(abs_genome_size_diff_list),
        "avg_fp_removed_fraction": avg_fp_removed,
        "avg_fn_recovered_fraction": avg_fn_recovered
    }
    return metrics


############################################
# Modified Training and Validation Function
############################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# --- Helper Loss Functions ---
def soft_f1_loss(y_true, y_pred, eps=1e-7):
    """
    Computes a differentiable approximation of the F1 loss.
    y_true and y_pred are tensors of shape (B, vocab_size).
    """
    tp = (y_true * y_pred).sum(dim=1)
    fp = ((1 - y_true) * y_pred).sum(dim=1)
    fn = (y_true * (1 - y_pred)).sum(dim=1)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return 1 - f1.mean()  # We want to maximize F1, so minimize (1-F1)

def combined_loss(y_true, y_pred, alpha=0.5):
    """
    Combines standard BCE loss with the soft F1 loss.
    alpha determines the trade-off between BCE and soft F1.
    """
    bce = F.binary_cross_entropy(y_pred, y_true)
    f1  = soft_f1_loss(y_true, y_pred)
    return alpha * bce + (1 - alpha) * f1

# --- Train and Validate Function ---
def train_and_validate(model, train_loader, val_loader, optimizer, num_epochs, device, 
                       threshold=0.5, max_iters=10, use_focal_loss=False, focal_alpha=1.0, focal_gamma=2.0,combined_alpha=0.5, 
                       use_combined_loss=False, criterion=None):
    """
    Train the model and evaluate using:
      1. Extended metrics: per-genome Accuracy, Precision, Recall, F1, and genome size difference.
      2. Iterative evaluation: refined predictions using iterative inference.
      
    Loss equation options:
      - Standard BCE Loss: L = -[ y*log(p) + (1-y)*log(1-p) ]
      - Focal Loss (if use_focal_loss=True):
          FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t)
      - Combined Loss (if use_combined_loss=True):
          combined_loss = alpha * BCE + (1 - alpha) * soft F1 loss.
    
    A ReduceLROnPlateau scheduler reduces the learning rate when the training loss plateaus.
    """
    model.to(device)
    
    # Set criterion if not provided.
    if criterion is None:
        if use_combined_loss:
            # Use combined loss (BCE + soft F1 loss)
            criterion = lambda preds, targets: combined_loss(targets, preds, alpha=combined_alpha)
            print_to_file("Using combined loss (BCE + soft F1 loss)")
        elif use_focal_loss:
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print_to_file("Using Focal Loss with alpha={} and gamma={}".format(focal_alpha, focal_gamma))
        else:
            criterion = nn.BCELoss()
            print_to_file("Using standard BCE Loss")
    
    # Set up learning rate scheduler.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc="Epoch {} Training".format(epoch+1))
        for tokens, mask, targets in train_bar:
            tokens  = tokens.to(device)
            mask    = mask.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(tokens, mask)  # Expected shape: (B, vocab_size)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print_to_file("Epoch [{}/{}] Training Loss: {:.4f}, Learning Rate: {:.6f}".format(
            epoch+1, num_epochs, avg_loss, current_lr))
        
        # Step the scheduler based on average training loss.
        scheduler.step(avg_loss)
        
        # Evaluate using extended metrics.
        extended_metrics = evaluate_metrics_extended(model, val_loader, device, threshold)
        print_to_file("\nExtended Evaluation Metrics (Epoch {}):".format(epoch+1))
        print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
        print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
        print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
        print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
        print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
        print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
        print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))    
    torch.save(model.state_dict(), "model_checkpoint_full.pth")
    print_to_file("Model checkpoint saved to model_checkpoint_full.pth")


import functools

# Define a helper function that accepts a file parameter
def _print_to_file(file, *args, sep=' ', end='\n', flush=True):
    output = sep.join(str(arg) for arg in args) + end
    file.write(output)
    if flush:
        file.flush()

# Open your file (make sure to manage its lifecycle appropriately)
output_file = open("SetTransformer.out", "w")

# Create a version of print that always writes to output_file
print_to_file = functools.partial(_print_to_file, output_file)


eggnog_csv = "filtered_all_eggnog.csv"
ar_metadata_tsv = "ar53_metadata_r220.tsv"
bac_metadata_tsv = "bac120_metadata_r220.tsv"


data, global_vocab, cog2idx = process_eggnog_and_metadata(eggnog_csv, ar_metadata_tsv, bac_metadata_tsv)




train_df, val_df = subsample_and_split_by_taxonomy(data, subsample_fraction=1.,
                                                       taxonomic_level="group", test_fraction=0.2,
                                                       random_state=42)

def initialize_weights(module):
    # Initialize Linear layers using Kaiming initialization for ReLU
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    # Optionally, initialize Embedding layers using Xavier initialization.
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)


dataset_train_low = GenomeDataset(train_df, global_vocab, cog2idx,
                              false_negative_rate=0.1, false_positive_rate=0.01,
                              count_noise_std=0.0, random_state=42)

dataset_train_med = GenomeDataset(train_df, global_vocab, cog2idx,
                              false_negative_rate=0.25, false_positive_rate=0.02,
                              count_noise_std=0.0, random_state=42)

dataset_train_high = GenomeDataset(train_df, global_vocab, cog2idx,
                              false_negative_rate=0.5, false_positive_rate=0.05,
                              count_noise_std=0.0, random_state=42)


dataset_val = GenomeDataset(val_df, global_vocab, cog2idx,
                            false_negative_rate=0.5, false_positive_rate=0.05,
                            count_noise_std=0.0, random_state=42)

dataset_val_low = GenomeDataset(val_df, global_vocab, cog2idx,
                            false_negative_rate=0.05, false_positive_rate=0.01,
                            count_noise_std=0.0, random_state=42)

dataset_val_med = GenomeDataset(val_df, global_vocab, cog2idx,
                            false_negative_rate=0.33, false_positive_rate=0.03,
                            count_noise_std=0.0, random_state=42)

dataset_val_high = GenomeDataset(val_df, global_vocab, cog2idx,
                            false_negative_rate=0.5, false_positive_rate=0.05,
                            count_noise_std=0.0, random_state=42)


pad_idx = len(global_vocab)
batch_size = 8
train_loader_low = DataLoader(dataset_train_low, batch_size=batch_size, shuffle=True,
                          collate_fn=lambda batch: collate_genomes(batch, pad_idx=pad_idx))
train_loader_med = DataLoader(dataset_train_med, batch_size=batch_size, shuffle=True,
                          collate_fn=lambda batch: collate_genomes(batch, pad_idx=pad_idx))
train_loader_high = DataLoader(dataset_train_high, batch_size=batch_size, shuffle=True,
                          collate_fn=lambda batch: collate_genomes(batch, pad_idx=pad_idx))

val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: collate_genomes(batch, pad_idx=pad_idx))

val_low_loader = DataLoader(dataset_val_low, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: collate_genomes(batch, pad_idx=pad_idx))

val_med_loader = DataLoader(dataset_val_med, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: collate_genomes(batch, pad_idx=pad_idx))
val_high_loader = DataLoader(dataset_val_high, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: collate_genomes(batch, pad_idx=pad_idx))


import gc
gc.collect()
torch.cuda.empty_cache()

# Apply this function to your model after construction:
model = GenomeSetTransformer(vocab_size=len(global_vocab), d_model=256, num_heads=4, num_sab=2, dropout=0.1)
model.apply(initialize_weights)

#model.load_state_dict(torch.load('model_checkpoint_full.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#model = GenomeSetTransformerWithCorrection(vocab_size=len(global_vocab), d_model=256, num_heads=8, num_sab=8, dropout=0.1, correction_hidden=128)

extended_metrics = evaluate_metrics_extended(model, val_low_loader, device, 0.5)
print_to_file("\n Low noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

extended_metrics = evaluate_metrics_extended(model, val_med_loader, device, 0.5)
print_to_file("\n Med noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

extended_metrics = evaluate_metrics_extended(model, val_high_loader, device, 0.5)
print_to_file("\n High noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

print_to_file("Training on device:", device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10
import gc
gc.collect()
torch.cuda.empty_cache()
train_and_validate(model, train_loader_low, val_loader, optimizer, num_epochs, device, threshold=0.5)
torch.save(model.state_dict(), "low_256_4_2_BCE.pth")

extended_metrics = evaluate_metrics_extended(model, val_low_loader, device, 0.5)
print_to_file("\n Low noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

extended_metrics = evaluate_metrics_extended(model, val_med_loader, device, 0.5)
print_to_file("\n Med noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

extended_metrics = evaluate_metrics_extended(model, val_high_loader, device, 0.5)
print_to_file("\n High noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))


train_and_validate(model, train_loader_med, val_loader, optimizer, num_epochs, device, threshold=0.5)
torch.save(model.state_dict(), "med_256_4_2_BCE.pth")

extended_metrics = evaluate_metrics_extended(model, val_low_loader, device, 0.5)
print_to_file("\n Low noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

extended_metrics = evaluate_metrics_extended(model, val_med_loader, device, 0.5)
print_to_file("\n Med noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

extended_metrics = evaluate_metrics_extended(model, val_high_loader, device, 0.5)
print_to_file("\n High noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))


train_and_validate(model, train_loader_high, val_loader, optimizer, num_epochs, device, threshold=0.5)
torch.save(model.state_dict(), "high_256_4_8_BCE_30.pth")

extended_metrics = evaluate_metrics_extended(model, val_low_loader, device, 0.5)
print_to_file("\n Low noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

extended_metrics = evaluate_metrics_extended(model, val_med_loader, device, 0.5)
print_to_file("\n Med noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

extended_metrics = evaluate_metrics_extended(model, val_high_loader, device, 0.5)
print_to_file("\n High noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

train_and_validate(model, train_loader_high, val_loader, optimizer, num_epochs, device, threshold=0.5)
torch.save(model.state_dict(), "high2_256_4_8_BCE_40.pth")

extended_metrics = evaluate_metrics_extended(model, val_low_loader, device, 0.5)
print_to_file("\n Low noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

extended_metrics = evaluate_metrics_extended(model, val_med_loader, device, 0.5)
print_to_file("\n Med noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

extended_metrics = evaluate_metrics_extended(model, val_high_loader, device, 0.5)
print_to_file("\n High noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

train_and_validate(model, train_loader_high, val_loader, optimizer, num_epochs, device, threshold=0.5)
torch.save(model.state_dict(), "high3_256_4_8_BCE_40.pth")

extended_metrics = evaluate_metrics_extended(model, val_low_loader, device, 0.5)
print_to_file("\n Low noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

extended_metrics = evaluate_metrics_extended(model, val_med_loader, device, 0.5)
print_to_file("\n Med noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))

extended_metrics = evaluate_metrics_extended(model, val_high_loader, device, 0.5)
print_to_file("\n High noise:")
print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))




