import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import random

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

def fast_save_dfs(train_df, val_df, train_path='train.feather', val_path='val.feather'):
    """
    Save train and validation DataFrames to disk in Feather format.
    
    Parameters:
        train_df (pd.DataFrame): The training DataFrame.
        val_df (pd.DataFrame): The validation DataFrame.
        train_path (str): File path to save the training DataFrame.
        val_path (str): File path to save the validation DataFrame.
    """
    # Reset index to ensure a clean save.
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Save DataFrames in Feather format.
    train_df.to_feather(train_path)
    val_df.to_feather(val_path)
    print_to_file(f"Saved train DataFrame to {train_path} and validation DataFrame to {val_path}.")

def fast_load_dfs(train_path='train.feather', val_path='val.feather'):
    """
    Load train and validation DataFrames from Feather files.
    
    Parameters:
        train_path (str): File path for the saved training DataFrame.
        val_path (str): File path for the saved validation DataFrame.
        
    Returns:
        tuple: (train_df, val_df) loaded as pandas DataFrames.
    """
    train_df = pd.read_feather(train_path)
    val_df = pd.read_feather(val_path)
    print_to_file(f"Loaded train DataFrame from {train_path} and validation DataFrame from {val_path}.")
    return train_df, val_df

def get_global_vocab_and_cog2idx_from_df(df):
    """
    Extracts columns from the DataFrame that correspond to COG or arCOG entries.
    
    Parameters:
      - df (pd.DataFrame): The DataFrame containing gene count data along with extra annotations.
      
    Returns:
      - global_vocab (list): Sorted list of column names that start with 'COG' or 'arCOG'.
      - cog2idx (dict): Mapping from each COG name in global_vocab to a unique index.
    """
    # Filter columns that start with 'COG' or 'arCOG'
    global_vocab = [col for col in df.columns if col.startswith('COG') or col.startswith('arCOG')]
    # Sort the vocabulary for consistency
    global_vocab = sorted(global_vocab)
    # Create mapping from each COG to a unique index
    cog2idx = {cog: idx for idx, cog in enumerate(global_vocab)}
    return global_vocab, cog2idx

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc

############################################
# 2. Dataset and Model Definition
############################################

class CopyNumberMLMDataset(Dataset):
    """
    Dataset for masked language modeling on copy number data.

    Each sample is a list of integer copy numbers for gene families.
    The copy numbers are converted to binary:
      - 0 if 0 copies (absence)
      - 1 if ≥1 copy (presence)
    A fraction of tokens are randomly masked (set to token id 2) and the 
    original binary token is used as the label at those positions 
    (others are set to -100 so they are ignored in the loss).
    
    NOTE: The dataset is stored on CPU to minimize overall GPU memory usage.
    """
    def __init__(self, sequences, mask_prob=0.15):
        """
        sequences: list of lists of integers (gene copy numbers).
        mask_prob: probability to mask each token.
        """
        self.sequences = sequences
        self.mask_prob = mask_prob
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Convert the copy number sequence to a CPU tensor.
        raw_seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        # Convert to binary: 0 if 0 copies; 1 if ≥1 copies.
        binary_seq = (raw_seq > 0).long()
        labels = binary_seq.clone()
        
        # Create a random mask for a fraction of tokens.
        mask = torch.rand(binary_seq.shape) < self.mask_prob
        
        # Mask the tokens by setting them to token id 2.
        input_seq = binary_seq.clone()
        input_seq[mask] = 2  # mask token id
        
        # For loss, ignore unmasked tokens by setting labels to -100.
        labels[~mask] = -100
        
        return input_seq, labels

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

############################################
# 3. Training and Validation Functions
############################################

def validate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc="validate.."):
            # Transfer only the current batch to GPU.
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, loss = model(input_ids, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100
            all_preds.extend(preds[mask].cpu().numpy().tolist())
            all_labels.extend(labels[mask].cpu().numpy().tolist())
            
            torch.cuda.empty_cache()
    
    if len(all_labels) == 0:
        return None
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, pos_label=1)
    rec = recall_score(all_labels, all_preds, pos_label=1)
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    avg_loss = total_loss / len(dataloader)
    metrics = {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }
    return metrics

def train_model_simple(model, train_loader, val_loader, device, epochs=5, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Set up mixed precision training.
    scaler = torch.cuda.amp.GradScaler()
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        metrics = validate(model, val_loader, device)
        if metrics:
            print_to_file(f"Validation Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        else:
            print_to_file("Warning: No masked tokens in validation set!")
        
        for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                _, loss = model(input_ids, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            del input_ids, labels, loss
            torch.cuda.empty_cache()
            
        avg_loss = total_loss / len(train_loader)
        print_to_file(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")
        
        metrics = validate(model, val_loader, device)
        if metrics:
            print_to_file(f"Validation Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        else:
            print_to_file("Warning: No masked tokens in validation set!")
        
        scheduler.step(avg_loss)
        
        torch.cuda.empty_cache()
        gc.collect()
        
    return model


############################################
# Combined Validation Function
############################################

def combined_validate(model, dataloader, device, corruption_settings=None):
    """
    Compute standard masked validation metrics and extended metrics on corrupted data.
    
    Standard metrics are computed on positions where labels != -100.
    
    For extended validation, unmasked tokens (labels == -100) are corrupted by:
      - Flipping 1 -> 0 with probability false_negative.
      - Flipping 0 -> 1 with probability false_positive.
    Then for each setting, we compute:
      - false_negative_recovered: fraction of corrupted tokens originally 1 predicted as 1.
      - false_positive_removed: fraction of corrupted tokens originally 0 predicted as 0.
      - masked_accuracy: accuracy on masked tokens (which remain uncorrupted).
    
    Returns a dictionary with "standard_metrics" and "extended_metrics".
    """
    if corruption_settings is None:
        corruption_settings = [(0.1, 0.01), (0.2, 0.01), (0.3, 0.01), (0.5, 0.01)]
    
    # Standard masked validation metrics.
    all_masked_preds = []
    all_masked_labels = []
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, loss = model(input_ids, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            masked_mask = (labels != -100)
            all_masked_preds.extend(preds[masked_mask].cpu().numpy().tolist())
            all_masked_labels.extend(labels[masked_mask].cpu().numpy().tolist())
    standard_loss = total_loss / len(dataloader)
    standard_metrics = {
        'loss': standard_loss,
        'accuracy': accuracy_score(all_masked_labels, all_masked_preds),
        'precision': precision_score(all_masked_labels, all_masked_preds, pos_label=1),
        'recall': recall_score(all_masked_labels, all_masked_preds, pos_label=1),
        'f1': f1_score(all_masked_labels, all_masked_preds, pos_label=1)
    }
    
    # Extended validation metrics.
    ext_results = {setting: {"fn_total": 0, "fn_recovered": 0,
                             "fp_total": 0, "fp_removed": 0,
                             "masked_total": 0, "masked_correct": 0}
                   for setting in corruption_settings}
    
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masked_mask = (labels != -100)
            unmasked_mask = (labels == -100)
            ground_truth = input_ids.clone()
            for false_negative, false_positive in corruption_settings:
                corrupted_input = input_ids.clone()
                rand_vals = torch.rand(corrupted_input.shape, device=corrupted_input.device)
                flip_fn = (ground_truth == 1) & unmasked_mask & (rand_vals < false_negative)
                flip_fp = (ground_truth == 0) & unmasked_mask & (rand_vals < false_positive)
                corrupted_input[flip_fn] = 0
                corrupted_input[flip_fp] = 1
                
                logits, _ = model(corrupted_input, labels)
                preds = torch.argmax(logits, dim=-1)
                
                fn_flipped = flip_fn
                fn_total = fn_flipped.sum().item()
                fn_recovered = (preds[fn_flipped] == 1).sum().item() if fn_total > 0 else 0
                
                fp_flipped = flip_fp
                fp_total = fp_flipped.sum().item()
                fp_removed = (preds[fp_flipped] == 0).sum().item() if fp_total > 0 else 0
                
                ext_results[(false_negative, false_positive)]["fn_total"] += fn_total
                ext_results[(false_negative, false_positive)]["fn_recovered"] += fn_recovered
                ext_results[(false_negative, false_positive)]["fp_total"] += fp_total
                ext_results[(false_negative, false_positive)]["fp_removed"] += fp_removed
                
                masked_preds = preds[masked_mask]
############################################
# Combined Validation Function
############################################

def combined_validate(model, dataloader, device, corruption_settings=None):
    """
    Compute standard masked validation metrics and extended metrics on corrupted data.
    
    Standard metrics are computed on positions where labels != -100.
    
    For extended validation, unmasked tokens (labels == -100) are corrupted by:
      - Flipping 1 -> 0 with probability false_negative.
      - Flipping 0 -> 1 with probability false_positive.
    Then for each setting, we compute:
      - false_negative_recovered: fraction of corrupted tokens originally 1 predicted as 1.
      - false_positive_removed: fraction of corrupted tokens originally 0 predicted as 0.
      - Masked token performance on the corrupted input: accuracy, precision, recall, and F1 
        computed on the masked tokens (which remain uncorrupted).
    
    Returns a dictionary with two keys:
         "standard_metrics": standard masked metrics.
         "extended_metrics": a dict keyed by corruption setting (tuple) with the following keys:
            - "false_negative_recovered"
            - "false_positive_removed"
            - "masked_accuracy"
            - "masked_precision"
            - "masked_recall"
            - "masked_f1"
    """
    if corruption_settings is None:
        corruption_settings = [(0.1, 0.01), (0.2, 0.01), (0.3, 0.01), (0.5, 0.01)]
    
    model.eval()
    
    # ---- Standard Masked Metrics ----
    all_masked_preds = []
    all_masked_labels = []
    total_loss = 0
    for input_ids, labels in tqdm(dataloader, desc="Standard Validation"):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, loss = model(input_ids, labels)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        masked_mask = (labels != -100)
        all_masked_preds.extend(preds[masked_mask].cpu().numpy().tolist())
        all_masked_labels.extend(labels[masked_mask].cpu().numpy().tolist())
    standard_loss = total_loss / len(dataloader)
    standard_metrics = {
        'loss': standard_loss,
        'accuracy': accuracy_score(all_masked_labels, all_masked_preds),
        'precision': precision_score(all_masked_labels, all_masked_preds, pos_label=1),
        'recall': recall_score(all_masked_labels, all_masked_preds, pos_label=1),
        'f1': f1_score(all_masked_labels, all_masked_preds, pos_label=1)
    }
    
    # ---- Extended Metrics on Corrupted Unmasked Tokens ----
    # For each corruption setting, accumulate:
    # - Counts for false negatives (flipped tokens originally 1) and false positives (flipped tokens originally 0).
    # - Also, accumulate masked token predictions and labels from the corrupted input.
    ext_results = {
        setting: {
            "fn_total": 0, "fn_recovered": 0,
            "fp_total": 0, "fp_removed": 0,
            "masked_total": 0, "masked_correct": 0,
            "masked_preds_list": [],
            "masked_labels_list": []
        }
        for setting in corruption_settings
    }
    
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc="Extended Validation"):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masked_mask = (labels != -100)
            unmasked_mask = (labels == -100)
            ground_truth = input_ids.clone()  # true values for unmasked tokens
            
            for false_negative, false_positive in corruption_settings:
                # Create a corrupted copy.
                corrupted_input = input_ids.clone()
                rand_vals = torch.rand(corrupted_input.shape, device=corrupted_input.device)
                flip_fn = (ground_truth == 1) & unmasked_mask & (rand_vals < false_negative)
                flip_fp = (ground_truth == 0) & unmasked_mask & (rand_vals < false_positive)
                corrupted_input[flip_fn] = 0
                corrupted_input[flip_fp] = 1
                
                logits, _ = model(corrupted_input, labels)
                preds = torch.argmax(logits, dim=-1)
                
                # For false negatives: tokens originally 1 that were flipped to 0.
                fn_flipped = flip_fn
                fn_total = fn_flipped.sum().item()
                fn_recovered = (preds[fn_flipped] == 1).sum().item() if fn_total > 0 else 0
                
                # For false positives: tokens originally 0 that were flipped to 1.
                fp_flipped = flip_fp
                fp_total = fp_flipped.sum().item()
                fp_removed = (preds[fp_flipped] == 0).sum().item() if fp_total > 0 else 0
                
                ext_results[(false_negative, false_positive)]["fn_total"] += fn_total
                ext_results[(false_negative, false_positive)]["fn_recovered"] += fn_recovered
                ext_results[(false_negative, false_positive)]["fp_total"] += fp_total
                ext_results[(false_negative, false_positive)]["fp_removed"] += fp_removed
                
                # For masked tokens (which remain uncorrupted), accumulate predictions and labels.
                masked_preds = preds[masked_mask]
                masked_labels = labels[masked_mask]
                ext_results[(false_negative, false_positive)]["masked_total"] += masked_mask.sum().item()
                ext_results[(false_negative, false_positive)]["masked_correct"] += (masked_preds == masked_labels).sum().item()
                # Also accumulate the lists for further metric computation.
                ext_results[(false_negative, false_positive)]["masked_preds_list"].append(masked_preds.cpu().numpy().tolist())
                ext_results[(false_negative, false_positive)]["masked_labels_list"].append(masked_labels.cpu().numpy().tolist())
    
    extended_metrics = {}
    for setting, counts in ext_results.items():
        fn_total = counts["fn_total"]
        fp_total = counts["fp_total"]
        recovered_frac = counts["fn_recovered"] / fn_total if fn_total > 0 else None
        removed_frac = counts["fp_removed"] / fp_total if fp_total > 0 else None
        masked_acc = counts["masked_correct"] / counts["masked_total"] if counts["masked_total"] > 0 else None
        
        # Flatten the lists of masked predictions and labels.
        flat_masked_preds = sum(counts["masked_preds_list"], [])
        flat_masked_labels = sum(counts["masked_labels_list"], [])
        # Compute additional metrics.
        masked_prec = precision_score(flat_masked_labels, flat_masked_preds, pos_label=1) if len(flat_masked_labels) > 0 else None
        masked_rec = recall_score(flat_masked_labels, flat_masked_preds, pos_label=1) if len(flat_masked_labels) > 0 else None
        masked_f1 = f1_score(flat_masked_labels, flat_masked_preds, pos_label=1) if len(flat_masked_labels) > 0 else None
        
        extended_metrics[setting] = {
            "false_negative_recovered": recovered_frac,
            "false_positive_removed": removed_frac,
            "masked_accuracy": masked_acc,
            "masked_precision": masked_prec,
            "masked_recall": masked_rec,
            "masked_f1": masked_f1
        }
    
    return {
        "standard_metrics": standard_metrics,
        "extended_metrics": extended_metrics
    }

############################################
# Training Loop with Combined Validation
############################################

def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    scaler = GradScaler()
    model.to(device)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        model.train()
        total_loss = 0
        for input_ids, labels in tqdm(train_loader, desc="Training"):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                _, loss = model(input_ids, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            del input_ids, labels, loss
            torch.cuda.empty_cache()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")
        
        combined_metrics = combined_validate(model, val_loader, device)
        std_met = combined_metrics["standard_metrics"]
        ext_met = combined_metrics["extended_metrics"]
        print("Post-training Validation Metrics (Standard masked):")
        print(f"  Loss: {std_met['loss']:.4f}, Accuracy: {std_met['accuracy']:.4f}, "
              f"Precision: {std_met['precision']:.4f}, Recall: {std_met['recall']:.4f}, F1: {std_met['f1']:.4f}")
        print("Extended Validation Metrics (Corrupted unmasked & masked performance):")
        for (fn_rate, fp_rate), metrics in ext_met.items():
            print(f"  Setting: false_negative={fn_rate}, false_positive={fp_rate}")
            print(f"    False negatives recovered: {metrics['false_negative_recovered']}")
            print(f"    False positives removed:  {metrics['false_positive_removed']}")
            print(f"    Masked token accuracy:      {metrics['masked_accuracy']}")
            print(f"    Masked token precision:     {metrics['masked_precision']}")
            print(f"    Masked token recall:        {metrics['masked_recall']}")
            print(f"    Masked token F1:            {metrics['masked_f1']}")
        
        scheduler.step(avg_loss)
        torch.cuda.empty_cache()
        gc.collect()
        
    return model

############################################
# 4. Main Routine
############################################

import functools

# Define a helper function that accepts a file parameter
def _print_to_file(file, *args, sep=' ', end='\n', flush=True):
    output = sep.join(str(arg) for arg in args) + end
    file.write(output)
    if flush:
        file.flush()

# Open your file (make sure to manage its lifecycle appropriately)
output_file = open("binaryMLM.out", "w")

# Create a version of print that always writes to output_file
print_to_file = functools.partial(_print_to_file, output_file)

# File paths for your data.
eggnog_csv = "only_COGs.csv"
ar_metadata_tsv = "ar53_metadata_r220.tsv"
bac_metadata_tsv = "bac120_metadata_r220.tsv"

# Process the data and build the gene count table with metadata.
data, global_vocab, cog2idx = process_eggnog_and_metadata(eggnog_csv, ar_metadata_tsv, bac_metadata_tsv)

# Subsample 10% of the genomes and split by taxonomy.
train_df, val_df = subsample_and_split_by_taxonomy(data, subsample_fraction=1.,
                                                   taxonomic_level="group", test_fraction=0.2,
                                                   random_state=42)


# Example usage:
# After subsampling:
# train_df, val_df = subsample_and_split_by_taxonomy(...)

# Save the DataFrames:
fast_save_dfs(train_df, val_df, train_path='COG_train1.feather', val_path='COG_val1.feather')

# Later, load them back:
#train_df, val_df = fast_load_dfs('COG_train01.feather', 'COG_val01.feather')
#global_vocab, cog2idx = get_global_vocab_and_cog2idx_from_df(train_df)

# Extract the copy number sequences from the global vocabulary columns.
# Each sequence is a list of copy numbers.
train_sequences = train_df[global_vocab].values.tolist()
val_sequences   = val_df[global_vocab].values.tolist()

# Create datasets and dataloaders.
mask_prob = 0.15
train_dataset = CopyNumberMLMDataset(train_sequences, mask_prob=mask_prob)
val_dataset   = CopyNumberMLMDataset(val_sequences, mask_prob=mask_prob)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)

# Initialize the MLM model.
# Set max_seq_len to the number of gene families (i.e. len(global_vocab)).
model = BinaryMLMModel(vocab_size=3,
                       embed_dim=512, 
                       num_layers=6, 
                       num_heads=8, 
                       dropout=0.1,
                       max_seq_len=len(global_vocab))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



import gc
gc.collect()
torch.cuda.empty_cache()
# Train the model.
epochs = 20
lr = 1e-4
train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
torch.save(model.state_dict(), "binMLM_512_6_8_01_e20.pth")

train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
torch.save(model.state_dict(), "binMLM_512_6_8_01_e40.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_to_file(torch.cuda.memory_summary(device=device, abbreviated=True))
