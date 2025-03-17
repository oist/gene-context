import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class GenomeDataset(Dataset):
    def __init__(self, df, global_vocab,
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
        #self.cog2idx = cog2idx
        self.vocab_size = len(global_vocab)
        self.false_negative_rate = false_negative_rate
        self.false_positive_rate = false_positive_rate
        self.count_noise_std = count_noise_std
        self.rng = np.random.RandomState(random_state)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get the gene counts and convert to binary target (presence/absence)
        row = self.df.iloc[idx]
      #  print(f"row = {row}")
        counts = row[self.global_vocab].values.astype(np.float32)

      #  print(f"counts = {counts}")
        target = (counts > 0).astype(np.float32)
        # print(f"fn rate = {self.false_negative_rate}")
        # print(f"fp rate = {self.false_positive_rate}")
        # print(f"len tar = {len(target)}")
        # print(f"num of 1s = {sum(target)}")
        # print(f"num of 0s = {len(target) - sum(target)}")

        observed_indices = []
        observed_counts = []
        # For each gene in the target, decide whether to keep it (simulate false negatives)
        for cog_idx, present in enumerate(target):
            if present:
                # If True -> skip and do not attach the cog to the observed list
                if self.rng.random() < self.false_negative_rate:
                    continue
                #original_count = counts[cog_idx]
                # Add noise to the count (even though later we threshold to binary)
                #noise = self.rng.normal(loc=1.0, scale=self.count_noise_std)
                #noisy_count = max(original_count * noise, 0.0)
                observed_indices.append(cog_idx)
                observed_counts.append(1)

            else: # if the cog was originally absent
                # If True -> add the absent cog to the observed list
                if self.rng.random() < self.false_positive_rate:    
                    observed_indices.append(cog_idx)
                    observed_counts.append(1)  # Use 1 or modify for noisy false positives        

        # # Simulate false positives: add a small number of genes that are truly absent.
        # num_false_positives = self.rng.poisson(lam=self.false_positive_rate * self.vocab_size)

        # absent_indices = np.where(target == 0)[0]

        # if len(absent_indices) > 0 and num_false_positives > 0:
        #     false_pos = self.rng.choice(absent_indices, size=min(num_false_positives, len(absent_indices)), replace=False)
        #     for fp in false_pos:
        #         noisy_count = 1;#abs(self.rng.normal(loc=1.0, scale=self.count_noise_std))
        #         observed_indices.append(fp)
        #         observed_counts.append(noisy_count)

        
        # Build token array: each token = [COG_index, noisy_count]
        if len(observed_indices) == 0:
            tokens = np.empty((0, 2), dtype=np.float32)
        else:
            tokens = np.stack([np.array(observed_indices, dtype=np.int64),
                               np.array(observed_counts, dtype=np.float32)], axis=-1)
            
     #   print(f"len observed_indices = {len(observed_indices)}")    
        
        sample = {
            'tokens': tokens,
            'target': target  # Ground truth binary vector (V,)
        }
        return sample


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

def print_to_file(output_file, *args, sep=' ', end='\n', flush=True):
    output = sep.join(str(arg) for arg in args) + end
    output_file.write(output)
    if flush:
        output_file.flush()


def load_list_from_txt(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f]  # Removes trailing \n

def print_to_file_block(output_file, extended_metrics):
    print_to_file(output_file, "  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
    print_to_file(output_file, "  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
    print_to_file(output_file, "  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
    print_to_file(output_file, "  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
    print_to_file(output_file, "  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
    print_to_file(output_file, "  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
    print_to_file(output_file, "  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))   


def process_eggnog_and_metadata(eggnog_csv, ar_metadata_tsv, bac_metadata_tsv, output_file):
    # Load and filter eggNOG data for COG/arCOG entries.
    print_to_file(output_file, "Loading eggNOG CSV data...")
    print("Loading eggNOG CSV data...")
    df_eggnog = pd.read_csv(eggnog_csv)
    print_to_file(output_file, "  Total eggNOG records loaded: {}".format(len(df_eggnog)))

    print(f"Total eggNOG records loaded: {len(df_eggnog)})")
    
    print_to_file(output_file, "Filtering eggNOG records for COG/arCOG entries...")
    cog_mask = df_eggnog['eggNOG_OGs'].str.startswith("COG") | df_eggnog['eggNOG_OGs'].str.startswith("arCOG")
    df_eggnog = df_eggnog[cog_mask]
    print_to_file(output_file, "  Records after filtering: {}".format(len(df_eggnog)))
    
    print_to_file(output_file, "Pivoting eggNOG data to build a gene count table per accession...")
    df_pivot = df_eggnog.pivot_table(
        index='acc', 
        columns='eggNOG_OGs', 
        values='count', 
        aggfunc='sum', 
        fill_value=0
    )
    df_pivot = df_pivot.reset_index().rename(columns={'acc': 'accession'})
    print_to_file(output_file, "  Pivoted gene count table shape: {}".format(df_pivot.shape))
    
    # Load metadata for archaeal and bacterial genomes.
    print_to_file(output_file, "Loading archaeal metadata...")
    df_ar = pd.read_csv(ar_metadata_tsv, sep="\t", low_memory=False)
    print_to_file(output_file, "  Archaeal metadata records: {}".format(len(df_ar)))
    
    print_to_file(output_file, "Loading bacterial metadata...")
    df_bac = pd.read_csv(bac_metadata_tsv, sep="\t", low_memory=False)
    print_to_file(output_file, "  Bacterial metadata records: {}".format(len(df_bac)))
    
    meta_cols_needed = ['accession', 'gtdb_taxonomy', 
                        'checkm_completeness', 'checkm_contamination', 
                        'coding_bases', 'genome_size', 'gc_percentage']
    print_to_file(output_file, "Selecting key metadata columns...")
    df_ar = df_ar[meta_cols_needed]
    df_bac = df_bac[meta_cols_needed]
    
    print_to_file(output_file, "Combining archaeal and bacterial metadata...")
    df_meta = pd.concat([df_ar, df_bac], ignore_index=True)
    
    print_to_file(output_file, "Merging gene count table with metadata...")
    merged_df = pd.merge(df_pivot, df_meta, how='left', on='accession')
    print_to_file(output_file, "  Merged data shape: {}".format(merged_df.shape))
    missing_meta = merged_df['gtdb_taxonomy'].isna().sum()
    if missing_meta > 0:
        print_to_file(output_file, "  WARNING: {} accessions are missing GTDB taxonomy metadata.".format(missing_meta))
    
    taxonomy_levels = ["domain", "phylum", "class", "order", "family", "group", "species"]
    print_to_file(output_file, "Splitting 'gtdb_taxonomy' into taxonomy columns:")
    print_to_file(output_file, "  Taxonomy levels: {}".format(taxonomy_levels))
    
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
    print_to_file(output_file, "  After adding taxonomy columns, data shape is: {}".format(merged_df.shape))
    
    pivot_cog_columns = [col for col in df_pivot.columns if col != "accession"]
    cog_columns = pivot_cog_columns
    
    global_vocab = sorted(cog_columns)
    cog2idx = {cog: i for i, cog in enumerate(global_vocab)}
    print_to_file(output_file, "Created global vocabulary ({} tokens) and cog2idx mapping.".format(len(global_vocab)))
    print_to_file(output_file, "Data processing complete.")
    return merged_df, global_vocab, cog2idx


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


def subsample_and_split_by_taxonomy(data, output_file, subsample_fraction=0.1, taxonomic_level="group", test_fraction=0.2, random_state=None):
    print_to_file(output_file, "Starting subsample_and_split_by_taxonomy...")
    print_to_file(output_file, "  Total genomes before subsampling: {}".format(data.shape[0]))
    
    print_to_file(output_file, "  Subsampling {}% of the genomes...".format(subsample_fraction * 100))
    subsampled = data.sample(frac=subsample_fraction, random_state=random_state).reset_index(drop=True)
    print_to_file(output_file, "  Genomes after subsampling: {}".format(subsampled.shape[0]))
    
    if taxonomic_level not in subsampled.columns:
        raise ValueError("Taxonomic level '{}' not found in data columns.".format(taxonomic_level))
    
    unique_groups = subsampled[taxonomic_level].dropna().unique()
    print_to_file(output_file, "  Found {} unique groups at taxonomic level '{}'.".format(len(unique_groups), taxonomic_level))
    
    rng = np.random.RandomState(random_state)
    shuffled_groups = list(unique_groups)
    rng.shuffle(shuffled_groups)
    
    total_genomes = subsampled.shape[0]
    target_test_count = test_fraction * total_genomes
    print_to_file(output_file, "  Target test set size: ~{} genomes".format(int(target_test_count)))
    
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
    
    print_to_file(output_file, "  Test set: {} genomes from {} groups; Train set: {} genomes".format(
        test_set.shape[0], len(test_groups), train_set.shape[0]))
    
    train_groups = set(train_set[taxonomic_level].unique())
    test_groups_set = set(test_set[taxonomic_level].unique())
    if train_groups.intersection(test_groups_set):
        print_to_file(output_file, "  WARNING: Overlap detected in taxonomic groups between train and test sets!")
    else:
        print_to_file(output_file, "  No overlap in taxonomic groups between train and test sets.")
    
    print_to_file(output_file, "Subsample and split complete.\n")
    return train_set, test_set


def fast_save_dfs(train_df, val_df, output_file, train_path='train.feather', val_path='val.feather'):
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
    print_to_file(output_file, f"Saved train DataFrame to {train_path} and validation DataFrame to {val_path}.")


def fast_load_dfs(output_file, train_path='train.feather', val_path='val.feather'):
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
    print_to_file(output_file, f"Loaded train DataFrame from {train_path} and validation DataFrame from {val_path}.")
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