import torch
import numpy as np
import pandas as pd
import functools

from torch.utils.data import Dataset



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


def print_to_file_block(extended_metrics):
    print_to_file("  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
    print_to_file("  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
    print_to_file("  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
    print_to_file("  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
    print_to_file("  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
    print_to_file("  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
    print_to_file("  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))   


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