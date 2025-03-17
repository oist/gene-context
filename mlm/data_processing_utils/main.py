import os
import argparse
import numpy as np
import pandas as pd

from data_processing_utils.data_processing_functions import print_to_file

def save_df_as_feather(df, feather_path):
    df = df.reset_index(drop=True)
    df.to_feather(feather_path)

def save_list_to_txt(list, txt_path):
    with open(txt_path, "w") as f:
        for item in list:
            f.write(f"{item}\n") 

def process_eggnog_and_metadata(eggnog_csv, ar_metadata_tsv, bac_metadata_tsv, output_file):
    # 1. Load and filter eggNOG data for COG/arCOG entries.
    print_to_file(output_file, "Loading eggNOG CSV data...")
    df_eggnog = pd.read_csv(eggnog_csv)
    print_to_file(output_file, "  Total eggNOG records loaded: {}".format(len(df_eggnog)))

    # 2. Filter out COGs and arCOG only
    print_to_file(output_file, "Filtering eggNOG records for COG/arCOG entries...")
    cog_mask = df_eggnog['eggNOG_OGs'].str.startswith("COG") | df_eggnog['eggNOG_OGs'].str.startswith("arCOG")
    df_eggnog = df_eggnog[cog_mask]
    print_to_file(output_file, "  Records after filtering: {}".format(len(df_eggnog)))

    # 3. Pivot the long column to a table
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
    
    # 4. Load metadata for archaeal and bacterial genomes.
    print_to_file(output_file, "Loading archaeal metadata...")
    df_ar = pd.read_csv(ar_metadata_tsv, sep="\t", low_memory=False)
    print_to_file(output_file, "  Archaeal metadata records: {}".format(len(df_ar)))
    
    print_to_file(output_file, "Loading bacterial metadata...")
    df_bac = pd.read_csv(bac_metadata_tsv, sep="\t", low_memory=False)
    print_to_file(output_file, "  Bacterial metadata records: {}".format(len(df_bac)))
    
    # 5. Merge metadata for archaeal and bacterial genomes.
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
    
    # 6. Splitting into taxonomy levels
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

    # 7. Create a cog vocabulary
    global_vocab = [col for col in df_pivot.columns if col != "accession"]
    # pivot_cog_columns = [col for col in df_pivot.columns if col != "accession"]
    # cog_columns = pivot_cog_columns
    # global_vocab = sorted(cog_columns)
    save_list_to_txt(global_vocab, f'{OUTPUT_DIRECTORY}/global_vocab.txt')

   # cog2idx = {cog: i for i, cog in enumerate(global_vocab)}
    print_to_file(output_file, "Created global vocabulary ({} tokens) and cog2idx mapping.".format(len(global_vocab)))
    print_to_file(output_file, "Data processing complete.")
    return merged_df


def subsample_and_split_by_taxonomy(data, output_file, subsample_fraction=0.1, taxonomic_level="group", test_fraction=0.2, random_state=None):
    # 1. Data subsampling (if needed)
    print_to_file(output_file, "Starting subsample_and_split_by_taxonomy...")
    print_to_file(output_file, "  Total genomes before subsampling: {}".format(data.shape[0]))
    print_to_file(output_file, "  Subsampling {}% of the genomes...".format(subsample_fraction * 100))
    subsampled = data.sample(frac=subsample_fraction, random_state=random_state).reset_index(drop=True)
    print_to_file(output_file, "  Genomes after subsampling: {}".format(subsampled.shape[0]))

    # 2. Splitting into train and test wrt the taxonomy group
    if taxonomic_level not in subsampled.columns:
        raise ValueError("Taxonomic level '{}' not found in data columns.".format(taxonomic_level))
    unique_groups = subsampled[taxonomic_level].dropna().unique()
    print_to_file(output_file, "  Found {} unique groups at taxonomic level '{}'.".format(len(unique_groups), taxonomic_level))
    
    rng = np.random.RandomState(random_state)
    shuffled_groups = list(unique_groups)
    rng.shuffle(shuffled_groups)
    
    total_genomes = subsampled.shape[0]
    target_test_count = test_fraction * total_genomes # size of the test dataset
    print_to_file(output_file, "  Target test set size: ~{} genomes".format(int(target_test_count)))
    
    test_groups = []
    train_group = []
    current_test_count = 0
    for group in shuffled_groups:
        group_count = subsampled[subsampled[taxonomic_level] == group].shape[0]
        if current_test_count < target_test_count:
            test_groups.append(group)
            current_test_count += group_count
        else:
            train_group.append(group)
    
    test_set = subsampled[subsampled[taxonomic_level].isin(test_groups)].reset_index(drop=True)
    train_set = subsampled[subsampled[taxonomic_level].isin(train_group)].reset_index(drop=True)

    print_to_file(output_file, "  Test set: {} genomes from {} groups; Train set: {} genomes".format(test_set.shape[0], len(test_groups), train_set.shape[0]))
    
    train_groups = set(train_set[taxonomic_level].unique())
    test_groups_set = set(test_set[taxonomic_level].unique())
    if train_groups.intersection(test_groups_set):
        print("  WARNING: Overlap detected in taxonomic groups between train and test sets!")
        print_to_file(output_file, "  WARNING: Overlap detected in taxonomic groups between train and test sets!")
    else:
        print("  No overlap in taxonomic groups between train and test sets.")
        print_to_file(output_file, "  No overlap in taxonomic groups between train and test sets.")
    
    print_to_file(output_file, "Subsample and split complete.\n")

    # 3. Save the train/test datasets and the corresponding taxonomic levels used in the split
    save_list_to_txt(test_groups, f'{OUTPUT_DIRECTORY}/{taxonomic_level}_test_split.txt')
    save_list_to_txt(train_group, f'{OUTPUT_DIRECTORY}/{taxonomic_level}_train_split.txt')

    save_df_as_feather(train_set, f'{OUTPUT_DIRECTORY}/cog_train_{taxonomic_level}_tax_level.feather')
    save_df_as_feather(test_set, f'{OUTPUT_DIRECTORY}/cog_test_{taxonomic_level}_tax_level.feather')

    return train_set, test_set

def process_args():
    parser = argparse.ArgumentParser(description="Process input arguments for the data reading and splitting.")
    parser.add_argument("--eggnog_csv", type=str, default="data/filtered_all_eggnog.csv", help="Eggnog data file path.")
    parser.add_argument("--ar_metadata", type=str, default="data/ar53_metadata_r220.tsv", help="Archaea data file path.")
    parser.add_argument("--bac_metadata", type=str, default="data/bac120_metadata_r220.tsv", help="Bacteria data file path.")
    parser.add_argument("--taxonomy_level", type=str, default="group", help="Taxonomy level for splitting into test/train datasets.")
    args = parser.parse_args()
    return args

OUTPUT_DIRECTORY = 'data/train_test_splits'
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

if __name__=='__main__':
    print("Processing the input files...")

    # 1. Read the input args
    args = process_args()

    # 2. Create the output directories and the log file
    output_directory_logs = f"{OUTPUT_DIRECTORY}/logs"
    if not os.path.exists(output_directory_logs):
        os.makedirs(output_directory_logs)
    output_file = open(os.path.join(output_directory_logs, f"logs_{args.taxonomy_level}_level.log"), "w") 

    # 3. Read and merge the eggnog and metadata files
    merged_df = process_eggnog_and_metadata(args.eggnog_csv, args.ar_metadata, args.bac_metadata, output_file)

    # 4. Split to the train and test datasets and save the result
    subsample_and_split_by_taxonomy(merged_df, output_file, subsample_fraction=1, taxonomic_level=args.taxonomy_level, test_fraction=0.2, random_state=42)
    print("The train/test splitting is finished!")

