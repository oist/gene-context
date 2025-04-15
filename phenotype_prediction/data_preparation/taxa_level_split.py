import sys
import random
import logging
import argparse
import polars as pl

"""
This script generates lists of taxonomy group (e.g. phylum, class, family) that are used for splitting the input data into test and train datasets.

Inputs:
    - gtdb metadata files,
    - input csv with genome names and annotations,
    - desired taxonomy level for splitting.

Outputs:
    - txt file with taxonomy groups for training,
    - txt file with taxonomy groups for testing.

How to run this script?
    cd ~/gene-context/phenotype_prediction
    python3 -m data_preparation.taxa_level_split  --tax_level [tax_level] -input-csv [input-csv]

E.g.
    python3 -m data_preparation.taxa_level_split  --tax_level family -input-csv data_diderm/gold_standard1.tsv
"""

BAC_TSV = 'data_preparation/gtdb_files/bac120_metadata_r202.tsv'
ARC_TSV = 'data_preparation/gtdb_files/ar122_metadata_r202.tsv'
RANDOM_SEED = 42

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

def process_args():
    parser = argparse.ArgumentParser(description="Process input arguments for model training.")
    parser.add_argument("--tax_level", type=str, help="Taxonomic lavel at which the data is split into train and test.")
    parser.add_argument('-i', '--input-csv', required=True, type=str, help="Input cvs with genome annotations.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Parse the input params
    parser = argparse.ArgumentParser()
    args = process_args()
    tax_level = args.tax_level

    # Dictionary with possible taxonomy levels and their indices in the gtdb table
    tax_levels = {"domain": 0, "phylum": 1, "class": 2, "order": 3, "family": 4, "genus": 5, "species": 6}

    # Try and process the input dfs for the provided tax level
    if tax_level not in tax_levels.keys():
        logging.error(f"Provided tax_level '{tax_level}' is not valid! Choose one of {list(tax_levels.keys())}.")
        sys.exit(1)
    else:
        # Read and concatenate the gtdb dfs
        gtdb_df = pl.concat([pl.read_csv(BAC_TSV, separator="\t"),
            pl.read_csv(ARC_TSV, separator="\t")])
        gtdb_df = gtdb_df.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(tax_levels[tax_level]).alias(tax_level))

        # Read input csv with annotations
        input_df = pl.read_csv(args.input_csv, separator="\t")
        
        # Concatenate it with the gtdb df
        joined_df = input_df.join(gtdb_df, on="accession", how="left")

        # Find all taxonomy groups at the provided input level
        testing_families = set()
        testing_set_size = 0
        all_groups = list(set(joined_df[tax_level].to_list()))
        print(f"Found {len(all_groups)} groups at {tax_level} taxonomy level from {len(joined_df)} data points")

        # Randomly order the groups
        random.seed(RANDOM_SEED)
        random.shuffle(all_groups)

        # Split them into test and train
        while testing_set_size < len(joined_df) * 0.2:
            group = all_groups.pop()
            testing_families.add(group)
            testing_set_size += len(joined_df.filter(pl.col("family") == group))
        print(f"Found {len(testing_families)} testing groups at {tax_level} taxonomy level, comprising {testing_set_size} data points")

        # Save the train/test group lists to txt files
        train_filename = f"data_preparation/train_groups_{tax_level}_tax_level"
        test_filename = f"data_preparation/test_groups_{tax_level}_tax_level"
       
        with open(train_filename, "w") as f:
            f.write("\n".join(all_groups))
        with open(test_filename, "w") as f:
            f.write("\n".join(testing_families))

        print("Finished!")
