import os
import sys
import random
import logging
import argparse
import polars as pl

"""
This script splits the input data into train and test with respect to the desired taxonomy group (e.g. phylum, class, family).

Inputs:
    - gtdb metadata files (stored in ~/gene-context/phenotype_prediction/data_preparation/gtdb_files/),
    - input_annotation_csv: input csv with genome names and annotations,
    - input_data_csv: input csv with COG counts,
    - tax_level: desired taxonomy level for splitting;
    - output_dir: desired output directory

Outputs:
    - txt file with taxonomy groups for training,
    - txt file with taxonomy groups for testing;
    - test/train csv's with annotations,
    - test/train csv's with COG counts.

How to run this script?
    cd ~/gene-context/phenotype_prediction
    python3 -m data_preparation.taxa_level_split  --tax_level [tax_level] --input_annotation_csv [input_annotation_csv] --input_data_csv [input_data_csv] --output_dir [output_dir]

E.g.
    python3 -m data_preparation.taxa_level_split --tax_level phylum --input_annotation_csv data_diderm/gold_standard1.tsv --input_data_csv data_diderm/all_gene_annotations.tsv --output_dir data_diderm/input_data
"""

BAC_TSV = 'data_preparation/gtdb_files/bac120_metadata_r202.tsv'
ARC_TSV = 'data_preparation/gtdb_files/ar122_metadata_r202.tsv'
TEST_DATA_SIZE = 0.2 # percentage of the initial dataset
RANDOM_SEED = 42

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

def process_args():
    parser = argparse.ArgumentParser(description="Process input arguments for model training.")
    parser.add_argument("--tax_level", type=str, help="Taxonomic lavel at which the data is split into train and test.")
    parser.add_argument('-annot', '--input_annotation_csv', required=True, type=str, help="Input cvs with genome annotations.")
    parser.add_argument('-data', '--input_data_csv', required=True, type=str, help="Input cvs with COG counts.")
    parser.add_argument('--output_dir', required=True, type=str, help="Output directory for the train/test files.")
    args = parser.parse_args()
    return args

def save_selected_data_and_annot(df, groups, tax_level, filename_data, filename_annot):
    # Prep the df
    df_filter = df.filter(pl.col(tax_level).is_in(groups))
    print("Training set has {} rows".format(len(df_filter)))

    # Select accession and accession and save as a csv
    train_df_annot = df_filter.select(["accession", "annotation"])
    train_df_annot.write_csv(filename_annot, separator="\t")

    # Select the counts and save as a csv
    df_filter = df_filter.drop([tax_level])
    df_filter = df_filter.drop("annotation")
    df_filter.write_csv(filename_data, separator="\t")

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

        gtdb_df = gtdb_df[['accession', tax_level]]
        
        # Read input csv with annotations
        input_df_annot = pl.read_csv(args.input_annotation_csv, separator="\t")
        old_name = input_df_annot.columns[1]
        input_df_annot = input_df_annot.rename({old_name: "annotation"})

        # Read input count table
        logging.info("Reading input count table ...")
        input_df_counts = pl.read_csv(args.input_data_csv, separator="\t")
        print(f"Reading input count table with {len(input_df_counts)} rows")

        # Concatenate it with the gtdb df
        joined_df = gtdb_df.join(input_df_annot, on="accession", how="left")
        joined_df = input_df_counts.join(joined_df, on="accession", how="left")

        # Find all taxonomy groups at the provided input level
        testing_families = set()
        testing_set_size = 0
        all_groups = list(set(joined_df[tax_level].to_list()))
        print(f"Found {len(all_groups)} groups at {tax_level} taxonomy level from {len(joined_df)} data points")

        # Randomly order the groups
        random.seed(RANDOM_SEED)
        random.shuffle(all_groups)

        # Split them into test and train
        while testing_set_size < len(joined_df) * TEST_DATA_SIZE:
            group = all_groups.pop()
            testing_families.add(group)
            testing_set_size += len(joined_df.filter(pl.col(tax_level) == group))
        print(f"Found {len(all_groups)} training groups at {tax_level} taxonomy level, comprising {len(joined_df) - testing_set_size} data points")    
        print(f"Found {len(testing_families)} testing groups at {tax_level} taxonomy level, comprising {testing_set_size} data points")

        # Create output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Save the train/test group lists to txt files
        train_filename = f"{args.output_dir}/train_groups_{tax_level}_tax_level"
        test_filename = f"{args.output_dir}/test_groups_{tax_level}_tax_level"

        with open(train_filename, "w") as f:
            f.write("\n".join(all_groups))
        with open(test_filename, "w") as f:
            f.write("\n".join(testing_families))

        # Save the train/test data files to txt files
        train_data_filename = f"{args.output_dir}/train_data_{tax_level}_tax_level"
        train_annot_filename = f"{args.output_dir}/train_annot_{tax_level}_tax_level"
        save_selected_data_and_annot(joined_df, all_groups, tax_level, train_data_filename, train_annot_filename)

        train_data_filename = f"{args.output_dir}/test_data_{tax_level}_tax_level"
        train_annot_filename = f"{args.output_dir}/test_annot_{tax_level}_tax_level"
        save_selected_data_and_annot(joined_df, testing_families, tax_level, train_data_filename, train_annot_filename)

        print("Finished!")
