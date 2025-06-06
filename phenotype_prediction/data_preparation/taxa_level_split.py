import os
import sys
import random
import logging
import argparse
import numpy as np
import polars as pl

import itertools
from collections import defaultdict

"""
This script splits the input data into train and test with respect to the desired taxonomy group (e.g. phylum, class, family).

Inputs:
    - gtdb metadata files (stored in ~/gene-context/phenotype_prediction/data_preparation/gtdb_files/),
    - input_annotation_csv: input csv with genome names and annotations,
    - input_data_csv: input csv with COG counts,
    - tax_level: desired taxonomy level for splitting OR "random" for random train/test split that doesn't take into account any taxonomy;
    - output_dir: desired output directory

Outputs:
    - txt file with taxonomy groups for training,
    - txt file with taxonomy groups for testing;
    - test/train csv's with annotations,
    - test/train csv's with COG counts,
    - test/train csv's with the corresponding taxa group names.

How to run this script?
    cd ~/gene-context/phenotype_prediction
    python3 -m data_preparation.taxa_level_split  --tax_level [tax_level] --input_annotation_csv [input_annotation_csv] --input_data_csv [input_data_csv] --output_dir [output_dir]

E.g.
    python3 -m data_preparation.taxa_level_split --tax_level phylum --input_annotation_csv data_diderm/gold_standard1.tsv --input_data_csv data_diderm/all_gene_annotations.tsv --output_dir data_diderm/input_data
    python3 -m data_preparation.taxa_level_split --tax_level phylum --input_annotation_csv data_host/all_annotations.csv --input_data_csv data_host/all_genes.csv --output_dir data_host/input_data
    python3 -m data_preparation.taxa_level_split --tax_level phylum --input_annotation_csv data_ogt/ogt_annot.csv --input_data_csv data_ogt/kegg.csv --output_dir data_ogt/input_data
    python3 -m data_preparation.taxa_level_split  --tax_level phylum --input_annotation_csv data_aerob/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv --input_data_csv data_aerob/all_gene_annotations.tsv --output_dir data_aerob/input_data

"""

BAC_TSV = 'data_preparation/gtdb_files/bac120_metadata_r202.tsv'
ARC_TSV = 'data_preparation/gtdb_files/ar122_metadata_r202.tsv'
TEST_DATA_SIZE = 0.2 # percentage of the initial dataset
RANDOM_SEED = 42

NUM_SPLITS = 10

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

def save_selected_data_and_annot(df, groups, tax_level, filename_data, filename_annot, filename_taxa):
    # Prep the df
    df_filter = df.filter(pl.col(tax_level).is_in(groups))
    df_filter = df_filter.unique(subset=["accession"], keep="first")
    print("The set has {} rows".format(len(df_filter)))

    # Select accession and accession and save as a csv
    train_df_annot = df_filter.select(["accession", "annotation"])
    train_df_annot.write_csv(filename_annot, separator="\t")

    # Save taxa groups as a csv
    df_taxa = df_filter.select(["accession", tax_level])
    df_taxa.write_csv(filename_taxa, separator="\t")

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


    # Fix the random seed
    random.seed(RANDOM_SEED)

    # Create output directory if it doesn't exist
    if not os.path.exists(f"{args.output_dir}/{tax_level}"):
        os.makedirs(f"{args.output_dir}/{tax_level}")

    # Try and process the input dfs for the provided tax level
    if tax_level not in tax_levels.keys() and tax_level != "random":
        logging.error(f"Provided tax_level '{tax_level}' is not valid! Choose one of {list(tax_levels.keys())} or 'random'.")
        sys.exit(1)
    elif  tax_level in tax_levels.keys():
        # Read and concatenate the gtdb dfs
        gtdb_df = pl.concat([pl.read_csv(BAC_TSV, separator="\t"),
            pl.read_csv(ARC_TSV, separator="\t")])
        gtdb_df = gtdb_df.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(tax_levels[tax_level]).alias(tax_level))

        gtdb_df = gtdb_df[['accession', tax_level]]
        
        # Read input csv with annotations
        input_df_annot = pl.read_csv(args.input_annotation_csv, separator=",")
        old_name = input_df_annot.columns[-1]
        input_df_annot = input_df_annot.rename({old_name: "annotation"})

        # Read input count table
        input_df_counts = pl.read_csv(args.input_data_csv, separator=",")
        print(f"Reading input count table with {len(input_df_counts)} rows...")

        # Concatenate it with the gtdb df
        joined_df = gtdb_df.join(input_df_annot, on="accession", how="left")
        
        joined_df = input_df_counts.join(joined_df, on="accession", how="left")

        joined_df = joined_df.filter(pl.col("annotation").is_not_null())
        joined_df = joined_df.filter(pl.col(tax_level).is_not_null())

        # Find all taxonomy groups at the provided input level
        testing_families = set()
        testing_set_size = 0
        all_groups = list(set(joined_df[tax_level].to_list()))
        print(f"Found {len(all_groups)} groups at {tax_level} taxonomy level from {len(joined_df)} data points")


        group_to_size_dict = defaultdict(int)

        # Count samples per group
        for group in all_groups:
            group_to_size_dict[group] = len(joined_df.filter(pl.col(tax_level) == group))

      #  print(f"group_to_size_dict = {group_to_size_dict}")    

        total_samples = len(joined_df)
        max_test_samples = len(joined_df) * TEST_DATA_SIZE

        # Store unique splits using frozensets to avoid repetition
        seen = set()
        unique_splits = []

        all_groups_set = set(all_groups)





        unique_splits = set()
        #unique_splits = []
        final_splits = []

        for _ in range(NUM_SPLITS):
            shuffled_groups = all_groups[:]  # copy
            random.shuffle(shuffled_groups)

            print(f"shuffled_groups = {shuffled_groups}")

            test_groups = set()
            test_size = 0
            i = 0

            print(f"test_size = {test_size}")
            print(f"max_test_samples = {max_test_samples}")

            # Fill the test set until we hit the test size threshold
            while test_size < max_test_samples:# and i < len(shuffled_groups) - 1:
                group = shuffled_groups[i]
                test_groups.add(group)
                print(f"group_to_size_dict[group] in while = {group_to_size_dict[group]}")
                test_size += group_to_size_dict[group]
                i += 1

            train_groups = set(shuffled_groups[i:])

            print(f"train_groups = {train_groups}")
            print(f"test_groups = {test_groups}")

            # Make canonical form to avoid mirrored duplicates
            canonical_split = frozenset([frozenset(train_groups), frozenset(test_groups)])
            if canonical_split not in unique_splits:
                unique_splits.add(canonical_split)
                final_splits.append((list(train_groups), list(test_groups)))


        # while len(unique_splits) < NUM_SPLITS:
        #     # Random subset size (excluding empty and full)
        #     r = random.randint(1, len(all_groups) - 1)
        #     train_groups = random.sample(all_groups, r)
        #     train_groups_set = frozenset(train_groups)

        #     train_size = sum(group_to_size_dict[g] for g in train_groups)
        #     if train_size > max_train_samples:
        #         continue

        #     test_groups_set = frozenset(all_groups_set - train_groups_set)

        #     # To avoid duplicate splits (mirror cases)
        #     canonical_split = frozenset([train_groups_set, test_groups_set])
        #     if canonical_split in seen:
        #         continue

        #     seen.add(canonical_split)
        #     unique_splits.append((list(train_groups_set), list(test_groups_set)))

        print(f"Generated {len(final_splits)} random valid train/test group splits.")

      #  print(f"unique_splits = {unique_splits}")

        for i in range(len(final_splits)):
            train_groups = final_splits[i][0]
            test_groups = final_splits[i][1]


        # # Randomly order the groups
        # random.shuffle(all_groups)
        
        # # Split them into test and train
        # while testing_set_size < len(joined_df) * TEST_DATA_SIZE and len(all_groups) > 1:
        #     group = all_groups.pop()
        #     testing_families.add(group)
        #     testing_set_size += len(joined_df.filter(pl.col(tax_level) == group))  
        # print(f"Found {len(all_groups)} training groups at {tax_level} taxonomy level, comprising {len(joined_df) - testing_set_size} data points")    
        # print(f"Found {len(testing_families)} testing groups at {tax_level} taxonomy level, comprising {testing_set_size} data points")

        # print(f"testing_families = {testing_families}")


            # Save the train/test group lists to txt files
            train_filename = f"{args.output_dir}/{tax_level}/train_groups_{tax_level}_tax_level_split_{i}"
            test_filename = f"{args.output_dir}/{tax_level}/test_groups_{tax_level}_tax_level_split_{i}"

            with open(train_filename, "w") as f:
                f.write("\n".join(x for x in train_groups if x is not None))
            with open(test_filename, "w") as f:
                f.write("\n".join(x for x in test_groups if x is not None))    

            # Resulting filenames
            train_data_filename = f"{args.output_dir}/{tax_level}/train_data_{tax_level}_tax_level_split_{i}"
            train_annot_filename = f"{args.output_dir}/{tax_level}/train_annot_{tax_level}_tax_level_split_{i}"
            train_taxa_filename = f"{args.output_dir}/{tax_level}/train_taxa_names_{tax_level}_tax_level_split_{i}"

            test_data_filename = f"{args.output_dir}/{tax_level}/test_data_{tax_level}_tax_level_split_{i}"
            test_annot_filename = f"{args.output_dir}/{tax_level}/test_annot_{tax_level}_tax_level_split_{i}"
            test_taxa_filename = f"{args.output_dir}/{tax_level}/test_taxa_names_{tax_level}_tax_level_split_{i}"
      
            # Save the train/test data files to txt files
            save_selected_data_and_annot(joined_df, train_groups, tax_level, train_data_filename, train_annot_filename, train_taxa_filename)
            save_selected_data_and_annot(joined_df, test_groups, tax_level, test_data_filename, test_annot_filename, test_taxa_filename)

        print("Finished!")
    else: # if "random" splitting
        for i in range(NUM_SPLITS):
            # Read input csv with annotations
            input_df_annot = pl.read_csv(args.input_annotation_csv, separator=",")
            old_name = input_df_annot.columns[-1]
            input_df_annot = input_df_annot.rename({old_name: "annotation"})

            # Read input count table
            input_df_counts = pl.read_csv(args.input_data_csv, separator=",")
            print(f"Reading input count table with {len(input_df_counts)} rows...")        

            # Concatenate it with the gtdb df
            joined_df = input_df_counts.join(input_df_annot, on="accession", how="left")
            joined_df = joined_df.filter(pl.col("annotation").is_not_null())

            # Shuffle the rows
            shuffled_indices = np.random.permutation(joined_df.height)
            joined_df = joined_df[shuffled_indices.tolist()]

            # Split the table into train and test
            test_size = int(len(joined_df) * TEST_DATA_SIZE)
            table_train = joined_df[test_size:]                 
            table_test  = joined_df[:test_size]               

            print(f"Found {len(table_train)} data points for the train set")    
            print(f"Found {len(table_test)} data points for the test set")

            # Save the train data and annotations
            train_data_filename = f"{args.output_dir}/{tax_level}/train_data_{tax_level}_tax_level_split_{i}"
            train_annot_filename = f"{args.output_dir}/{tax_level}/train_annot_{tax_level}_tax_level_split_{i}"

            train_df_annot = table_train.select(["accession", "annotation"])
            train_df_annot.write_csv(train_annot_filename, separator="\t")
            table_train = table_train.drop("annotation")
            table_train.write_csv(train_data_filename, separator="\t")

            # Save the test data and annotations
            test_data_filename = f"{args.output_dir}/{tax_level}/test_data_{tax_level}_tax_level_split_{i}"
            test_annot_filename = f"{args.output_dir}/{tax_level}/test_annot_{tax_level}_tax_level_split_{i}"

            train_df_annot = table_test.select(["accession", "annotation"])
            train_df_annot.write_csv(test_annot_filename, separator="\t")
            table_test = table_test.drop("annotation")
            table_test.write_csv(test_data_filename, separator="\t")

        print("Finished!")