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
import gc
import os
import argparse
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import  DataLoader

from set_transformer.utils.architecture import GenomeSetTransformer
from set_transformer.utils.metrics import evaluate_metrics_extended
from set_transformer.utils.training_functions import initialize_weights, train_and_validate
from data_processing_utils.data_processing_functions import GenomeDataset, collate_genomes, process_eggnog_and_metadata, print_to_file, print_to_file_block, subsample_and_split_by_taxonomy, load_list_from_txt

EGGNOG_CSV = "data/filtered_all_eggnog.csv"
AR_METADATA_TSV = "data/ar53_metadata_r220.tsv"
BAC_MATADATA_TSV = "data/bac120_metadata_r220.tsv"


def generate_noisy_dataset(df, global_vocab, batch_size, pad_idx, fn_rate, fp_rate, count_noise_std=0, random_state=42):
     dataset = GenomeDataset(df, global_vocab,
                               false_negative_rate=fn_rate, false_positive_rate=fp_rate,
                               count_noise_std=count_noise_std, random_state=random_state)
     dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_genomes(batch, pad_idx=pad_idx))    
     return dataset_loader

def process_args():
    parser = argparse.ArgumentParser(description="Process input arguments for model training.")
    parser.add_argument("--train_feather_path", type=str, help="Path to the train feather data file.")
    parser.add_argument("--test_feather_path", type=str, help="Path to the test feather data file.")
    parser.add_argument("--global_vocab_path", type=str, help="Path to the global vocabulary data file.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for training")
    parser.add_argument("--embedd_dim", type=int, default=256, help="Embedding dimencionality")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--num_sab", type=int, default=2, help="Number of SAB layers")  
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")   
    args = parser.parse_args()
    args_dict = {"train_feather_path": args.train_feather_path, "test_feather_path": args.test_feather_path, "global_vocab_path": args.global_vocab_path, 
                 "batch_size": args.batch_size, "embedd_dim": args.embedd_dim, "num_heads": args.num_heads, "num_sab": args.num_sab, "num_epochs": args.num_epochs}
    return args_dict

def main(args_dict):

    # 2. Define the hyperparamerers of the model
    batch_size = args_dict["batch_size"]
    embedd_dim = args_dict["embedd_dim"]
    num_heads = args_dict["num_heads"]
    num_sab = args_dict["num_sab"]
    filename_specs = f"set_transf_embedd_{embedd_dim}_heads_{num_heads}_sab_{num_sab}_BCE"
    model_filename = filename_specs + ".pth"
    output_filename = filename_specs + ".out"
    output_directory = "set_transformer/output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = open(os.path.join(output_directory, output_filename), "w") 

    # 2. Load the train, test datasets, and the global dictionary (all COG names from the initial dataset)
    train_df = pd.read_feather(args_dict["train_feather_path"])
    val_df = pd.read_feather(args_dict["test_feather_path"])
    global_vocab = load_list_from_txt(args_dict["global_vocab_path"])
    pad_idx = len(global_vocab) # padding size

    # 4. Generate train data loaders in different noise regimes
    train_loader_low = generate_noisy_dataset(train_df, global_vocab, batch_size, pad_idx, fn_rate=0.1, fp_rate=0.01)
    train_loader_med = generate_noisy_dataset(train_df, global_vocab, batch_size, pad_idx, fn_rate=0.25, fp_rate=0.02)
    train_loader_high = generate_noisy_dataset(train_df, global_vocab, batch_size, pad_idx, fn_rate=0.5, fp_rate=0.05)

    # 5. Generate validation data loaders in different noise regimes
    val_loader = generate_noisy_dataset(val_df, global_vocab, batch_size, pad_idx, fn_rate=0.5, fp_rate=0.05)  # WHY???
    val_low_loader = generate_noisy_dataset(val_df, global_vocab, batch_size, pad_idx, fn_rate=0.05, fp_rate=0.01)  
    val_med_loader = generate_noisy_dataset(val_df, global_vocab, batch_size, pad_idx, fn_rate=0.33, fp_rate=0.03)  
    val_high_loader = generate_noisy_dataset(val_df, global_vocab, batch_size, pad_idx, fn_rate=0.5, fp_rate=0.05)  

    # 6. Training preparations / model initialization
    gc.collect() # free Python memory
    torch.cuda.empty_cache() # free unused GPU memory

    model = GenomeSetTransformer(vocab_size=len(global_vocab), d_model=embedd_dim, num_heads=num_heads, num_sab=num_sab, dropout=0.1)
    model.apply(initialize_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 7. Find and save the pre-training metrics
    for noise_name, val_loader_noise in zip(["Low", "Med", "High"], [val_low_loader, val_med_loader, val_high_loader]):
        extended_metrics = evaluate_metrics_extended(model, val_loader_noise, device, 0.5)
        print_to_file(output_file, f"\n {noise_name} noise:")
        print_to_file_block(output_file, extended_metrics)

    gc.collect()
    torch.cuda.empty_cache()
    print_to_file(output_file, "Training on device:", device)

    num_epochs = args_dict["num_epochs"]
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 8. Training on Low-Noise 
    train_and_validate(model, train_loader_low, val_loader, optimizer, num_epochs, device, output_file, threshold=0.5)
    torch.save(model.state_dict(), "low_" + model_filename)

    for noise_name, val_loader_noise in zip(["Low", "Med", "High"], [val_low_loader, val_med_loader, val_high_loader]):
        extended_metrics = evaluate_metrics_extended(model, val_loader_noise, device, 0.5)
        print_to_file(output_file, f"\n {noise_name} noise:")
        print_to_file_block(output_file, extended_metrics)

    # 9. Training on Med-Noise 
    train_and_validate(model, train_loader_med, val_loader, optimizer, num_epochs, device, output_file, threshold=0.5)
    torch.save(model.state_dict(), "med_" + model_filename)

    for noise_name, val_loader_noise in zip(["Low", "Med", "High"], [val_low_loader, val_med_loader, val_high_loader]):
        extended_metrics = evaluate_metrics_extended(model, val_loader_noise, device, 0.5)
        print_to_file(output_file, f"\n {noise_name} noise:")
        print_to_file_block(output_file, extended_metrics)

    # 10. Training on High-Noise 
    train_and_validate(model, train_loader_high, val_loader, optimizer, num_epochs, device, output_file, threshold=0.5)
    torch.save(model.state_dict(), "high_" + model_filename)

    for noise_name, val_loader_noise in zip(["Low", "Med", "High"], [val_low_loader, val_med_loader, val_high_loader]):
        extended_metrics = evaluate_metrics_extended(model, val_loader_noise, device, 0.5)
        print_to_file(output_file, f"\n {noise_name} noise:")
        print_to_file_block(output_file, extended_metrics)


# train_and_validate(model, train_loader_high, val_loader, optimizer, num_epochs, device, threshold=0.5)
# torch.save(model.state_dict(), "high2_256_4_8_BCE_40.pth")

# for noise_name, val_loader in zip(["Low", "Med", "High"], [val_low_loader, val_med_loader, val_high_loader]):
#     extended_metrics = evaluate_metrics_extended(model, val_loader, device, 0.5)
#     print_to_file(output_file, f"\n {noise_name} noise:")
#     print_to_file_block(output_file, extended_metrics)


# train_and_validate(model, train_loader_high, val_loader, optimizer, num_epochs, device, threshold=0.5)
# torch.save(model.state_dict(), "high3_256_4_8_BCE_40.pth")

# for noise_name, val_loader in zip(["Low", "Med", "High"], [val_low_loader, val_med_loader, val_high_loader]):
#     extended_metrics = evaluate_metrics_extended(model, val_loader, device, 0.5)
#     print_to_file(output_file, f"\n {noise_name} noise:")
#     print_to_file_block(output_file, extended_metrics)


if __name__=='__main__':
    # 1. Read the input args
    args_dict = process_args()
    main(args_dict)