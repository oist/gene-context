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
import argparse
import torch
import torch.optim as optim
from torch.utils.data import  DataLoader

from architecture import GenomeSetTransformer
from metrics import evaluate_metrics_extended
from training_functions import initialize_weights, train_and_validate
from data_processing_functions import GenomeDataset, collate_genomes, process_eggnog_and_metadata, print_to_file, print_to_file_block, subsample_and_split_by_taxonomy

EGGNOG_CSV = "filtered_all_eggnog.csv"
AR_METADATA_TSV = "ar53_metadata_r220.tsv"
BAC_MATADATA_TSV = "bac120_metadata_r220.tsv"

def read_and_split_input():
    data, global_vocab, cog2idx = process_eggnog_and_metadata(EGGNOG_CSV, AR_METADATA_TSV, BAC_MATADATA_TSV)
    train_df, val_df = subsample_and_split_by_taxonomy(data, subsample_fraction=1.,
                                                        taxonomic_level="group", test_fraction=0.2,
                                                        random_state=42)
    return global_vocab, cog2idx, train_df, val_df

def generate_noisy_dataset(df, global_vocab, cog2idx, batch_size, pad_idx, fn_rate, fp_rate, count_noise_std=0, random_state=42):
     dataset = GenomeDataset(df, global_vocab, cog2idx,
                               false_negative_rate=fn_rate, false_positive_rate=fp_rate,
                               count_noise_std=count_noise_std, random_state=random_state)
     train_loader_low = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_genomes(batch, pad_idx=pad_idx))    
     return train_loader_low

def process_args():
    parser = argparse.ArgumentParser(description="Process input arguments for model training.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for training")
    parser.add_argument("--embedd_dim", type=int, default=256, help="Embedding dimencionality")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--num_sab", type=int, default=2, help="Number of SAB layers")  
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")   
    args = parser.parse_args()
    return args

def main():
    args = process_args()
    global_vocab, cog2idx, train_df, val_df = read_and_split_input()

    # Hyperparamerers of the model
    pad_idx = len(global_vocab) # padding size
    batch_size = args.batch_size
    embedd_dim = args.embedd_dim
    num_heads = args.num_heads
    num_sab = args.num_sab
    model_filename_specs = f"set_transf_embedd_{embedd_dim}_heads_{num_heads}_sab_{num_sab}_BCE.pth"

    # Generate train data loaders in different noise regimes
    train_loader_low = generate_noisy_dataset(train_df, global_vocab, cog2idx, batch_size, pad_idx, fn_rate=0.1, fp_rate=0.01)
    train_loader_med = generate_noisy_dataset(train_df, global_vocab, cog2idx, batch_size, pad_idx, fn_rate=0.25, fp_rate=0.02)
    train_loader_high = generate_noisy_dataset(train_df, global_vocab, cog2idx, batch_size, pad_idx, fn_rate=0.5, fp_rate=0.05)

    # Generate validation data loaders in different noise regimes
    val_loader = generate_noisy_dataset(val_df, global_vocab, cog2idx, batch_size, pad_idx, fn_rate=0.5, fp_rate=0.05)  # WHY???
    val_low_loader = generate_noisy_dataset(val_df, global_vocab, cog2idx, batch_size, pad_idx, fn_rate=0.05, fp_rate=0.01)  
    val_med_loader = generate_noisy_dataset(val_df, global_vocab, cog2idx, batch_size, pad_idx, fn_rate=0.33, fp_rate=0.03)  
    val_high_loader = generate_noisy_dataset(val_df, global_vocab, cog2idx, batch_size, pad_idx, fn_rate=0.5, fp_rate=0.05)  

    #_______________________________ Training preparations _______________________________#

    gc.collect() # free Python memory
    torch.cuda.empty_cache() # free unused GPU memory

    # Initialize the model
    model = GenomeSetTransformer(vocab_size=len(global_vocab), d_model=embedd_dim, num_heads=num_heads, num_sab=num_sab, dropout=0.1)
    model.apply(initialize_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Pre-training metrics
    for noise_name, val_loader in zip(["Low", "Med", "High"], [val_low_loader, val_med_loader, val_high_loader]):
        extended_metrics = evaluate_metrics_extended(model, val_loader, device, 0.5)
        print_to_file(f"\n {noise_name} noise:")
        print_to_file_block(extended_metrics)

    gc.collect()
    torch.cuda.empty_cache()
    print_to_file("Training on device:", device)

    num_epochs = args.num_epochs
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    #_______________________________ Training on Low-Noise _______________________________#
    train_and_validate(model, train_loader_low, val_loader, optimizer, num_epochs, device, threshold=0.5)
    torch.save(model.state_dict(), "low_" + model_filename_specs)

    for noise_name, val_loader in zip(["Low", "Med", "High"], [val_low_loader, val_med_loader, val_high_loader]):
        extended_metrics = evaluate_metrics_extended(model, val_loader, device, 0.5)
        print_to_file(f"\n {noise_name} noise:")
        print_to_file_block(extended_metrics)

    #_______________________________ Training on Med-Noise _______________________________#
    train_and_validate(model, train_loader_med, val_loader, optimizer, num_epochs, device, threshold=0.5)
    torch.save(model.state_dict(), "med_" + model_filename_specs)

    for noise_name, val_loader in zip(["Low", "Med", "High"], [val_low_loader, val_med_loader, val_high_loader]):
        extended_metrics = evaluate_metrics_extended(model, val_loader, device, 0.5)
        print_to_file(f"\n {noise_name} noise:")
        print_to_file_block(extended_metrics)

    #_______________________________ Training on High-Noise _______________________________#
    train_and_validate(model, train_loader_high, val_loader, optimizer, num_epochs, device, threshold=0.5)
    torch.save(model.state_dict(), "high_" + model_filename_specs)

    for noise_name, val_loader in zip(["Low", "Med", "High"], [val_low_loader, val_med_loader, val_high_loader]):
        extended_metrics = evaluate_metrics_extended(model, val_loader, device, 0.5)
        print_to_file(f"\n {noise_name} noise:")
        print_to_file_block(extended_metrics)


# train_and_validate(model, train_loader_high, val_loader, optimizer, num_epochs, device, threshold=0.5)
# torch.save(model.state_dict(), "high2_256_4_8_BCE_40.pth")

# for noise_name, val_loader in zip(["Low", "Med", "High"], [val_low_loader, val_med_loader, val_high_loader]):
#     extended_metrics = evaluate_metrics_extended(model, val_loader, device, 0.5)
#     print_to_file(f"\n {noise_name} noise:")
#     print_to_file_block(extended_metrics)


# train_and_validate(model, train_loader_high, val_loader, optimizer, num_epochs, device, threshold=0.5)
# torch.save(model.state_dict(), "high3_256_4_8_BCE_40.pth")

# for noise_name, val_loader in zip(["Low", "Med", "High"], [val_low_loader, val_med_loader, val_high_loader]):
#     extended_metrics = evaluate_metrics_extended(model, val_loader, device, 0.5)
#     print_to_file(f"\n {noise_name} noise:")
#     print_to_file_block(extended_metrics)


if __name__=='__main__':
    main()