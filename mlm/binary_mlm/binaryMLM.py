import gc
import argparse

import torch
from torch.utils.data import  DataLoader

from binary_mlm.utils.architecture import BinaryMLMModel
from binary_mlm.utils.training_functions import validate, train_model_simple, train_model, combined_validate
from data_processing_utils.data_processing_functions import CopyNumberMLMDataset, print_to_file, process_eggnog_and_metadata, subsample_and_split_by_taxonomy, fast_save_dfs, fast_load_dfs, get_global_vocab_and_cog2idx_from_df

EGGNOG_CSV =  "only_COGs.csv"
AR_METADATA_TSV = "ar53_metadata_r220.tsv"
BAC_MATADATA_TSV = "bac120_metadata_r220.tsv"

def read_and_split_input(output_file):
    data, global_vocab, cog2idx = process_eggnog_and_metadata(EGGNOG_CSV, AR_METADATA_TSV, BAC_MATADATA_TSV, output_file)
    train_df, val_df = subsample_and_split_by_taxonomy(data, output_file, subsample_fraction=1.,
                                                        taxonomic_level="group", test_fraction=0.2,
                                                        random_state=42)
    return global_vocab, cog2idx, train_df, val_df

def process_args():
    parser = argparse.ArgumentParser(description="Process input arguments for model training.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--embedd_dim", type=int, default=512, help="Embedding dimencionality")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")  
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")   
    parser.add_argument("--mask_prob", type=float, default=0.15, help="Masking probability.")  
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")  
    args = parser.parse_args()
    return args

def main():
    # 1. Read the input args
    args = process_args()

    # 2. Define the hyperparamerers of the model and output filenames
    batch_size = args.batch_size
    embedd_dim = args.embedd_dim
    num_heads = args.num_heads
    num_layers = args.num_layers
    filename_specs = f"binary_mlm_embedd_{embedd_dim}_heads_{num_heads}_layers_{num_layers}_BCE"
    model_filename = filename_specs + ".pth"
    output_filename = filename_specs + ".out"
    output_file = open(output_filename, "w")

    # 3. Read and split the input dataset into the training and test ones
    global_vocab, cog2idx, train_df, val_df = read_and_split_input(output_file)

    # Save the DataFrames:
    fast_save_dfs(train_df, val_df, output_file, train_path='COG_train1.feather', val_path='COG_val1.feather')

    # Later, load them back:
    #train_df, val_df = fast_load_dfs('COG_train01.feather', 'COG_val01.feather')
    #global_vocab, cog2idx = get_global_vocab_and_cog2idx_from_df(train_df)

    # Extract the copy number sequences from the global vocabulary columns.
    # Each sequence is a list of copy numbers.
    train_sequences = train_df[global_vocab].values.tolist()
    val_sequences   = val_df[global_vocab].values.tolist()

    # Create datasets and dataloaders.
    mask_prob = args.mask_prob
    train_dataset = CopyNumberMLMDataset(train_sequences, mask_prob=mask_prob)
    val_dataset   = CopyNumberMLMDataset(val_sequences, mask_prob=mask_prob)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the MLM model.
    # Set max_seq_len to the number of gene families (i.e. len(global_vocab)).
    model = BinaryMLMModel(vocab_size=3,
                        embed_dim=embedd_dim, 
                        num_layers=num_layers, 
                        num_heads=num_heads, 
                        dropout=0.1,
                        max_seq_len=len(global_vocab))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)




    gc.collect()
    torch.cuda.empty_cache()
    # Train the model.
    epochs = args.num_epochs
    lr = args.learning_rate #1e-4
    train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
    torch.save(model.state_dict(), "binMLM_512_6_8_01_e20.pth")

    train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
    torch.save(model.state_dict(), "binMLM_512_6_8_01_e40.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_to_file(torch.cuda.memory_summary(device=device, abbreviated=True))



if __name__=='__main__':
    main()