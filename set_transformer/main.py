import os
import torch
import argparse
from collections import namedtuple

from utils.utils import read_ogt_data, process_aerob_dataset, str2bool
from set_transformer.train_test_func import cross_validation, train, test

"""
This is the main script for training the SetTransformer model for the chosen parameters and phenotype. At the moment, only two phenotypes are supported: 'aerob' and 'ogt'.
The script calls cross-validation, training and test functions, which are implemented in set_transformer/train_test_func.py. 

The input train and test data are the following:

1. For phenotype  = 'aerob':
    - data_aerob/all_gene_annotations.added_incompleteness_and_contamination.subsampled.training.tsv
    - data_aerob/all_gene_annotations.added_incompleteness_and_contamination.subsampled.testing.tsv
    - data_aerob/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv
    - data_aerob/bac120_metadata_r202.tsv
    - data_aerob/ar122_metadata_r202.tsv

2. For phenotype  = 'ogt':
    - data_ogt/kegg.csv
    - data_ogt/ogt_splits.csv

The results of the training and cross-validation are saved to results/SetTransformer/{phenotype}. The scripts generate and save:
    - prediction_probabilities_cross_valid_fold_{fold_id}_SetTransformer_indPoints_{num_ind_points}.csv
    - prediction_probabilities_holdout_test_SetTransformer_indPoints_{num_ind_points}.csv
    - trained_model_SetTransformer_indPoints_{num_ind_points}_D_{D_val}_K_{K_val}_dim_output_{dim_output_val}.model

How to use this script?    
    python3 -m set_transformer.main --num_inds {num_inds_val} --learning_rate {learning_rate_val} --num_epochs {num_epochs_val} --batch_size {batch_size_val} --phenotype [ogt/aerob]

E.g.
    python3 -m set_transformer.main --num_inds 20 --learning_rate 0.0001 --num_epochs 10 --batch_size 32 --phenotype ogt
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       

# Number of classes in the classification task for OGT (this number = 2 for aerob. phenotype)
NUM_CLASSES_OGT = 50 

def process_args():
    """
    Processes input args.
    """
    parser = argparse.ArgumentParser(description="Process command-line arguments and create a named tuple.")
    
    # Add arguments
    parser.add_argument("--num_inds", type=int, required=True, help="Number of inducing point in SetTransformer.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs for training.")
    parser.add_argument("--phenotype", type=str, required=True, help="Phenotype for the clasification task (aerob/ogt).")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size.")
    parser.add_argument("--data_filename_train", type=str, required=False, help="Filename of the train dataset.", default = "data_aerob/all_gene_annotations.added_incompleteness_and_contamination.training.tsv")
    parser.add_argument("--data_filename_test", type=str, required=False, help="Filename of the test dataset.", default="data_aerob/all_gene_annotations.added_incompleteness_and_contamination.testing.tsv")
    parser.add_argument("--y_filename", type=str, required=False, help="Filename of the label dataset.", default="data_aerob/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv")
    parser.add_argument("--ogt_continuous_flag", type=str2bool, required=False, help="Flag for using continuous predictions for the OGT data.", default="false")
    # Parse arguments
    args = parser.parse_args()
    Parameters = namedtuple("Parameters", ["num_inds", "learning_rate", "num_epochs", "phenotype", "batch_size", "data_filename_train", "data_filename_test", "y_filename", "ogt_continuous_flag"])
    return Parameters(num_inds=args.num_inds, learning_rate=args.learning_rate, num_epochs=args.num_epochs, phenotype=args.phenotype, batch_size=args.batch_size, data_filename_train = args.data_filename_train, data_filename_test=args.data_filename_test, y_filename=args.y_filename, ogt_continuous_flag=args.ogt_continuous_flag)
    
def create_directory_for_results(Parameters):
    phenotype = Parameters.phenotype
    # Directory to save 
    save_dir = os.path.join('results', 'SetTransformer', phenotype)

    if phenotype == 'ogt': #create sub-directories for discrete/continuous types of predictions
        if Parameters.ogt_continuous_flag == True:
            save_dir = os.path.join(save_dir, 'continuous_predict')
        else:
            save_dir = os.path.join(save_dir, 'discrete_predict')   

    print(f"save_dir = {save_dir}")         

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_dir_train_mod = os.path.join(save_dir, 'trained_models')
    if not os.path.isdir(save_dir_train_mod):
        os.makedirs(save_dir_train_mod)
    return save_dir   

if __name__ == '__main__':

    # 1. Process input parameters
    Parameters = process_args()

    # 2. Process train and test datasets
    print(f"\nProcessing train and test datasets for {Parameters.phenotype} phenotype...\n")
    if Parameters.phenotype == "ogt":
        X_train, X_train_column_names, y_train, X_test, X_test_column_names, y_test, categories_linspace = read_ogt_data(device, NUM_CLASSES_OGT, Parameters.ogt_continuous_flag)
        d_gtdb_train = None
        d_gtdb_test = None
        num_classes = NUM_CLASSES_OGT
    elif Parameters.phenotype == "aerob":
        print(f"Processing train dataset at {Parameters.data_filename_train}...")
        X_train, X_train_column_names, y_train, d_gtdb_train = process_aerob_dataset(Parameters.data_filename_train, Parameters.y_filename, device)
        print(f"Processing test dataset at {Parameters.data_filename_test}...")
        X_test, X_test_column_names, y_test, d_gtdb_test = process_aerob_dataset(Parameters.data_filename_test, Parameters.y_filename, device)
        num_classes = 2
    else:
        print("Specify the right phenotype! Only 'aerob' and 'ogt' are supported for now... ")    
    
    print(f"\nNumber of classes in the classification task is {num_classes}...\n")    

    # 3. Create a directory to save the outputs
    save_dir = create_directory_for_results(Parameters)

    # 4. Run cross-validation on the train dataset
    print(f"Running cross-validation on the train dataset...")
    cross_validation(X_train, y_train, d_gtdb_train, Parameters, device, num_classes, save_dir)

    # 5. Train the model on the train dataset
    print(f"Training the model on the train dataset...")
    net = train(X_train, y_train, Parameters, device, num_classes, save_dir)

    # 6. Test the trained model on the test dataset
    print(f"Testing the model on the test dataset...")
    test_loss, test_accuracy = test(net, X_test, y_test, d_gtdb_test, Parameters, device, save_dir)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")





 
 
 



