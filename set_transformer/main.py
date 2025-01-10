import numpy as np 

import argparse
from collections import namedtuple

import torch
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

from utils.utils import read_xy_data
from set_transformer.train_test_func import cross_validation, train, test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        


def process_dataset(X_filename, y_filename):
    d3_train, X_train, y_train = read_xy_data(X_filename, y_filename)

    d_gtdb_train = d3_train.to_pandas()

    X_train = X_train.drop(columns=["phylum_right", "class_right", "order_right", "family_right", "genus_right"])

    matrix = X_train.values
    X_data = torch.tensor(matrix)
    X_train = X_data.float().to(device)
    X_train_numpy = X_train.cpu().numpy()
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train_numpy)
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)

    y_train = torch.tensor(y_train.values).to(device)
    y_train = y_train.squeeze(1)
    y_train = y_train.float()

    print(f"X = {X_train.shape}")
    print(f"y = {y_train.shape}")

    return X_train, y_train, d_gtdb_train

def process_args():
    parser = argparse.ArgumentParser(description="Process command-line arguments and create a named tuple.")
    
    # Add arguments
    parser.add_argument("--num_inds", type=int, required=True, help="Number of inducing point in SetTransformer.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size.")
    parser.add_argument("--data_filename_train", type=str, required=True, help="Filename of the train dataset.")
    parser.add_argument("--data_filename_test", type=str, required=True, help="Filename of the test dataset.")
    parser.add_argument("--y_filename", type=str, required=True, help="Filename of the label dataset.")
    
    # Parse arguments
    args = parser.parse_args()
    Parameters = namedtuple("Parameters", ["num_inds", "learning_rate", "num_epochs", "batch_size", "data_filename_train", "data_filename_test", "y_filename"])
    return Parameters(num_inds=args.num_inds, learning_rate=args.learning_rate, num_epochs=args.num_epochs, batch_size=args.batch_size, data_filename_train = args.data_filename_train, data_filename_test=args.data_filename_test, y_filename=args.y_filename)
    
def read_ogt_data():

    filename = "data_ogt/kegg.csv"
    df_keggs = pd.read_csv(filename,sep=",")
    # Replace empty or NaN cells with 0
    df_keggs.fillna(0, inplace=True)

    filename_labels = "data_ogt/ogt_splits.csv"
    df_labels = pd.read_csv(filename_labels, sep=",")

    df_merged = pd.merge(df_keggs, df_labels, on='acc', how='inner') 

    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(df_merged["ogt"], bins=50)#, bins=50, edgecolor='black')
  #  plt.show()

    # Split the table based on "ogt_split" values
    df_train = df_merged.loc[df_merged['ogt_split'] == 'train']
    df_test = df_merged.loc[df_merged['ogt_split'] == 'test']

    print(f"df_merged = {df_merged}")

    y_total_unique = []

    # Y train
    y_train = pd.DataFrame(df_train)
    y_train = y_train[['ogt']]
    
    print(f"uniq in train  = {np.unique(y_train.values)}; len = {len(np.unique(y_train.values))}")
    y_total_unique +=  list(np.unique(y_train.values))
    y_train = torch.tensor(y_train.values).to(device)
    y_train = y_train.squeeze(1)
    y_train = y_train.float()

    # X train
    X_train = df_train.drop(columns=["acc", "ogt", "min", "max", "ogt_split", "min_split", "max_split"])
    matrix = X_train.values
    X_data = torch.tensor(matrix)
    X_train = X_data.float().to(device)
    X_train_numpy = X_train.cpu().numpy()
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train_numpy)
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)

    # Y test
    y_test = pd.DataFrame(df_test)
    y_test  = y_test[['ogt']]
    print(f"uniq in test  = {np.unique(y_test.values)}; len = {len(np.unique(y_test.values))}")
    y_total_unique += list(np.unique(y_test.values))
    y_test  = torch.tensor(y_test.values).to(device)
    y_test  = y_test .squeeze(1)
    y_test  = y_test.float()
    print(f"y = {y_test.shape}")
    
    # X test
    X_test = df_test.drop(columns=["acc", "ogt", "min", "max", "ogt_split", "min_split", "max_split"])
    matrix = X_test.values
    X_data = torch.tensor(matrix)
    X_test = X_data.float().to(device)
    X_test_numpy = X_test.cpu().numpy()
    scaler = MaxAbsScaler()
    X_test_scaled = scaler.fit_transform(X_test_numpy)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # Convert to 0-N categories
    y_total_unique = list(np.unique(y_total_unique))
    num_classes = len(y_total_unique)
    print(f"y_total_unique  ={y_total_unique}; len= {num_classes}")

    num_classes = 30

    # Create the linspace
    categories_linspace = np.linspace(min(y_total_unique), max(y_total_unique), num_classes)
    print(f"categories_linspace = {categories_linspace}")

    print(f"y_test = {y_test}")
    
    
    
    indices = np.digitize(y_total_unique, categories_linspace)
    
    print(f"min y_total_unique  ={min(y_total_unique)}; max y_total_unique  ={max(y_total_unique)}")


    y_train_np = y_train.cpu().numpy() if y_train.is_cuda else y_train.numpy()
    y_train = np.digitize(y_train_np, categories_linspace, right=True)#[y_total_unique.index(yi) for yi in y_train]
    y_test_np = y_test.cpu().numpy() if y_test.is_cuda else y_test.numpy()
    y_test = np.digitize(y_test_np, categories_linspace, right=True)#[y_total_unique.index(yi) for yi in y_test]

    print(f"y_test categor = {y_test}")


    plt.figure()
    plt.hist(y_train, bins=50)#, bins=40)#, bins=50, edgecolor='black')

    plt.hist(y_test, bins=50)#, bins=40)#, bins=50, edgecolor='black')
    plt.show()


    y_test  = torch.tensor(y_test).to(device)
   # y_test  = y_test.squeeze(1)
    y_test  = y_test.float()

    y_train  = torch.tensor(y_train).to(device)
   # y_train  = y_train.squeeze(1)
    y_train  = y_train.float()
    print(f"y_test new = {y_test}")


    print(f"y_test = {max(y_test)}")
    print(f"num_classes = {num_classes}")
    print(f"y_train = {y_train}; max =  {max(y_train)}")
    

    return X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device), num_classes

if __name__ == '__main__':

    # 1. Process input parameters
    Parameters = process_args()

    phenotype = "ogt" # "aerob"

    # 2. Process train and test datasets

    if phenotype == "ogt":
        X_train, y_train, X_test, y_test, num_classes = read_ogt_data()
        d_gtdb_train = None
        d_gtdb_test = None
    elif phenotype == "aerob":
        print(f"Processing train dataset at {Parameters.data_filename_train}...")
        X_train, y_train, d_gtdb_train = process_dataset(Parameters.data_filename_train, Parameters.y_filename)
        print(f"Processing test dataset at {Parameters.data_filename_test}...")
        X_test, y_test, d_gtdb_test = process_dataset(Parameters.data_filename_test, Parameters.y_filename)

    # 3. Run cross-validation on the train dataset
    print(f"Running cross-validation on the train dataset...")
    cross_validation(X_train, y_train, d_gtdb_train, Parameters, device, phenotype, num_classes)

    # 4. Train the model on the train dataset
    print(f"Training the model on the train dataset...")
    net = train(X_train, y_train, Parameters, phenotype, num_classes)

    # 5. Test the trained model on the test dataset
    print(f"Testing the model on the test dataset...")
    test_loss, test_accuracy = test(net, X_test, y_test, d_gtdb_test, Parameters, device, phenotype)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
 
 
 



