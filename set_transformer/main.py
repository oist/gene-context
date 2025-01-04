import argparse
from collections import namedtuple

import torch
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
    

if __name__ == '__main__':

    # 1. Process input parameters
    Parameters = process_args()

    # 2. Process train and test datasets
    print(f"Processing train dataset at {Parameters.data_filename_train}...")
    X_train, y_train, d_gtdb_train = process_dataset(Parameters.data_filename_train, Parameters.y_filename)
    print(f"Processing test dataset at {Parameters.data_filename_test}...")
    X_test, y_test, d_gtdb_test = process_dataset(Parameters.data_filename_test, Parameters.y_filename)

    # 3. Run cross-validation on the train dataset
    print(f"Running cross-validation on the train dataset...")
    cross_validation(X_train, y_train, d_gtdb_train, Parameters, device)

    # 4. Train the model on the train dataset
    print(f"Training the model on the train dataset...")
    net = train(X_train, y_train, Parameters, device)

    # 5. Test the trained model on the test dataset
    print(f"Testing the model on the test dataset...")
    test_loss, test_accuracy = test(net, X_test, y_test, d_gtdb_test, Parameters)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
 
 
 



