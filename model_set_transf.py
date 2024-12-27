import numpy as np
import logging
import time
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
import os

import polars as pl
import logging 
import torch

import pandas as pd

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from modules import *
from models import SetTransformer 


num_inds = 10#32
num_epochs = 30
lr = 1e-5

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=num_inds, dim_hidden=128, num_heads=4, ln=True):  #num_inds is the number of inducing points m
        super(SetTransformer, self).__init__()
        print("HERE")
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
              #  nn.Dropout(0.3))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):

        return self.dec(self.enc(X))    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

# Input data files
target_column = "oxytolerance"
data_filename = 'data/all_gene_annotations.added_incompleteness_and_contamination.training.tsv'#'data/all_gene_annotations.tsv'
data_filename_test = "data/all_gene_annotations.added_incompleteness_and_contamination.testing.tsv"

#y_filename = "data/bacdive_scrape_20230315.json.parsed.anaerobe_vs_rest.with_cyanos.apply_respiration_gene_exclusion.csv"
#y_filename = "data/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.apply_respiration_gene_set_aerobic.csv"
#y_filename = "data/bacdive_scrape_20230315.json.parsed.anaerobe_vs_rest.with_cyanos.csv"
y_filename = "data/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv"
#y_filename = "data/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.apply_respiration_gene_exclusion.csv"

# Read y

y0 = pl.read_csv(y_filename, separator="\t")
print(f"y name = {y_filename}")
y1 = y0.unique() # There are some duplicates in the cyanos, so dedup
logging.info("Read y: %s", y1.shape)
# Log counts of each class
logging.info("Counts of each class amongst unique accessions: %s", y1.group_by(target_column).agg(pl.len()))

gtdb = pl.concat([
    pl.read_csv('data/bac120_metadata_r202.tsv', separator="\t"),
    pl.read_csv('data/ar122_metadata_r202.tsv', separator="\t")
])

gtdb = gtdb.filter(pl.col("gtdb_representative") == "t")
logging.info("Read in {} GTDB reps".format(len(gtdb)))
gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(1).alias("phylum"))
gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(2).alias("class"))
gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(3).alias("order"))
gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(4).alias("family"))
gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(5).alias("genus"))

# Read training and test data
d = pl.read_csv(data_filename,separator="\t")
d_test = pl.read_csv(data_filename_test,separator="\t")
logging.info("Read training data: %s", d.shape)

# Ignore all but training data
d2 = d.join(gtdb.select(['accession','phylum','class','order','family','genus']), on="accession", how="left")
d2_test = d_test.join(gtdb.select(['accession','phylum','class','order','family','genus']), on="accession", how="left")

d3 = d2.join(y1, on="accession", how="inner") # Inner join because test accessions are in y1 but not in d2
print(f"d3 = {d3}")
d3_test = d2_test.join(y1, on="accession", how="inner") # Inner join because test accessions are in y1 but not in d2
print(f"d3_test = {d3_test}")
print(f"Counts of each class in training/test data: {d3.group_by(target_column).agg(pl.len())}")


d_gtdb = d3.to_pandas()
d_gtdb["false_negative_rate"] = 0
d_gtdb["false_positive_rate"] = 0
d_gtdb_test = d3_test.to_pandas()
d_gtdb_test["false_negative_rate"] = 0
d_gtdb_test["false_positive_rate"] = 0

X = d3.select(pl.exclude(['accession',target_column,'phylum','class','order','family','genus','false_negative_rate','false_positive_rate'])).to_pandas()
X_test = d3_test.select(pl.exclude(['accession',target_column,'phylum','class','order','family','genus','false_negative_rate','false_positive_rate'])).to_pandas()



# Map oxytolerance to 0, 1, 2
if 'anaerobic_with_respiration_genes' in d3['oxytolerance'].to_list():
    classes_map = {
        'anaerobe': 0,
        'aerobe': 1,
        'anaerobic_with_respiration_genes': 2,
    }
else:
    classes_map = {
        'anaerobe': 0,
        'aerobe': 1,
    }

# Generate training y vector with 0s, and 1s
y = d3.select(
    pl.when(pl.col(target_column) == 'anaerobe').then(0)
    .when(pl.col(target_column) == 'aerobe').then(1)
    .when(pl.col(target_column) == 'anaerobic_with_respiration_genes').then(2)
    .otherwise(None)  # Handle cases not in the map
    .alias(target_column)
)
print(f"Counts of y training: {y.group_by(target_column).agg(pl.len())}")
y = y.to_pandas()

# Generate test y vector with 0s, and 1s
y_test = d3_test.select(
    pl.when(pl.col(target_column) == 'anaerobe').then(0)
    .when(pl.col(target_column) == 'aerobe').then(1)
    .when(pl.col(target_column) == 'anaerobic_with_respiration_genes').then(2)
    .otherwise(None)  # Handle cases not in the map
    .alias(target_column)
)
y_test = y_test.to_pandas()

print(f"y_test = {y_test}")

groups = d3['family'].to_list()


# Convert the test and training data to tensors with floats
matrix = X.values
X_data = torch.tensor(matrix).to(device)
X_train = X_data.float()

y_train = torch.tensor(y.values).to(device)
y_train = y_train.squeeze(1)
y_train = y_train.float()

print(f"X_train shape = {X_train.shape}")
print(f"y_train shape = {y_train.shape}")

matrix = X_test.values
X_data_test = torch.tensor(matrix)
X_test = X_data_test.float()

y_test = torch.tensor(y_test.values)
y_test = y_test.squeeze(1)
y_test = y_test.float()

print(f"X_train shape = {X_test.shape}")
print(f"y_train shape = {y_test.shape}")

# Directory to save 
run_name = 'bact_data'
save_dir = os.path.join('results', 'SetTransformer', run_name)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

save_results_dir  = os.path.join('results', 'SetTransformer', 'accuracy_results')
if not os.path.isdir(save_results_dir):
    os.makedirs(save_results_dir)

############### Train SetTransformer ###############

# 2. Set the model


batch_size = 20
#lr = 1e-3
test_freq = 0.1     
save_freq = 0.1 
D = X_train.shape[1] # 2677#15696#50  # Example input dimension (features per sample)
K = 1#2#10   # number of embeddings 
dim_output = 1#2# K  # i.e. probabil of belonging to each cluster
net = SetTransformer(D, K, dim_output)#.cuda() #  dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False

result_filename = f"model_accuracy_SetTransformer_NumEp_{num_epochs}_InsPoin_{num_inds}_LearRate_{lr}.txt"

# 4. Training
def train():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(run_name)
    logger.addHandler(logging.FileHandler(
        os.path.join(save_dir,
            'train_'+time.strftime('%Y%m%d-%H%M')+'.log'),
        mode='w'))
    #logger.info(str(args) + '\n')

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    net = SetTransformer(D, K, dim_output)#.cuda()

   
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay = 0.01)
    loss_function = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()  # Example for classification
    tick = time.time()
    net.train()
    for epoch in range(num_epochs):  #for t in range(1, num_steps+1): # num

       # if epoch == int(0.5*num_epochs):
       #     optimizer.param_groups[0]['lr'] *= 0.1
        
        epoch_loss = 0
        for batch_X, batch_y in dataloader:

           # print(f"batch_X.shape = {batch_X.shape}")

            optimizer.zero_grad()
            batch_X = batch_X.unsqueeze(1)

            outputs = net(batch_X)
            
            outputs = outputs.squeeze()
            #print(f"!!!!!!!!outputs  = {outputs}")
            
            # Calculate loss based on outputs and true label
            loss = loss_function(outputs, batch_y)  

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
       
       
        if epoch % 2 == 0:
            print(f"Training epoch # = {epoch}; Loss = {epoch_loss}")


        if epoch % test_freq == 0:
            line = 'epoch {}, lr {:.3e}, '.format(
                    epoch, optimizer.param_groups[0]['lr'])
           # line += test(bench, verbose=False)
            line += ' ({:.3f} secs)'.format(time.time()-tick)
            tick = time.time()
            logger.info(line)

        if epoch % save_freq == 0:
            torch.save({'state_dict':net.state_dict()},
                    os.path.join(save_dir, 'model.tar'))

    torch.save({'state_dict':net.state_dict()},
        os.path.join(save_dir, 'model.tar'))
    


    return net



def train_cross_valid():
    num_folds = 5
    group_kfold = GroupKFold(n_splits=num_folds)   

    net = SetTransformer(D, K, dim_output)#.cuda()
    num_params = count_parameters(net)
    print(f"The model has {num_params:,} trainable parameters.")

    # Optimizer and loss function
    #lr = 1e-3

    fold_accuracy = []
    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X_train, y_train, groups=groups)):
        print(f"\nFold {fold+1}/{num_folds}")
        print("-" * 30)

        # Split the data into training and testing subsets
        train_data = torch.tensor(X_train[train_idx], dtype=torch.float32).to(device)
        train_labels = torch.tensor(y_train[train_idx], dtype=torch.long).to(device)
        test_data = torch.tensor(X_train[test_idx], dtype=torch.float32).to(device)
        test_labels = torch.tensor(y_train[test_idx], dtype=torch.long).to(device)

        # Create DataLoader for mini-batch training
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        net = SetTransformer(D, K, dim_output)#.cuda()
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
        criterion = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()  # Example for classification

        # Training loop
        net.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()

                batch_X = batch_data.unsqueeze(1)
                
                outputs = net(batch_X)
                
                outputs = outputs.squeeze()


                batch_labels = batch_labels.float()
                loss = criterion(outputs, batch_labels)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Fold {fold + 1} | Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss / len(train_loader):.4f}")

        


        from sklearn.metrics import accuracy_score
        net.eval()
        with torch.no_grad():
            test_data = test_data.unsqueeze(1)
            test_outputs = net(test_data)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(test_outputs).squeeze()  # Convert logits to probabilities and remove unnecessary dimensions
            
            # Apply a threshold of 0.5 to convert probabilities to binary predictions
            test_predictions = (probabilities > 0.5).int().cpu().numpy()  # Convert probabilities to 0 or 1
        
            test_accuracy = accuracy_score(test_labels.cpu().numpy(), test_predictions)

            df1 = pd.DataFrame(test_predictions, columns=['prediction'])    
            df1['y_actual'] = test_labels.cpu().numpy()
            probabilities_cpu = probabilities.cpu().numpy()
            df1['probabilities'] = probabilities_cpu
            df1['accession'] = d_gtdb.loc[test_idx, 'accession'].values
            df1['false_negative_rate'] = d_gtdb.loc[test_idx, 'false_negative_rate'].values
            df1['false_positive_rate'] = d_gtdb.loc[test_idx, 'false_positive_rate'].values
            df1['predictor'] = "SetTransformer"

            model_name =  "SetTransformer"
            csv_filename = f"resuls_SetTransformer/prediction_probabilities_cross_valid_fold_{fold}_{model_name}_indPoints_{num_inds}.csv"
            df1.to_csv(csv_filename, index=False, sep="\t", header=True)


        print(f"Fold {fold+1} Accuracy: {test_accuracy:.4f}")
        
        fold_accuracy.append(test_accuracy)
    print(f"Overall cross-validation accuracy over all folds: {fold_accuracy}")
    print(f"Mean cross-validation accuracy = {np.mean(fold_accuracy)}")    

    with open(os.path.join(save_results_dir, result_filename), 'w') as file:
        file.write(f"Cross-Validation Accuracy = {fold_accuracy}\n")  
        file.write(f"Mean Cross-Validation Accuracy = {np.mean(fold_accuracy)}\n")     



def test(net):
    """
    Test the model's performance on a test dataset.
    
    Parameters:
        net (torch.nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        loss_function (torch.nn.Module): Loss function used for evaluation.

    Returns:
        tuple: Average test loss and accuracy.
    """
    net.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    test_dataset = TensorDataset(X_test, y_test)  # Prepare test data
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    loss_function = torch.nn.BCEWithLogitsLoss()

    df2 = pd.DataFrame(columns=['prediction', 'y_actual', 'probabilities', 'accession', 'false_negative_rate', 'false_positive_rate', 'predictor'])


    with torch.no_grad():  # Disable gradient computation for testing
        for batch_idx, (batch_X, batch_y) in enumerate(test_loader):#for batch_X, batch_y in test_loader:
            start_idx = batch_idx * test_loader.batch_size  # Start index of this batch
            end_idx = start_idx + len(batch_X)  # End index
            batch_indices = list(range(start_idx, end_idx))  # Indices for this batch

         #   print(f"batch_indices = {batch_indices}")

            #Append false_negative_rate
         #   new_fal_neg_rows = pd.DataFrame({'false_negative_rate':d_gtdb_test.loc[batch_indices, 'false_negative_rate'].values})
        #    df2 = pd.concat([df2, new_fal_neg_rows], ignore_index=True)

            #Append false_positive_rate
           # new_fal_pos_rows = pd.DataFrame({'false_positive_rate':d_gtdb_test.loc[batch_indices, 'false_positive_rate'].values})
          #  df2 = pd.concat([df2, new_fal_pos_rows], ignore_index=True)

            #Append accession
           # new_accession_rows = pd.DataFrame({'accession':d_gtdb_test.loc[batch_indices, 'accession'].values})
          #  df2 = pd.concat([df2, new_accession_rows], ignore_index=True)

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            #Append y_actual to df
           # new_y_actual_rows = pd.DataFrame({'y_actual': batch_y.cpu().numpy()})
           # df2 = pd.concat([df2, new_y_actual_rows], ignore_index=True)

            batch_X = batch_X.unsqueeze(1)  # Add the required dimension
            outputs = net(batch_X)

            #Calculate probabilities
            probabilities = torch.sigmoid(outputs).squeeze()
            probabilities_cpu = probabilities.cpu().numpy()

            #Append probabilities to df
           # new_prob_rows = pd.DataFrame({'probabilities': probabilities_cpu})
           # df2 = pd.concat([df2, new_prob_rows], ignore_index=True)

            outputs = outputs.squeeze()  # Ensure the shape matches batch_y
            loss = loss_function(outputs, batch_y)
            total_loss += loss.item()

            # Convert logits to probabilities and then to predictions
            predictions = (torch.sigmoid(outputs) > 0.5)#.int().cpu().numpy()

            #Append predictions to df
           # new_predict_rows = pd.DataFrame({'prediction': predictions.int().cpu().numpy()})
          #  df2 = pd.concat([df2, new_predict_rows], ignore_index=True)

            predictions = predictions.to(device)
            correct_predictions += (predictions == batch_y).sum().item()
            total_samples += batch_y.size(0)


            # Prepare a DataFrame for the current batch
            batch_data = {
                'false_negative_rate': d_gtdb_test.loc[batch_indices, 'false_negative_rate'].values,
                'false_positive_rate': d_gtdb_test.loc[batch_indices, 'false_positive_rate'].values,
                'accession': d_gtdb_test.loc[batch_indices, 'accession'].values,
                'prediction': predictions.int().cpu().numpy(),
                'y_actual': batch_y.int().cpu().numpy(),
                'probabilities': probabilities_cpu,
                'predictor': ['SetTransformer'] * len(batch_y)  # Assuming 'SetTransformer' is the value for the predictor column
            }

            new_batch_df = pd.DataFrame(batch_data)

            df2 = pd.concat([df2, new_batch_df], ignore_index=True)

    model_name =  "SetTransformer"
  # df2['predictor'] = model_name

    csv_filename = f"resuls_SetTransformer/prediction_probabilities_holdout_test_{model_name}_indPoints_{num_inds}.csv"
    df2.to_csv(csv_filename, index=False, sep="\t", header=True)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    with open(os.path.join(save_results_dir, result_filename), 'a') as file:
        file.write(f"Test Data Accuracy = {accuracy}\n")  

    return avg_loss, accuracy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    print("Training the model...")
    # Cross-validation
  #  train_cross_valid()

    # Training
    net = train()

    # Evaluate the model on the test dataset
    test_loss, test_accuracy = test(net)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
 
 
 



