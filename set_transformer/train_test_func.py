import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from set_transformer.set_transformer_architecture import SetTransformer


K = 1#2#10   # number of embeddings 
dim_output = 1#2# K  # i.e. probabil of belonging to each cluster
#net = SetTransformer(D, K, dim_output)#.cuda() #  dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False


# Directory to save 
run_name = 'bact_data'
save_dir = os.path.join('results', 'SetTransformer', run_name)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)



save_results_dir  = os.path.join('results', 'SetTransformer', 'accuracy_results')
if not os.path.isdir(save_results_dir):
    os.makedirs(save_results_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

test_freq = 0.1     
save_freq = 0.1 

# 4. Training
def train(X_train, y_train, Parameters, phenotype, num_classes):

    if phenotype == "aerob":
        dim_output = 1
        loss_function = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()  # Example for classification
    elif phenotype == "ogt": 
        dim_output = num_classes
        loss_function = torch.nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=Parameters.batch_size, shuffle=True, drop_last=True)

    D = X_train.shape[1] # 2677#15696#50  # Example input dimension (features per sample)

    net = SetTransformer(D, K, dim_output)#.cuda()
    net = net.to(device)
   
    optimizer = optim.AdamW(net.parameters(), lr=Parameters.learning_rate, weight_decay=0.01)#optim.Adam(net.parameters(), lr=lr)

    tick = time.time()
    for epoch in range(Parameters.num_epochs):  #for t in range(1, num_steps+1): # num

        if epoch == int(0.5*Parameters.num_epochs):
            optimizer.param_groups[0]['lr'] *= 0.1
        net.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:

            optimizer.zero_grad()
            batch_X = batch_X.unsqueeze(1)

            outputs = net(batch_X)
            outputs = outputs.squeeze()

            if phenotype == "ogt":     
                batch_y = batch_y.long()
            
            # Calculate loss based on outputs and true label
            loss = loss_function(outputs, batch_y)  

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
       
       
        if epoch % 2 == 0:
            print(f"Training epoch # = {epoch}; Loss = {epoch_loss / len(dataloader):.4f}")

        if epoch % test_freq == 0:
            line = 'epoch {}, lr {:.3e}, '.format(
                    epoch, optimizer.param_groups[0]['lr'])
           # line += test(bench, verbose=False)
            line += ' ({:.3f} secs)'.format(time.time()-tick)
            tick = time.time()
           # logger.info(line)

        if epoch % save_freq == 0:
            torch.save({'state_dict':net.state_dict()},
                    os.path.join(save_dir, 'model.tar'))

    torch.save({'state_dict':net.state_dict()},
        os.path.join(save_dir, 'model.tar'))

    model_name =  "SetTransformer"    

    save_path = f"set_transformer/resuls_SetTransformer/trained_models/trained_model_{model_name}_phenotype_{phenotype}_indPoints_{Parameters.num_inds}_D_{D}_K_{K}_dim_output_{dim_output}.model" 
    torch.save(net.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return net



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cross_validation(X_train, y_train, d_gtdb_train, Parameters, device, phenotype, num_classes):
    num_folds = 5

    if phenotype == "aerob":
        groups = d_gtdb_train['family'].to_list()# d3_train['family'].to_list()
        group_kfold = GroupKFold(n_splits=num_folds)  
        fold_train_test_ind = enumerate(group_kfold.split(X_train, y_train, groups=groups))
        dim_output = 1
    elif phenotype == "ogt": 
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_train_test_ind = enumerate(kfold.split(X_train, y_train))
        dim_output = num_classes

    D = X_train.shape[1]

    net = SetTransformer(D, K, dim_output, Parameters.num_inds)#.cuda()
    net = net.to(device)
    num_params = count_parameters(net)
    print(f"The model has {num_params:,} trainable parameters.")

    fold_accuracy = []
    for fold, (train_idx, test_idx) in fold_train_test_ind:
        print(f"\nFold {fold+1}/{num_folds}")
        print("-" * 30)

        # Split the data into training and testing subsets
        train_data = X_train[train_idx].clone().detach().to(torch.float32).to(device)
        train_labels = y_train[train_idx].clone().detach().to(torch.long).to(device)
        test_data = X_train[test_idx].clone().detach().to(torch.float32).to(device)
        test_labels = y_train[test_idx].clone().detach().to(torch.long).to(device)

        # Create DataLoader for mini-batch training
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Parameters.batch_size, shuffle=True) #True

        net = SetTransformer(D, K, dim_output, Parameters.num_inds)#.cuda()
        net = net.to(device)
        optimizer = optim.AdamW(net.parameters(), lr=Parameters.learning_rate, weight_decay=0.01)
        if phenotype == "ogt":
            criterion = torch.nn.CrossEntropyLoss()
        elif phenotype == "aerob":
            criterion = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss() 

        # Training loop
        net.train()
        for epoch in range(Parameters.num_epochs):
            epoch_loss = 0.0
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()

                batch_X = batch_data.unsqueeze(1)
                
                outputs = net(batch_X)
                outputs = outputs.squeeze()

                batch_labels = batch_labels.float()
                batch_labels = batch_labels.long()


                if outputs.ndimension() == 1:  # outputs shape will be [num_classes] when batch_size is 1
                    outputs = outputs.unsqueeze(0)  # Reshape to [1, num_classes]

             #   print(f"outputs in CV  = {outputs}; len = {len(outputs)}")
             #   print(f"batch_labels  in CV = {batch_labels}; len = {len(batch_labels)}")
                loss = criterion(outputs, batch_labels)

              #  print(f"loss = {loss}")

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Fold {fold + 1} | Epoch {epoch + 1}/{Parameters.num_epochs} | Loss: {epoch_loss / len(train_loader):.4f}")

        
        net.eval()
        with torch.no_grad():
            test_data = test_data.unsqueeze(1)
            test_outputs = net(test_data)

           # print(f"test_outputs = {test_outputs}; len = {len(test_outputs)}; shape = {test_labels.shape}")
           # print(f"test_labels = {test_labels}; len = {len(test_labels)}; shape = {test_labels.shape}")
            
            # Apply sigmoid to get probabilities
            if phenotype == "aerob":
                probabilities = torch.sigmoid(test_outputs).squeeze()  # Convert logits to probabilities and remove unnecessary dimensions
                # Apply a threshold of 0.5 to convert probabilities to binary predictions
                test_predictions = (probabilities > 0.5).int().cpu().numpy()  # Convert probabilities to 0 or 1
            elif phenotype == "ogt":
                probabilities = torch.nn.functional.softmax(test_outputs).squeeze()
              #  print(f"probabilities = {probabilities}")
                test_predictions = torch.argmax(probabilities, dim=1)
               # print(f"test_predictions = {test_predictions}")

            test_accuracy = accuracy_score(test_labels.cpu().numpy(), test_predictions.cpu().numpy())
         #   print(f"test_accuracy = {test_accuracy}")

            probabilities_cpu = probabilities.cpu().numpy()
            df1 = pd.DataFrame(probabilities_cpu)
            df1['prediction'] = test_predictions.cpu().numpy()

           # df1 = pd.DataFrame(test_predictions, columns=['prediction'])    
            df1['y_actual'] = test_labels.cpu().numpy()
          #  df1['probabilities'] = probabilities_cpu
            
            if phenotype == "aerob":
                df1['accession'] = d_gtdb_train.loc[test_idx, 'accession'].values
                df1['false_negative_rate'] = d_gtdb_train.loc[test_idx, 'false_negative_rate'].values
                df1['false_positive_rate'] = d_gtdb_train.loc[test_idx, 'false_positive_rate'].values
            df1['predictor'] = "SetTransformer"

            model_name =  "SetTransformer"
            csv_filename = f"set_transformer/resuls_SetTransformer/prediction_probabilities_cross_valid_fold_{fold}_{model_name}_indPoints_{Parameters.num_inds}.csv"
            df1.to_csv(csv_filename, index=False, sep="\t", header=True)


        print(f"Fold {fold+1} Accuracy: {test_accuracy:.4f}")
        
        fold_accuracy.append(test_accuracy)
    print(f"Overall cross-validation accuracy over all folds: {fold_accuracy}")
    print(f"Mean cross-validation accuracy = {np.mean(fold_accuracy)}") 

    result_filename = f"model_accuracy_SetTransformer_NumEp_{Parameters.num_epochs}_InsPoin_{Parameters.num_inds}_LearRate_{Parameters.learning_rate}.txt"
   

    with open(os.path.join(save_results_dir, result_filename), 'w') as file:
        file.write(f"Cross-Validation Accuracy = {fold_accuracy}\n")  
        file.write(f"Mean Cross-Validation Accuracy = {np.mean(fold_accuracy)}\n")     



def test(net, X_test, y_test, d_gtdb_test, Parameters, device, phenotype):
    """
    Test the model's performance on a test dataset.
    
    Parameters:
        net (torch.nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        loss_function (torch.nn.Module): Loss function used for evaluation.

    Returns:
        tuple: Average test loss and accuracy.
    """
    print(f"test X shape = {X_test.shape}")
    print(f"test y  shape = {y_test.shape}")
    net.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    test_dataset = TensorDataset(X_test, y_test)  # Prepare test data
    test_loader = DataLoader(test_dataset, batch_size=Parameters.batch_size, shuffle=True)

    if phenotype == "ogt":
        loss_function = torch.nn.CrossEntropyLoss()
    elif phenotype == "aerob":
        loss_function = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss() 

   # df2 = pd.DataFrame()#columns=['prediction', 'y_actual', 'probabilities', 'accession', 'false_negative_rate', 'false_positive_rate', 'predictor'])

    df_list = []

    with torch.no_grad():  # Disable gradient computation for testing
        for batch_idx, (batch_X, batch_y) in enumerate(test_loader):#for batch_X, batch_y in test_loader:
            start_idx = batch_idx * test_loader.batch_size  # Start index of this batch
            end_idx = start_idx + len(batch_X)  # End index
            batch_indices = list(range(start_idx, end_idx))  # Indices for this batch

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)


            batch_X = batch_X.unsqueeze(1)  # Add the required dimension
            outputs = net(batch_X)

            #Calculate probabilities
           # probabilities = torch.sigmoid(outputs).squeeze()
            
            outputs = outputs.squeeze()  # Ensure the shape matches batch_y
        #    print(f"outputs in test  = {outputs}; len = {len(outputs)}")
         #   print(f"batch_y  in test = {batch_y}; len = {len(batch_y)}")
         
            if phenotype == "ogt":         
                batch_y = batch_y.long()
                if outputs.ndimension() == 1:  # outputs shape will be [num_classes] when batch_size is 1
                    outputs = outputs.unsqueeze(0)  # Reshape to [1, num_classes]

            loss = loss_function(outputs, batch_y)
            total_loss += loss.item()

            if phenotype == "aerob":
                probabilities = torch.sigmoid(outputs).squeeze()  # Convert logits to probabilities and remove unnecessary dimensions
                # Apply a threshold of 0.5 to convert probabilities to binary predictions
                predictions = (probabilities > 0.5)#.int().cpu().numpy()  # Convert probabilities to 0 or 1
            elif phenotype == "ogt":
                if len(outputs) != 1:
                    probabilities = torch.nn.functional.softmax(outputs, dim=0).squeeze()
                else:    
                    probabilities = torch.nn.functional.softmax(outputs).squeeze()
                if len(probabilities) == 1:#!= Parameters.batch_size:
                    probabilities = probabilities.unsqueeze(0)
              #  print(f"probabilities in test = {probabilities}; len = {len(probabilities)}")    

                predictions = torch.argmax(probabilities, dim=1)



            probabilities_cpu = probabilities.cpu().numpy()    

            # Convert logits to probabilities and then to predictions
           # predictions = (torch.sigmoid(outputs) > 0.5)#.int().cpu().numpy()

            predictions = predictions.to(device)
            correct_predictions += (predictions == batch_y).sum().item()
            total_samples += batch_y.size(0)

            df2 = pd.DataFrame(probabilities_cpu)
           # df2 = pd.concat([df2, new_batch_df], ignore_index=True)
            df2['prediction'] =  predictions.int().cpu().numpy()
            df2['y_actual'] =  batch_y.int().cpu().numpy()
            df2['predictor'] =  ['SetTransformer'] * len(batch_y) 


            # Prepare a DataFrame for the current batch
         #   batch_data = {
               # 'false_negative_rate': d_gtdb_test.loc[batch_indices, 'false_negative_rate'].values,
              #  'false_positive_rate': d_gtdb_test.loc[batch_indices, 'false_positive_rate'].values,
              #  'accession': d_gtdb_test.loc[batch_indices, 'accession'].values,
            #     'prediction': predictions.int().cpu().numpy(),
            #     'y_actual': batch_y.int().cpu().numpy(),
            #     'probabilities': probabilities_cpu,
            #     'predictor': ['SetTransformer'] * len(batch_y)  # Assuming 'SetTransformer' is the value for the predictor column
            # }

            if phenotype == "aerob":
                df2['accession'] = d_gtdb_test.loc[batch_indices, 'accession'].values
                df2['false_negative_rate'] = d_gtdb_test.loc[batch_indices, 'false_negative_rate'].values
                df2['false_positive_rate'] = d_gtdb_test.loc[batch_indices, 'false_positive_rate'].values

           # new_batch_df = pd.DataFrame(batch_data)

            df_list.append(df2)

         #   df2 = pd.concat([df2, new_batch_df], ignore_index=True)

    model_name =  "SetTransformer"

    df2 = pd.concat(df_list, ignore_index=True)

    csv_filename = f"set_transformer/resuls_SetTransformer/prediction_probabilities_holdout_test_{model_name}_indPoints_{Parameters.num_inds}.csv"
    df2.to_csv(csv_filename, index=False, sep="\t", header=True)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    result_filename = f"model_accuracy_SetTransformer_NumEp_{Parameters.num_epochs}_InsPoin_{Parameters.num_inds}_LearRate_{Parameters.learning_rate}.txt"


    with open(os.path.join(save_results_dir, result_filename), 'a') as file:
        file.write(f"Test Data Accuracy = {accuracy}\n")  

    return avg_loss, accuracy