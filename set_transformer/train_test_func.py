import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from set_transformer.set_transformer_architecture import SetTransformer

MODEL_NAME =  "SetTransformer"
K = 1  # number of embeddings in the SetTransformer model

def train(X_train, y_train, Parameters, device, num_classes, save_dir):
    """
    Trains the model on the train dataset.s
    """
    if Parameters.phenotype == "aerob":
        dim_output = 1
        loss_function = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()  # Example for classification
    elif Parameters.phenotype == "ogt": 
        if Parameters.ogt_continuous_flag == False:
            dim_output = num_classes
            loss_function = torch.nn.CrossEntropyLoss()
        else:
            dim_output = 1    
            loss_function = torch.nn.MSELoss()

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=Parameters.batch_size, shuffle=True, drop_last=True)

    D = X_train.shape[1] # Example input dimension (features per sample)
    net = SetTransformer(D, K, dim_output, Parameters.num_inds, Parameters.dim_hidden, Parameters.num_heads)#.cuda()
    net = net.to(device)
   
    optimizer = optim.AdamW(net.parameters(), lr=Parameters.learning_rate, weight_decay=0.01)#optim.Adam(net.parameters(), lr=lr)

    for epoch in range(Parameters.num_epochs): 

        if epoch == int(0.5*Parameters.num_epochs):
            optimizer.param_groups[0]['lr'] *= 0.1
        net.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:

            optimizer.zero_grad()
            batch_X = batch_X.unsqueeze(1)

            outputs = net(batch_X)
            outputs = outputs.squeeze()

            if Parameters.phenotype == "ogt":     
                batch_y = batch_y.long()
            
            # Calculate loss based on outputs and true label
            loss = loss_function(outputs, batch_y)  

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
        if epoch % 2 == 0:
            print(f"Training epoch # = {epoch}; Loss = {epoch_loss / len(dataloader):.4f}")

    save_path = f"{save_dir}/trained_models/trained_model_{MODEL_NAME}_indPoints_{Parameters.num_inds}_D_{D}_K_{K}_dim_output_{dim_output}.model" 
    torch.save(net.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return net


def count_parameters(model):
    """
    Counts the number of trainable params in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cross_validation(X_train, y_train, d_gtdb_train, Parameters, device, num_classes, save_dir):
    """
    Performs cross-validation of the model on the train dataset.
    """
    num_folds = 5

    from sklearn.model_selection import GroupShuffleSplit

    if Parameters.phenotype == "aerob":
        groups = d_gtdb_train['family'].to_list()
       # group_kfold = GroupKFold(n_splits=num_folds)  
       # fold_train_test_ind = enumerate(group_kfold.split(X_train, y_train, groups=groups))

        gss = GroupShuffleSplit(n_splits=num_folds, test_size=0.2, random_state=42)
        fold_train_test_ind = list(enumerate(gss.split(X_train, y_train, groups=groups)))

        dim_output = 1
    elif Parameters.phenotype == "ogt": 
        kfold = KFold(n_splits=num_folds, shuffle=True)#, random_state=42)
        fold_train_test_ind = enumerate(kfold.split(X_train, y_train))
        if Parameters.ogt_continuous_flag == False:
            dim_output = num_classes
        else:
            dim_output = 1    

    D = X_train.shape[1]

    net = SetTransformer(D, K, dim_output, Parameters.num_inds)
    net = net.to(device)
    num_params = count_parameters(net)
    print(f"The model has {num_params:,} trainable parameters.")

    fold_accuracy = []
    for fold, (train_idx, test_idx) in fold_train_test_ind:
        print(f"\nFold {fold+1}/{num_folds}")
        print("-" * 30)

        # Split the data into training and testing subsets
        train_data = X_train[train_idx].clone().detach().to(torch.float32).to(device)
        test_data = X_train[test_idx].clone().detach().to(torch.float32).to(device)
        if Parameters.phenotype == 'ogt' and Parameters.ogt_continuous_flag == True: 
            train_labels = y_train[train_idx].clone().detach().to(torch.float32).to(device)
            test_labels = y_train[test_idx].clone().detach().to(torch.float32).to(device)
        else:
            train_labels = y_train[train_idx].clone().detach().to(torch.long).to(device)
            test_labels = y_train[test_idx].clone().detach().to(torch.long).to(device)

        # Create DataLoader for mini-batch training
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Parameters.batch_size, shuffle=True) #True

        net = SetTransformer(D, K, dim_output, Parameters.num_inds)#.cuda()
        net = net.to(device)
        optimizer = optim.AdamW(net.parameters(), lr=Parameters.learning_rate, weight_decay=0.01)
        if Parameters.phenotype == "ogt":
            if Parameters.ogt_continuous_flag == False:
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.MSELoss()    
        elif Parameters.phenotype == "aerob":
            criterion = torch.nn.BCEWithLogitsLoss() 

        print(f"criterion = {criterion}")    


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
                

                if Parameters.phenotype == "ogt":
                    if Parameters.ogt_continuous_flag == False:
                        batch_labels = batch_labels.long()
                    else:
                        batch_labels = batch_labels.float()    
                    if outputs.ndimension() == 1:  
                        outputs = outputs.unsqueeze(0)  

              #  print(f"outputs = {outputs}") 
              #  print(f"batch_labels = {batch_labels}")       

                loss = criterion(outputs, batch_labels)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"epoch_loss = {epoch_loss}")
                print(f"len(train_loader) = {len(train_loader)}")
            print(f"Fold {fold + 1} | Epoch {epoch + 1}/{Parameters.num_epochs} | Loss: {epoch_loss / len(train_loader):.4f}")

        
        net.eval()
        with torch.no_grad():
            test_data = test_data.unsqueeze(1)
            test_outputs = net(test_data)

            # Apply sigmoid to get probabilities
            if Parameters.phenotype == "aerob":
                probabilities = torch.sigmoid(test_outputs).squeeze()  # Convert logits to probabilities and remove unnecessary dimensions
                # Apply a threshold of 0.5 to convert probabilities to binary predictions
                test_predictions = (probabilities > 0.5).int() # Convert probabilities to 0 or 1
            elif Parameters.phenotype == "ogt":
                if Parameters.ogt_continuous_flag == False:
                    probabilities = torch.nn.functional.softmax(test_outputs).squeeze()
                    test_predictions = torch.argmax(probabilities, dim=1)
                else:
                    test_predictions = test_outputs.squeeze()[:]    

            if Parameters.phenotype == "ogt" and Parameters.ogt_continuous_flag == True:
                print(f"test_labels = {test_labels.shape}")
                print(f"test_labels = {test_predictions.shape}")
                test_accuracy = mean_squared_error(test_labels.cpu().numpy(), test_predictions.cpu().numpy())
            else:    
                test_accuracy = accuracy_score(test_labels.cpu().numpy(), test_predictions.cpu().numpy())

            if Parameters.phenotype == "ogt" and Parameters.ogt_continuous_flag == True:
                df1 = pd.DataFrame(test_predictions.cpu().numpy(), columns=["prediction"])
                df1['y_actual'] = test_labels.cpu().numpy()
            else:
                probabilities_cpu = probabilities.cpu().numpy()
                df1 = pd.DataFrame(probabilities_cpu)
                df1['prediction'] = test_predictions.cpu().numpy()
                df1['y_actual'] = test_labels.cpu().numpy()

            if Parameters.phenotype == "aerob":
                df1['accession'] = d_gtdb_train.loc[test_idx, 'accession'].values
                df1['false_negative_rate'] = d_gtdb_train.loc[test_idx, 'false_negative_rate'].values
                df1['false_positive_rate'] = d_gtdb_train.loc[test_idx, 'false_positive_rate'].values
            df1['predictor'] = "SetTransformer"

            csv_filename = f"{save_dir}/prediction_probabilities_cross_valid_fold_{fold}_{MODEL_NAME}_indPoints_{Parameters.num_inds}.csv"
            df1.to_csv(csv_filename, index=False, sep="\t", header=True)

        print(f"Fold {fold+1} Accuracy: {test_accuracy:.4f}")
        
        fold_accuracy.append(test_accuracy)
    print(f"Overall cross-validation accuracy over all folds: {fold_accuracy}")
    print(f"Mean cross-validation accuracy = {np.mean(fold_accuracy)}") 

    result_filename = f"model_accuracy_{MODEL_NAME}_NumEp_{Parameters.num_epochs}_InsPoin_{Parameters.num_inds}_LearRate_{Parameters.learning_rate}.txt"
   
    with open(os.path.join(save_dir, result_filename), 'w') as file:
        file.write(f"Cross-Validation Accuracy = {fold_accuracy}\n")  
        file.write(f"Mean Cross-Validation Accuracy = {np.mean(fold_accuracy)}\n")     



def test(net, X_test, y_test, d_gtdb_test, Parameters, device, save_dir):
    """
    Test the model's performance on a test dataset.
    """

    net.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    test_dataset = TensorDataset(X_test, y_test)  # Prepare test data
    test_loader = DataLoader(test_dataset, batch_size=Parameters.batch_size, shuffle=True)

    if Parameters.phenotype == "ogt":
        loss_function = torch.nn.CrossEntropyLoss()
    elif Parameters.phenotype == "aerob":
        loss_function = torch.nn.BCEWithLogitsLoss()

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
            outputs = outputs.squeeze()  # Ensure the shape matches batch_y

            if Parameters.phenotype == "ogt":         
                batch_y = batch_y.long()
                if outputs.ndimension() == 1:  # outputs shape will be [num_classes] when batch_size is 1
                    outputs = outputs.unsqueeze(0)  # Reshape to [1, num_classes]

            loss = loss_function(outputs, batch_y)
            total_loss += loss.item()

            if Parameters.phenotype == "aerob":
                probabilities = torch.sigmoid(outputs).squeeze()  # Convert logits to probabilities and remove unnecessary dimensions
                # Apply a threshold of 0.5 to convert probabilities to binary predictions
                predictions = (probabilities > 0.5) # Convert probabilities to 0 or 1
            elif Parameters.phenotype == "ogt":
                if len(outputs) != 1:
                    probabilities = torch.nn.functional.softmax(outputs, dim=0).squeeze()
                else:    
                    probabilities = torch.nn.functional.softmax(outputs).squeeze()
                if len(probabilities) == 1:
                    probabilities = probabilities.unsqueeze(0)

                predictions = torch.argmax(probabilities, dim=1)

            probabilities_cpu = probabilities.cpu().numpy()    

            # Convert logits to probabilities and then to predictions
            predictions = predictions.to(device)
            correct_predictions += (predictions == batch_y).sum().item()
            total_samples += batch_y.size(0)

            df2 = pd.DataFrame(probabilities_cpu)
            df2['prediction'] =  predictions.int().cpu().numpy()
            df2['y_actual'] =  batch_y.int().cpu().numpy()
            df2['predictor'] =  ['SetTransformer'] * len(batch_y) 

            if Parameters.phenotype == "aerob":
                df2['accession'] = d_gtdb_test.loc[batch_indices, 'accession'].values
                df2['false_negative_rate'] = d_gtdb_test.loc[batch_indices, 'false_negative_rate'].values
                df2['false_positive_rate'] = d_gtdb_test.loc[batch_indices, 'false_positive_rate'].values
            df_list.append(df2)

    df2 = pd.concat(df_list, ignore_index=True)

    csv_filename = f"{save_dir}/prediction_probabilities_holdout_test_{MODEL_NAME}_indPoints_{Parameters.num_inds}.csv"
    df2.to_csv(csv_filename, index=False, sep="\t", header=True)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    result_filename = f"model_accuracy_SetTransformer_NumEp_{Parameters.num_epochs}_InsPoin_{Parameters.num_inds}_LearRate_{Parameters.learning_rate}.txt"
    with open(os.path.join(save_dir, result_filename), 'a') as file:
        file.write(f"Test Data Accuracy = {accuracy}\n")  

    return avg_loss, accuracy