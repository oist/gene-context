import os
import random 
import torch
import logging 
import pandas as pd
import numpy as np
import polars as pl
import argparse
from matplotlib import cm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score

from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MaxAbsScaler

from sklearn.model_selection import cross_val_predict, KFold
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error,r2_score

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_ogt_data(device, num_class, ogt_continuous_flag, precence_only_flag = False):
    # Read the csv file with keggs
    try:
        filename = "data_ogt/kegg.csv"
        df_keggs = pd.read_csv(filename,sep=",")
    except FileNotFoundError as e: 
        filename = "../data_ogt/kegg.csv"
        df_keggs = pd.read_csv(filename,sep=",")

    # Replace empty or NaN cells with 0
    df_keggs.fillna(0, inplace=True)

    # Read the csv file with the splits 
    try:
        filename_labels = "data_ogt/ogt_splits.csv"
        df_labels = pd.read_csv(filename_labels, sep=",")
        df_merged = pd.merge(df_keggs, df_labels, on='acc', how='inner') 
    except FileNotFoundError as e: 
        filename_labels = "../data_ogt/ogt_splits.csv"
        df_labels = pd.read_csv(filename_labels, sep=",")
        df_merged = pd.merge(df_keggs, df_labels, on='acc', how='inner') 
    # Split the table based on "ogt_split" values
    df_train = df_merged.loc[df_merged['ogt_split'] == 'train']
    df_test = df_merged.loc[df_merged['ogt_split'] == 'test']
    y_total_unique = []

    # Y train
    y_train = pd.DataFrame(df_train)
    y_train = y_train[['ogt']]
    y_total_unique +=  list(np.unique(y_train.values))
    y_train = torch.tensor(y_train.values).to(device)
    y_train = y_train.squeeze(1)
    y_train = y_train.float()

    # X train
    X_train = df_train.drop(columns=["acc", "ogt", "min", "max", "ogt_split", "min_split", "max_split"])
    X_train_column_names = X_train.columns
    matrix = X_train.values
    X_data = torch.tensor(matrix)
    X_train = X_data.float().to(device)
    X_train_numpy = X_train.cpu().numpy()
    if precence_only_flag == True:
        X_train_numpy = (X_train_numpy > 0).astype(int)
    scaler = MaxAbsScaler()
    X_train_scaled = X_train_numpy#scaler.fit_transform(X_train_numpy)
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)

    # Y test
    y_test = pd.DataFrame(df_test)
    y_test  = y_test[['ogt']]
    y_total_unique += list(np.unique(y_test.values))
    y_test  = torch.tensor(y_test.values).to(device)
    y_test  = y_test .squeeze(1)
    y_test  = y_test.float()
    
    # X test
    X_test = df_test.drop(columns=["acc", "ogt", "min", "max", "ogt_split", "min_split", "max_split"])
    X_test_column_names = X_test.columns
    matrix = X_test.values
    X_data = torch.tensor(matrix)
    X_test = X_data.float().to(device)
    X_test_numpy = X_test.cpu().numpy()
    if precence_only_flag == True:
        X_test_numpy = (X_test_numpy > 0).astype(int)
    #scaler = MaxAbsScaler()
    X_test_scaled = X_test_numpy#scaler.fit_transform(X_test_numpy)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # Convert to 0-N categories
    y_total_unique = list(np.unique(y_total_unique))

    # Create the linspace and distribute the point sto categories
    categories_linspace = np.linspace(min(y_total_unique), max(y_total_unique), num_class)

    y_train_np = y_train.cpu().numpy() if y_train.is_cuda else y_train.numpy()
    y_test_np = y_test.cpu().numpy() if y_test.is_cuda else y_test.numpy()
    if ogt_continuous_flag == True:
        y_train = y_train_np[:]
        y_test = y_test_np[:]
    else:
        y_train = np.digitize(y_train_np, categories_linspace, right=True)
        y_test = np.digitize(y_test_np, categories_linspace, right=True)
    

    # Convert labels to the right format
    y_test  = torch.tensor(y_test).to(device)
    y_test  = y_test.float()
    y_train  = torch.tensor(y_train).to(device)
    y_train  = y_train.float()

    return X_train.to(device), X_train_column_names, y_train.to(device), X_test.to(device), X_test_column_names, y_test.to(device), categories_linspace

def read_diderm_data(X_filename, y_filename, device):
    df_x_data = pd.read_csv(X_filename,sep="\t")

    X_train_column_names = df_x_data.columns

    df_y_labels = pd.read_csv(y_filename,sep="\t")

    df_merged = pd.merge(df_x_data, df_y_labels, on='accession', how='inner') 

    X_val = df_merged.drop(columns=['high_throughput_dermy', 'accession']).values
    X_val = torch.tensor(X_val)
    X_val = X_val.float().to(device)
    X_val_numpy = X_val.cpu().numpy()
 #   scaler = MaxAbsScaler()
   # X_val_scaled = scaler.fit_transform(X_val_numpy)
    X_val = torch.tensor(X_val_numpy, dtype=torch.float32).to(device)

    y_label = df_merged["high_throughput_dermy"].map({'Diderm': 0, 'Monoderm': 1})
    y_label = torch.tensor(y_label.values).to(device)
   # y_label = y_label.squeeze(1)
    y_label = y_label.float()

    return X_val, y_label, X_train_column_names[1:]

def process_aerob_dataset(X_filename, y_filename, device, remove_noise):
    d3_train, X_train, y_train = read_xy_data(X_filename, y_filename, remove_noise)
    d_gtdb_train = d3_train.to_pandas()

   # X_train = X_train.drop(columns=["family_right", "phylum_right", "class_right", "order_right", "genus_right"])
    X_train_column_names = X_train.columns

    matrix = X_train.values
    X_data = torch.tensor(matrix)
    X_train = X_data.float().to(device)
    X_train_numpy = X_train.cpu().numpy()
    scaler = MaxAbsScaler()
   # X_train_scaled = scaler.fit_transform(X_train_numpy)
    X_train = torch.tensor(X_train_numpy, dtype=torch.float32).to(device).float()

    y_train = torch.tensor(y_train.values).to(device)
    y_train = y_train.squeeze(1)
    y_train = y_train.float()

    return X_train, X_train_column_names, y_train, d_gtdb_train


def read_xy_data(data_filename, y_filename, remove_noise = True):

    try:
        gtdb = pl.concat([
            pl.read_csv('data_aerob/bac120_metadata_r202.tsv', separator="\t"),
            pl.read_csv('data_aerob/ar122_metadata_r202.tsv', separator="\t")
        ])
    except FileNotFoundError as e:  
        gtdb = pl.concat([
            pl.read_csv('../data_aerob/bac120_metadata_r202.tsv', separator="\t"),
            pl.read_csv('../data_aerob/ar122_metadata_r202.tsv', separator="\t")
        ])
    gtdb = gtdb.filter(pl.col("gtdb_representative") == "t")
    logging.info("Read in {} GTDB reps".format(len(gtdb)))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(1).alias("phylum"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(2).alias("class"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(3).alias("order"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(4).alias("family"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(5).alias("genus"))

    
    target_column = "oxytolerance"
    # Read y
    y0 = pl.read_csv(y_filename, separator="\t")
    y1 = y0.unique() # There are some duplicates in the cyanos, so dedup
    logging.info("Read y: %s", y1.shape)
    # Log counts of each class
    logging.info("Counts of each class amongst unique accessions: %s", y1.group_by(target_column).agg(pl.len()))
    
    # Read the data
    d = pl.read_csv(data_filename,separator="\t")
    d2 = d.join(gtdb.select(['accession','phylum','class','order','family','genus']), on="accession", how="left")

    d3 = d2.join(y1, on="accession", how="inner") # Inner join because test accessions are in y1 but not in d2

    if remove_noise == True:
        d3 = d3.filter(pl.col("false_negative_rate") == 0)
        d3 = d3.filter(pl.col("false_positive_rate") == 0)

    print(f"Counts of each class in training/test data: {d3.group_by(target_column).agg(pl.len())}")
    
    d_gtdb = d3.to_pandas()
    X = d3.select(pl.exclude(['accession',target_column,'phylum','class','order','family','genus','false_negative_rate','false_positive_rate'])).to_pandas()

    # Blacklist these as they aren't in the current ancestral file, not sure why
    X = X.drop(['COG0411', 'COG0459', 'COG0564', 'COG1344', 'COG4177'], axis=1)


    # Generate y vector with 0s, and 1s
    y = d3.select(
        pl.when(pl.col(target_column) == 'anaerobe').then(0)
        .when(pl.col(target_column) == 'aerobe').then(1)
        .when(pl.col(target_column) == 'anaerobic_with_respiration_genes').then(2)
        .otherwise(None)  # Handle cases not in the map
        .alias(target_column)
    )
    y = y.to_pandas()
    return d3, X, y

def table_row_subsampling(d3):

   target_column = "oxytolerance"

   X = d3.select(pl.exclude([target_column])).to_pandas() #'phylum','class','order','family','genus'
   
   # Generate y vector with 0s, and 1s
   y = d3.select(
       pl.when(pl.col(target_column) == 'anaerobe').then(0)
       .when(pl.col(target_column) == 'aerobe').then(1)
       .when(pl.col(target_column) == 'anaerobic_with_respiration_genes').then(2)
       .otherwise(None)  # Handle cases not in the map
       .alias(target_column)
   )
   y = y.to_pandas()

   num_aerobs = y['oxytolerance'].sum()
   num_anaerobs = len(y['oxytolerance']) - num_aerobs
   
   # Sub-sampling training data
   indices_aerobs = y[y['oxytolerance'] == 1].index.tolist()
   indices_anaerobs = y[y['oxytolerance'] == 0].index.tolist()

   if num_aerobs > num_anaerobs:
       print(f"Sub-sampling {num_aerobs} aerobs to {num_anaerobs} anaerobs")
       subsampled_aerobs = random.sample(indices_aerobs, num_anaerobs)
       final_row_indices = subsampled_aerobs + indices_anaerobs
   else:
       print(f"Sub-sampling {indices_anaerobs} aerobs to {num_aerobs} anaerobs")
       subsampled_anaerobs = random.sample(indices_anaerobs, num_aerobs)
       final_row_indices = subsampled_anaerobs + indices_aerobs

   X_subsampled = X.iloc[final_row_indices].reset_index(drop=True)
   y_subsampled = y.iloc[final_row_indices].reset_index(drop=True)

   print(f"Sub-sampled table length = {len(y_subsampled)} with { y_subsampled['oxytolerance'].sum()} aerobs and  {len(y_subsampled['oxytolerance']) - y_subsampled['oxytolerance'].sum()} anaerobs")
   
   return X_subsampled, y_subsampled


def generate_colors_from_colormap(colormap_name, N):
   # Get the colormap
   colormap = plt.cm.get_cmap(colormap_name, N)
   
   # Generate the N colors from the colormap
   colors = [colormap(i) for i in range(N)]
   
   # Create a ListedColormap from the N colors
   listed_cmap = ListedColormap(colors)
   
   return listed_cmap, colors

def pca_run_and_plot(X_train_val, n_compon, y_train_val = None, category_names = None,  colors = None):
   scaler = MaxAbsScaler()

   # Fit and transform the data
   X_train_val = scaler.fit_transform(X_train_val)

   # Run PCA on the X-data
   pca = PCA(n_components=n_compon)
   X_train_pca = pca.fit_transform(X_train_val)
   print(f"Data after PCA reduction: {X_train_pca.shape}")

   # Find the explained variance
   explained_variance_ratio = pca.explained_variance_ratio_

   print("Explained variance ratio:", explained_variance_ratio)
   print("Total explained variance:", sum(explained_variance_ratio))

   listed_cmap = None

  # plt.figure(figsize=(4, 4))
   if y_train_val is not None:
       #Ensure 'y_train_val' has unique, valid labels
       unique_ids = np.unique(y_train_val)

       # Check if 'colors' is already a ListedColormap or needs to be generated
       if isinstance(colors, ListedColormap):
           listed_cmap = colors  # Use the passed ListedColormap directly
       else:
           # Generate colors based on the unique labels in y_train_val
           
           listed_cmap = ListedColormap(cm.nipy_spectral(np.linspace(0, 1, len(unique_ids))))
       scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_val, alpha=0.6, s = 10, label = category_names, cmap=listed_cmap)

       if category_names is not None:
           categ_name_dict = defaultdict(int)
           for i in range(len(y_train_val)):
               categ_id = y_train_val[i]
               #if categ_id not in categ_name_dict.keys():
               categ_id = int(categ_id)
               categ_name_dict[categ_id] = category_names[i]
           labels = [categ_name_dict[unique_id] for unique_id in unique_ids]       

           # Create legend handles and labels based on unique labels
           handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=listed_cmap(i / len(unique_ids)), markersize=10) for i in range(len(unique_ids))]
       

           plt.legend(handles=handles, labels=labels ,loc='upper center', title="Categories", ncol=5) #, bbox_to_anchor=(1.05, 1)
       else:
           plt.colorbar()    
   else:
       scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.6, s = 10)
          
   plt.xlabel(f"PC 1; var = {round(explained_variance_ratio[0],2)}")
   plt.ylabel(f"PC 2; var = {round(explained_variance_ratio[1],2)}")
   plt.title("PCA space")
   plt.grid(True, zorder=1)
   #plt.show()

   return listed_cmap    


false_posit_uniq = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
false_negat_uniq = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def tsne_plot(X_train, perplexity, learning_rate, random_seed, y_train = None, colors = None):
    scaler = MaxAbsScaler()

    # Fit and transform the data
    X_train_scal = scaler.fit_transform(X_train)


    # Initialize and apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, max_iter=3000, init='pca', random_state=random_seed) 

    if colors is None:
        listed_cmap = ListedColormap(cm.nipy_spectral(np.linspace(0, 1, len(np.unique(y_train)))))
        #colors = ListedColormap(["tab:green", "tab:purple"])
    else:
        listed_cmap = colors    


    X_tsne = tsne.fit_transform(X_train_scal) 

    print(f"Shape of the projected data = {X_tsne.shape}")

    # Visualize the t-SNE output
    if y_train is not None:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, alpha=0.5, s = 10, cmap=listed_cmap)
        if colors is None:
            plt.colorbar()    
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5, s = 10)
    plt.xlabel("tSNE1")
    plt.ylabel("tSNE2")
    plt.title("tSNE space")    
    plt.grid(True, zorder=1)      


def generate_tables(grouped):
    tables = []
    
    false_positive_unique = grouped['false_positive_rate'].unique()
    
    # Loop over each unique value in false_positive_rate
    for false_positive_value in false_positive_unique:
        # Filter the rows with this false_positive_rate value
        filtered_df = grouped[grouped['false_positive_rate'] == false_positive_value]
        
        # Optionally, you can add the false_positive_rate as a column in the filtered DataFrame
        filtered_df = filtered_df[['false_negative_rate', 'matching_probability', 'mean_fp_prediction', 'mean_fn_prediction']]
    
        # Append to the tables list
        tables.append(filtered_df)
    return tables    

def find_aver_accuracy(table_dict):
    average_arr = []
    for key in table_dict.keys():
        average_arr.append(table_dict[key]['matching_probability'].mean())
    return np.mean(average_arr)   

def find_average_table(csv_files_cross_valid, result_directory):
    tables_all_folds = defaultdict(list)
    tables_average_folds = defaultdict(pd.DataFrame)
    
    for csv_file in csv_files_cross_valid:
        file_path = result_directory + csv_file
        grouped = group_matching_probab(file_path)
        tables = generate_tables(grouped)
        for i in range(len(tables)):
            tables_all_folds[false_posit_uniq[i]].append(tables[i])
    
    for key in tables_all_folds.keys():
        tables = tables_all_folds[key]
        
        # Concatenate all tables into one DataFrame
        combined = pd.concat(tables)
        
        # Group by 'false_negative_rate' and calculate the mean of 'matching_probability'
        average_table = (
            combined.groupby('false_negative_rate', as_index=False)
            .agg({
                'matching_probability': 'mean',
                'mean_fp_prediction': 'mean',
                'mean_fn_prediction': 'mean'
            })
        )
       
        tables_average_folds[key] = average_table
    return tables_average_folds   

def group_matching_probab(file_path):
    df = pd.read_csv(file_path, delimiter='\t')
    # Assuming df is your DataFrame
    df['prediction_correct'] = df['prediction'] == df['y_actual']

    df['fp_prediction'] = (df['prediction'] == 1) & (df['y_actual'] == 0)
    df['fn_prediction'] = (df['prediction'] == 0) & (df['y_actual'] == 1)

  #  print(df)
    
    grouped = (
        df.groupby(['false_negative_rate', 'false_positive_rate'])
        .agg({
            'prediction_correct': 'mean',
            'fp_prediction': 'mean',
            'fn_prediction': 'mean'
        })
        .reset_index()
    )
    
    # Rename columns for clarity if needed
    grouped.rename(
        columns={
            'prediction_correct': 'matching_probability',
            'fp_prediction': 'mean_fp_prediction',
            'fn_prediction': 'mean_fn_prediction'
        },
        inplace=True
    )
    
    return grouped

def find_accuracies(num_ind_points, result_directory):
    # List of all csv files with cross_validation results
    csv_files_cross_valid = [f for f in os.listdir(result_directory) if f.endswith('.csv') and "cross_valid" in f and f"indPoints_{num_ind_points}" in f]
    
    csv_files_holdout_test = [f for f in os.listdir(result_directory) if f.endswith('.csv') and "holdout_test" in f and f"indPoints_{num_ind_points}" in f]
    

    grouped = group_matching_probab(result_directory+csv_files_holdout_test[0])
    holdout_test_accur_aver = grouped['matching_probability'].mean()

    print(f"\nHold-out (test) dataset results for {num_ind_points} inducing points:")
    print(f"Average accuracy: {round(grouped['matching_probability'].mean(),3)};")
    print(f"Average false_positive predictions: {round(grouped['mean_fp_prediction'].mean(),3)};")
    print(f"Average false_negative predictions: {round(grouped['mean_fn_prediction'].mean(),3)}")
    
    
    tables_average_folds = find_average_table(csv_files_cross_valid, result_directory)
    cross_valid_aver = find_aver_accuracy(tables_average_folds)
    
    print(f"\nCross-validation accuracy = {round(cross_valid_aver,3)} for {num_ind_points} inducing points")
    return tables_average_folds

def plot_results(column_name, num_ind_points, fp_to_plot, tables_average_folds):
    plt.figure(figsize=(6, 4))
    idx = 0
    for false_posit in tables_average_folds.keys():
        false_positive_value = false_posit_uniq[idx]
        if false_positive_value in fp_to_plot:
            table = tables_average_folds[false_posit]
            plt.scatter(table['false_negative_rate'], table[column_name])
            plt.plot(table['false_negative_rate'], table[column_name], label=f'extra genes rate = {false_positive_value}')
        idx += 1    
    
  #  plt.ylim([0.83, 1])
    plt.xlabel('gene removal rate')
    plt.ylim([0.85,1])
    plt.ylabel('accuracy')
    plt.legend()

    plt.title(f"{column_name} for SetTransformer with {num_ind_points} inducing points")
    
    plt.grid(True, zorder=1)
    plt.show()  

def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBRegressor(
    n_jobs=-1,                # Use all CPU cores
    tree_method="hist",   # Use "hist" for CPU, "gpu_hist" for GPU
    objective="reg:squarederror",  # Default loss function for regression
    )

    # Define cross-validation (e.g., 5-fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    y_true_list = []
    y_pred_list = []

    for train_idx, test_idx in kf.split(X_train):
        X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
        y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]

        model.fit(X_fold_train, y_fold_train)
        y_pred_fold = model.predict(X_fold_test)

        y_true_list.append(y_fold_test)
        y_pred_list.append(y_pred_fold)

    # Convert lists to arrays
    y_true_cv = np.concatenate(y_true_list)
    y_pred_cv = np.concatenate(y_pred_list)   

    model.fit(X_train.cpu(), y_train.cpu().numpy())

    # Make predictions
    y_pred_test = model.predict(X_test.cpu())

    return  y_true_cv, y_pred_cv, y_pred_test

def xgboost_accuracy_contin(X_train, X_test, y_train, y_test, sorted_cog_idx, feat_step, feat_removal = False):
    rmse_test_arr = []
    r2_test_arr = []
    rmse_cv_arr = []
    r2_cv_arr = []
    
    num_feat = range(1,len(sorted_cog_idx),feat_step)
    num_feat_plot = []
    for N in num_feat:
        if feat_removal == False:
            select_feat = list(sorted_cog_idx[:N])
        else:
            select_feat = list(sorted_cog_idx[N:])
        num_feat_plot.append(N)#len(select_feat))    
        X_train_select_feat = X_train[:, select_feat]
        X_test_select_feat = X_test[:, select_feat]
        y_true_cv, y_pred_cv, y_pred_test  = train_xgboost(X_train_select_feat, y_train, X_test_select_feat, y_test)
        
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        rmse_test_arr.append(rmse_test)
        r2_test = r2_score(y_test, y_pred_test)
        r2_test_arr.append(r2_test)
        rmse_cv = np.sqrt(mean_squared_error(y_true_cv, y_pred_cv))
        rmse_cv_arr.append(rmse_cv)
        r2_cv = r2_score(y_true_cv, y_pred_cv)
        r2_cv_arr.append(r2_cv)
    return rmse_test_arr, r2_test_arr, rmse_cv_arr, r2_cv_arr, num_feat_plot 