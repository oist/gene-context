import numpy as np
import polars as pl
import logging 
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import random 
from sklearn.preprocessing import MaxAbsScaler

from matplotlib.lines import Line2D


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

def read_xy_data(data_filename, y_filename):

    gtdb = pl.concat([
        pl.read_csv('data_aerob/bac120_metadata_r202.tsv', separator="\t"),
        pl.read_csv('data_aerob/ar122_metadata_r202.tsv', separator="\t")
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

    print(f"Counts of each class in training/test data: {d3.group_by(target_column).agg(pl.len())}")
    
    d_gtdb = d3.to_pandas()
    X = d3.select(pl.exclude(['accession',target_column,'phylum','class','order','family','genus','false_negative_rate','false_positive_rate'])).to_pandas()
    
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

   d_gtdb = d3.to_pandas()
   X = d3.select(pl.exclude([target_column])).to_pandas() #'phylum','class','order','family','genus'
   

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

def pca_run_and_plot(X_train_val, y_train_val, category_names, n_compon, colors):

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



   # Ensure 'y_train_val' has unique, valid labels
   unique_ids = np.unique(y_train_val)

   # Check if 'colors' is already a ListedColormap or needs to be generated
   if isinstance(colors, ListedColormap):
       listed_cmap = colors  # Use the passed ListedColormap directly
   else:
       # Generate colors based on the unique labels in y_train_val
       from matplotlib import cm
       listed_cmap = ListedColormap(cm.nipy_spectral(np.linspace(0, 1, len(unique_ids))))
     #  listed_cmap = ListedColormap(cm.nipy_spectral(np.linspace(0, 1, len(np.unique(y_train_val)+1))))
      # listed_cmap = plt.get_cmap(colors, len(np.unique(y_train_val)))#listed_cmap, _ = generate_colors_from_colormap(colors, N=len(np.unique(y_train_val)))


   plt.figure(figsize=(8, 6))
   scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_val, alpha=1, s = 10, label = category_names, cmap=listed_cmap)
   plt.xlabel(f"PC 1; var = {round(explained_variance_ratio[0],2)}")
   plt.ylabel(f"PC 2; var = {round(explained_variance_ratio[1],2)}")
   # handles, labels = scatter.legend_elements()  # Get handles and labels from scatter plot



   categ_name_dict = defaultdict(int)
   for i in range(len(y_train_val)):
       categ_id = y_train_val[i]
       #if categ_id not in categ_name_dict.keys():
       categ_id = int(categ_id)
       categ_name_dict[categ_id] = category_names[i]




   # Map the unique labels to their corresponding category names
  # label_to_category = {label: category_names[label] for label in unique_labels}

    # Create legend handles and labels based on unique labels
   handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=listed_cmap(i / len(unique_ids)), markersize=10) for i in range(len(unique_ids))]
   labels = [categ_name_dict[unique_id] for unique_id in unique_ids]

   plt.legend(handles=handles, labels=labels ,loc='lower left', bbox_to_anchor=(1.05, 1), title="Categories", ncol=5)
   
   return listed_cmap    