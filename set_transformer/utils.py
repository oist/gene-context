import numpy as np
import polars as pl
import logging 
import pandas as pd
import matplotlib.pyplot as plt

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