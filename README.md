
This repository contains the implementation of the Set Transformer model for predicting phenotypes from gene count data.  

At the moment, two phenotypes are supported: `aerob` and `ogt`.
The `set_transformer/main.py` script calls cross-validation, training and test functions, which are implemented in `set_transformer/train_test_func.py`. 

The input train and test data are the following:

1. For phenotype  = `aerob`:
    - data_aerob/all_gene_annotations.added_incompleteness_and_contamination.subsampled.training.tsv
    - data_aerob/all_gene_annotations.added_incompleteness_and_contamination.subsampled.testing.tsv
    - data_aerob/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv
    - data_aerob/bac120_metadata_r202.tsv
    - data_aerob/ar122_metadata_r202.tsv

2. For phenotype  = 'ogt':
    - data_ogt/kegg.csv
    - data_ogt/ogt_splits.csv

The results of the training and cross-validation are saved to `results/SetTransformer/{phenotype}`. 

The scripts generate and save:
- prediction_probabilities_cross_valid_fold_{fold_id}_SetTransformer_indPoints_{num_ind_points}.csv
- prediction_probabilities_holdout_test_SetTransformer_indPoints_{num_ind_points}.csv
- trained_model_SetTransformer_indPoints_{num_ind_points}_D_{D_val}_K_{K_val}_dim_output_{dim_output_val}.model

The input data files can be found  [here](https://office365oist-my.sharepoint.com/:f:/g/personal/a-koldaeva_oist_jp/Es-FfClDP6JFgNvpg0ga1aEB_v3foyEEJQ2oED3Ic-dTrw?email=GERGELY.SZOLLOSI%40OIST.JP&e=y5R136) (request access if needed).



The below command will run cross-validation, training, and testing of a Set Transformer model for the specified parameters.    
```bash
    python3 -m set_transformer.main --num_inds {num_inds_val} --learning_rate {learning_rate_val} --num_epochs {num_epochs_val} --batch_size {batch_size_val} --phenotype [ogt/aerob]
```
E.g.
```bash
    python3 -m set_transformer.main --num_inds 20 --learning_rate 0.0001 --num_epochs 10 --batch_size 32 --phenotype ogt
```
Or

```bash
    python3 -m set_transformer.main --num_inds 20 --learning_rate 0.0001 --num_epochs 10 --batch_size 32 --phenotype aerob
```
