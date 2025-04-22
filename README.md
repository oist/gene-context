
This repository contains the following directories.

# Genome Denoising
TODO

# Phenotype Prediction

This directory contains implementations of the phenotype prediction and feature selection pipelines. The directory structure is the following:
- data_preparation: contains pipeline and data for test/train data split,
- data_[phenotype]: input/output data directory for [phenotype],
- jupyter_notebooks: contains notebooks with the scripts for phenotype prediction and feature selection for each phenotype.

At the moment, there are three phenotypes:
- aerobicity,
- OGT,
- mono/didermy.

First, the input test/train datasets should be generated for the chosen phenotype. The split is done at a chosen taxonomy level (i.e. samples from the same taxonomy group are not split between train and test). To generate the input splits, run the following (see `taxa_level_split.py` description for more details)


```bash 
cd ~/gene-context/phenotype_prediction
python3 -m data_preparation.taxa_level_split \ --tax_level [tax_level] \ --input_annotation_csv [input_annotation_csv] \ --input_data_csv [input_data_csv] \ --output_dir [output_dir]
```

Please, note that `ar122_metadata_r202.tsv` and `bac120_metadata_r202.tsv` are required to be stored in `gene-context/phenotype_prediction/data_preparation/gtdb_files` in order to run the above command.

After the inputs are generated, a notebook for the corresponding [phenotype] can be run.


