To run the Set Transformer model: 

python3 -m set_transformer.main --num_inds 2 --learning_rate 0.0001 --num_epochs 3 --batch_size 32 --data_filename_train 'data/all_gene_annotations.added_incompleteness_and_contamination.subsampled.training.tsv' --data_filename_test data/all_gene_annotations.added_incompleteness_and_contamination.subsampled.testing.tsv --y_filename data/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv
