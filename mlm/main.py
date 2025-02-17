import os
import torch
from utils.utils import process_aerob_dataset
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

if __name__ == '__main__':

    # 1. Process input parameters
   # Parameters = process_args()

   data_filename_train = "data_aerob/all_gene_annotations.added_incompleteness_and_contamination.training.tsv"
   y_filename = "data_aerob/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv"

    # 2. Process train and test datasets
   X_train, X_train_column_names, y_train, d_gtdb_train = process_aerob_dataset(data_filename_train, y_filename, device, remove_noise=True)

  # print(X_train.detach().cpu().numpy())

   X_train = X_train.detach().cpu().numpy()

   cog_tokens = []
   for sample in X_train:
      #cog_tokens.append([])
      sequence = []
      for i in range(len(sample)):
        if sample[i] > 0:
            sequence.append(i)#X_train_column_names[i])
      cog_tokens.append(torch.tensor(sequence))
  # print(cog_tokens[0])   


   padded_sequences = pad_sequence(cog_tokens, batch_first=True, padding_value=0)
   #print(padded_sequences)     


   attention_masks = (padded_sequences != 0).long()
   print(attention_masks)



   