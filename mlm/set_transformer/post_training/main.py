import pandas as pd
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import  DataLoader

from collections import OrderedDict

from set_transformer.utils.architecture import GenomeSetTransformer
from data_processing_utils.data_processing_functions import GenomeDataset, collate_genomes
from data_processing_utils.data_processing_functions import load_list_from_txt

"""
How to run this script?

cd ~\gene-context\mlm
python -m set_transformer.post_training.main --train_feather_path .\data\train_test_splits\cog_train_family_tax_level.feather --test_feather_path 
.\data\train_test_splits\cog_test_family_tax_level.feather
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_per_sample_extended(set_transformer, dataloader, device,global_vocab,
                                 threshold=0.5, apply_mlm=True, block_frac=0.15, dummy_mode=False,
                                 label_string="default"):
    """
    Evaluates the SetTransformer model on the dataloader and computes per-sample metrics,
    while also accumulating TP, TN, FP, FN counts per COG (token) over all samples.
    
    For each sample:
      1. Compute SetTransformer probabilities and a discrete assignment by thresholding at 0.5.
      2. Reconstruct the observed (noisy) input from the tokens.
      3. Compute pre-MLM metrics using the discrete assignment.
      4. Accumulate per-COG counts from these predictions.
      5. Optionally (if apply_mlm is True), refine the combined input:
           - Form the combined base by OR-ing the observed input and the discrete assignment.
           - Partition the entire vocabulary [0, V) into blocks of size ceil(V*block_frac) (at least 1 token per block).
           - For each block, mask the tokens (set to 2), run binaryMLM, and replace the tokens in that block
             with the binary MLM predictions.
           - Compute MLM metrics from the final refined prediction and also accumulate per-COG counts.
      6. Append per-sample metrics to a list.
    
    After processing all samples, write out the per-COG aggregated counts to
         "COG_metrics_"+label_string+".csv".
    
    Returns:
      A list of dictionaries (one per sample) with the computed metrics.
    """
    sample_metrics = []
    global_sample_index = 0

    # These will accumulate counts per COG. They will be initialized on the first batch.
    global_pre_TP = None
    global_pre_TN = None
    global_pre_FP = None
    global_pre_FN = None
    if apply_mlm:
        global_mlm_TP = None
        global_mlm_TN = None
        global_mlm_FP = None
        global_mlm_FN = None

    for tokens, mask, targets in tqdm(dataloader, desc="Evaluating (extended)", leave=False): # mask = padding mask
        tokens = tokens.to(device)
       # print(f"targets = {targets}")
        mask = mask.to(device)
        targets = targets.to(device)
        
        # Find predictions of the trained model on test dataset
        with torch.no_grad():
            probs = set_transformer(tokens, mask)  # shape: (B, vocab_size)
        st_probs_np = probs.cpu().detach().numpy()  # shape: (B, V)
        # Pre-MLM discrete assignment: threshold at 0.5.
        discrete_assignment = (st_probs_np >= threshold).astype(int)
        
        targets_np = targets.cpu().detach().numpy().astype(int)
        tokens_np = tokens.cpu().detach().numpy()
        mask_np = mask.cpu().detach().numpy()
        batch_size = targets_np.shape[0]
        V = targets_np.shape[1] # vocab size
        
        # Initialize per-COG accumulators on the first batch.
        if global_pre_TP is None:
            global_pre_TP = np.zeros(V, dtype=int)
            global_pre_TN = np.zeros(V, dtype=int)
            global_pre_FP = np.zeros(V, dtype=int)
            global_pre_FN = np.zeros(V, dtype=int)
            # if apply_mlm:
            #     global_mlm_TP = np.zeros(V, dtype=int)
            #     global_mlm_TN = np.zeros(V, dtype=int)
            #     global_mlm_FP = np.zeros(V, dtype=int)
            #     global_mlm_FN = np.zeros(V, dtype=int)
        
        for i in range(batch_size):
            # Reconstruct observed (noisy) input.
            observed = np.zeros(V, dtype=int)
            valid_tokens = tokens_np[i][mask_np[i] == False]
            for token in valid_tokens:
                cog_id = int(token[0])
                if cog_id < V:
                    observed[cog_id] = 1
            
            # Pre-MLM predictions from SetTransformer.
            pre_preds = discrete_assignment[i]
            
            # Compare target (true) values vs the model predictions on test 
            global_pre_TP += ((targets_np[i] == 1) & (pre_preds == 1)).astype(int) # element-wise sum
            global_pre_TN += ((targets_np[i] == 0) & (pre_preds == 0)).astype(int)
            global_pre_FP += ((targets_np[i] == 0) & (pre_preds == 1)).astype(int)
            global_pre_FN += ((targets_np[i] == 1) & (pre_preds == 0)).astype(int)

            
            # Compute "noisy" metrics: compare target (true) values vs noisy target
            TP_noisy = np.sum((targets_np[i] == 1) & (observed == 1))
            TN_noisy = np.sum((targets_np[i] == 0) & (observed == 0))
            FP_noisy = np.sum((targets_np[i] == 0) & (observed == 1))
            FN_noisy = np.sum((targets_np[i] == 1) & (observed == 0))
            noisy_acc = (TP_noisy + TN_noisy) / V
            prec_noisy = TP_noisy / (TP_noisy + FP_noisy) if (TP_noisy + FP_noisy) > 0 else 0.0
            rec_noisy = TP_noisy / (TP_noisy + FN_noisy) if (TP_noisy + FN_noisy) > 0 else 0.0
            f1_noisy =  2 * prec_noisy * rec_noisy / (prec_noisy + rec_noisy) if (prec_noisy + rec_noisy) > 0 else 0.0

            # Compute pre-MLM metrics.
            TP = np.sum((targets_np[i] == 1) & (pre_preds == 1))
            TN = np.sum((targets_np[i] == 0) & (pre_preds == 0))
            FP = np.sum((targets_np[i] == 0) & (pre_preds == 1))
            FN = np.sum((targets_np[i] == 1) & (pre_preds == 0))
            acc = (TP + TN) / V
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            true_size = np.sum(targets_np[i])
            pre_pred_size = np.sum(pre_preds)
            genome_diff = np.abs(pre_pred_size / true_size) if true_size > 0 else np.nan


            fp_noise = (observed == 1) & (targets_np[i] == 0)
            fn_noise = (observed == 0) & (targets_np[i] == 1)
            fp_removed = (np.sum((pre_preds == 0) & fp_noise) / np.sum(fp_noise)
                          if np.sum(fp_noise) > 0 else np.nan)
            fn_recovered = (np.sum((pre_preds == 1) & fn_noise) / np.sum(fn_noise)
                            if np.sum(fn_noise) > 0 else np.nan)
            

            # Metrics for each sample
            sample_dict = {
                "sample_id": global_sample_index,
                "noisy_accuracy": noisy_acc,# (np.sum((targets_np[i] == 1) & (observed == 1)) +np.sum((targets_np[i] == 0) & (observed == 0))) / V,
                "noisy_precision": prec_noisy,#TP_noisy / (TP_noisy + FP_noisy) if (TP_noisy + FP_noisy) > 0 else 0.0,
                "noisy_recall": rec_noisy,#TP_noisy / (TP_noisy + FN_noisy) if (TP_noisy + FN_noisy) > 0 else 0.0,
                "noisy_f1": f1_noisy,#2 * prec_noisy * rec_noisy / (prec_noisy + rec_noisy) if (prec_noisy + rec_noisy) > 0 else 0.0,
                "noisy_genome_diff": np.abs(np.sum(observed) / true_size) if true_size > 0 else np.nan,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "genome_size_diff": genome_diff,
                "fp_removed": fp_removed,
                "fn_recovered": fn_recovered
            }
            
#             # Optional: MLM refinement step.
#             if apply_mlm:
#                 true_size = np.sum(targets_np[i])
#                 # Combined base: OR of observed and pre_preds.
#                 combined_base = np.maximum(observed, pre_preds)
# #                combined_base = np.maximum(pre_preds, pre_preds)
#                 current_output = combined_base.copy()
#                 # Partition the vocabulary [0, V) into blocks.
#                 block_size = max(1, int(np.ceil(V * block_frac)))
#                 all_indices = np.arange(V)
#                 # Process each block sequentially.
#                 for start in range(0, V, block_size):
#                     current_block = all_indices[start:start+block_size]
#                     modified_input = current_output.copy()
#                     modified_input[current_block] = 2  # mask these tokens.
#                     if dummy_mode:
#                         refined_preds = current_output.copy()
#                     else:
#                         discrete_tensor = torch.tensor(modified_input).unsqueeze(0).to(device)
#                         with torch.no_grad():
#                             logits, _ = binary_mlm(discrete_tensor)
#                             mlm_probs = torch.softmax(logits, dim=-1)[:, :, 1]
#                         refined_preds = (mlm_probs.cpu().detach().numpy() >= threshold).astype(int).squeeze(0)
#                     # Replace the tokens in the current block with the MLM predictions.
#                     current_output[current_block] = refined_preds[current_block]
                
#                 final_mlm_preds = current_output
#                 # Accumulate per-COG counts for MLM predictions.

#                 global_mlm_TP += ((targets_np[i] == 1) & (final_mlm_preds == 1)).astype(int)
#                 global_mlm_TN += ((targets_np[i] == 0) & (final_mlm_preds == 0)).astype(int)
#                 global_mlm_FP += ((targets_np[i] == 0) & (final_mlm_preds == 1)).astype(int)
#                 global_mlm_FN += ((targets_np[i] == 1) & (final_mlm_preds == 0)).astype(int)
                    
#                 TP_mlm = np.sum((targets_np[i] == 1) & (final_mlm_preds == 1))
#                 TN_mlm = np.sum((targets_np[i] == 0) & (final_mlm_preds == 0))
#                 FP_mlm = np.sum((targets_np[i] == 0) & (final_mlm_preds == 1))
#                 FN_mlm = np.sum((targets_np[i] == 1) & (final_mlm_preds == 0))
#                 MLM_acc = (TP_mlm + TN_mlm) / V
#                 MLM_prec = TP_mlm / (TP_mlm + FP_mlm) if (TP_mlm + FP_mlm) > 0 else 0.0
#                 MLM_rec = TP_mlm / (TP_mlm + FN_mlm) if (TP_mlm + FN_mlm) > 0 else 0.0
#                 MLM_f1 = 2 * MLM_prec * MLM_rec / (MLM_prec + MLM_rec) if (MLM_prec + MLM_rec) > 0 else 0.0                
#                 MLM_pred_size = np.sum(final_mlm_preds)
#                 MLM_genome_diff = MLM_pred_size / true_size if true_size > 0 else np.nan
#                 #print(MLM_pred_size,true_size,MLM_genome_diff)
#                 fp_removed_mlm = (np.sum((final_mlm_preds == 0) & fp_noise) / np.sum(fp_noise)
#                                   if np.sum(fp_noise) > 0 else np.nan)
#                 fn_recovered_mlm = (np.sum((final_mlm_preds == 1) & fn_noise) / np.sum(fn_noise)
#                                     if np.sum(fn_noise) > 0 else np.nan)
#                 sample_dict.update({
#                     "MLM_accuracy": MLM_acc,
#                     "MLM_precision": MLM_prec,
#                     "MLM_recall": MLM_rec,
#                     "MLM_f1": MLM_f1,
#                     "MLM_genome_diff": MLM_genome_diff,
#                     "MLM_fp_removed": fp_removed_mlm,
#                     "MLM_fn_recovered": fn_recovered_mlm
#                 })
            
            sample_metrics.append(sample_dict)
            global_sample_index += 1

    # After processing all samples, write per-COG metrics.
    # Create a dictionary with rows per COG index.
    per_cog_data = {
        "COG": global_vocab,
        "pre_TP": global_pre_TP,
        "pre_TN": global_pre_TN,
        "pre_FP": global_pre_FP,
        "pre_FN": global_pre_FN
    }
    # if apply_mlm:
    #     per_cog_data.update({
    #         "MLM_TP": global_mlm_TP,
    #         "MLM_TN": global_mlm_TN,
    #         "MLM_FP": global_mlm_FP,
    #         "MLM_FN": global_mlm_FN
    #     })
    df_cog_metrics = pd.DataFrame(per_cog_data)
    df_cog_metrics.to_csv("COG_metrics_" + label_string + ".csv", index=False)
    
    return sample_metrics




def process_args():
    parser = argparse.ArgumentParser(description="Process input arguments for model training.")
    parser.add_argument("--train_feather_path", type=str, help="Path to the train feather data file.")
    parser.add_argument("--test_feather_path", type=str, help="Path to the test feather data file.")
    parser.add_argument("--global_vocab_path", default = "data/train_test_splits/global_vocab.txt", type=str, help="Path to the global vocabulary data file.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for training")
    parser.add_argument("--embedd_dim", type=int, default=256, help="Embedding dimencionality")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--num_sab", type=int, default=2, help="Number of SAB layers")  
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")   
    args = parser.parse_args()
    args_dict = {"train_feather_path": args.train_feather_path, "test_feather_path": args.test_feather_path, "global_vocab_path": args.global_vocab_path, 
                 "batch_size": args.batch_size, "embedd_dim": args.embedd_dim, "num_heads": args.num_heads, "num_sab": args.num_sab, "num_epochs": args.num_epochs}
    return args_dict




def main(args_dict):

    train_df = pd.read_feather(args_dict["train_feather_path"])
    val_df = pd.read_feather(args_dict["test_feather_path"])
    global_vocab = load_list_from_txt(args_dict["global_vocab_path"])

        # Load pretrained SetTransformer
    set_transformer = GenomeSetTransformer(vocab_size=len(global_vocab), d_model=256,
                                            num_heads=4, num_sab=2, dropout=0.1)
    state_dict = torch.load('model_checkpoint_full.pth', weights_only=True) #TODO change
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
       # print(f"k = {k}")
        name = k[7:] if k.startswith("module.") else k
      #  print(f"name = {name}")
        new_state_dict[name] = v
    set_transformer.load_state_dict(new_state_dict)
    #set_transformer.load_state_dict(torch.load('model_checkpoint_full.pth'))
    set_transformer = set_transformer.to(device)
    return val_df, global_vocab, set_transformer


def training_summary(val_df, global_vocab, pre_train_model):

    # Define FN rates to test and fixed FP rate
    fn_rates = [0.0, 0.1]#, 0.25, 0.5, 0.75, 0.85,0.95,1.0]
    fp_rate = 0.0

    all_sample_records = []  # Will collect records from both stages
    # We'll print summary statistics for Pre-MLM separately for "noisy" (do nothing) metrics and processed ones.

    for fn in fn_rates:
        print(f"\nEvaluating for FN rate = {fn} and FP rate = {fp_rate}...")
        val_dataset = GenomeDataset(val_df, global_vocab=global_vocab, 
                                    false_negative_rate=fn, false_positive_rate=fp_rate,
                                    count_noise_std=0.0, random_state=42)
        val_dataloader = DataLoader(val_dataset, batch_size=16,
                                    collate_fn=lambda batch: collate_genomes(batch, pad_idx=len(global_vocab)))

        # Evaluate Pre-MLM metrics (using SetTransformer output)
        #pre_metrics = validate_per_sample_extended(set_transformer, val_dataloader, device,
        #                                             threshold=0.5, lower_thresh=-0.2, upper_thresh=0.8)
        pre_metrics = validate_per_sample_extended(pre_train_model, val_dataloader, device,global_vocab,label_string="FN"+repr(fn)+"_FP"+repr(fp_rate))

        for rec in pre_metrics:
            rec["FN_rate"] = fn
            rec["Stage"] = "Pre-MLM"
        # Print summary statistics for Pre-MLM stage (both "noisy" and processed metrics)
        print(f"\nSummary for FN rate {fn}, Stage Pre-MLM (Noisy Input):")
        for metric in ["noisy_accuracy", "noisy_precision", "noisy_recall", "noisy_f1", "noisy_genome_diff"]:
            vals = [r[metric] for r in pre_metrics if r.get(metric) is not None and not np.isnan(r[metric])]
            if len(vals) > 0:
                mean_val = np.mean(vals)
                q25 = np.nanpercentile(vals, 25)
                median_val = np.nanpercentile(vals, 50)
                q75 = np.nanpercentile(vals, 75)
                print(f"  {metric.title():<20}: Mean={mean_val:.4f}, 25th={q25:.4f}, Median={median_val:.4f}, 75th={q75:.4f}")
            else:
                print(f"  {metric.title():<20}: No valid values")

        print(f"\nSummary for FN rate {fn}, Stage Post-SetTransformer (Processed):")
        for metric in ["accuracy", "precision", "recall", "f1", "genome_size_diff", "fp_removed", "fn_recovered"]:
            vals = [r[metric] for r in pre_metrics if r.get(metric) is not None and not np.isnan(r[metric])]
            if len(vals) > 0:
                mean_val = np.mean(vals)
                q25 = np.nanpercentile(vals, 25)
                median_val = np.nanpercentile(vals, 50)
                q75 = np.nanpercentile(vals, 75)
                print(f"  {metric.title():<20}: Mean={mean_val:.4f}, 25th={q25:.4f}, Median={median_val:.4f}, 75th={q75:.4f}")
            else:
                print(f"  {metric.title():<20}: No valid values")

        print(f"\nSummary for FN rate {fn}, Stage Post-MLM (Processed):")
        for metric in ["MLM_accuracy", "MLM_precision", "MLM_recall", "MLM_f1", "MLM_genome_diff", "MLM_fp_removed", "MLM_fn_recovered"]:
            vals = [r[metric] for r in pre_metrics if r.get(metric) is not None and not np.isnan(r[metric])]
            if len(vals) > 0:
                mean_val = np.mean(vals)
                q25 = np.nanpercentile(vals, 25)
                median_val = np.nanpercentile(vals, 50)
                q75 = np.nanpercentile(vals, 75)
                print(f"  {metric.title():<20}: Mean={mean_val:.4f}, 25th={q25:.4f}, Median={median_val:.4f}, 75th={q75:.4f}")
            else:
                print(f"  {metric.title():<20}: No valid values")


        print("-" * 60)
        all_sample_records.extend(pre_metrics)

    # Combine all records into a DataFrame and save to CSV.
    df_metrics = pd.DataFrame(all_sample_records)
    df_metrics.to_csv("per_sample_combined_metrics_FP"+repr(fp_rate)+".csv", index=False)
    print("\nSaved per-sample combined metrics to per_sample_combined_metrics_FP"+repr(fp_rate)+".csv")


if __name__=='__main__':
    # 1. Read the input args
    args_dict = process_args()
    val_df, global_vocab, set_transformer = main(args_dict)
    training_summary(val_df, global_vocab, set_transformer)