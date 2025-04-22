import os
import sys
import requests
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import  DataLoader

parent_directory = os.path.abspath(os.path.join('..'))
grandparent_directory = os.path.abspath(os.path.join(parent_directory, '..'))

sys.path.append(parent_directory)
sys.path.append(grandparent_directory)

from set_transformer.utils.architecture import GenomeSetTransformer
from data_processing_utils.data_processing_functions import GenomeDataset, collate_genomes

def generate_noisy_dataset(df, global_vocab, batch_size, pad_idx, fn_rate, fp_rate, count_noise_std=0, random_state=42):
     dataset = GenomeDataset(df, global_vocab,
                               false_negative_rate=fn_rate, false_positive_rate=fp_rate,
                               count_noise_std=count_noise_std, random_state=random_state)
     dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_genomes(batch, pad_idx=pad_idx))    
     return dataset_loader

def load_model(path, global_vocab, device, d_model,num_heads,num_sab):
# Load pretrained SetTransformer
    model = GenomeSetTransformer(vocab_size=len(global_vocab), d_model=d_model,
                                                num_heads=num_heads, num_sab=num_sab, dropout=0.1)

    # Load SetTransformer weights using map_location=device
    model_state = torch.load(path, map_location=device,weights_only=True)   #COG_high3_256_4_8_BCE_40.pth
    # Remove the "module." prefix if trained with DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in model_state.items():
        new_key = key.replace("module.", "")  # Remove "module." prefix
        new_state_dict[new_key] = value

    # Load into model
    model.load_state_dict(new_state_dict)
    model.to(device)
    return model


def create_cogs_with_functions_df():

    # URL for the COG definitions file
    url = "https://ftp.ncbi.nih.gov/pub/COG/COG2020/data/cog-20.def.tab"
    response = requests.get(url)
    if response.status_code == 200:
        with open("cog-20.def.tab", "wb") as f:
            f.write(response.content)
        print("Downloaded cog-20.def.tab successfully.")
    else:
        raise Exception(f"Error downloading file: HTTP {response.status_code}")
    
    # Read the tab-delimited file (it has no header)
    df_cog = pd.read_csv("cog-20.def.tab", sep="\t", header=None, engine="python", encoding="latin1")
    # Assign column names; based on the provided sample:
    df_cog = pd.read_csv("cog-20.def.tab", sep="\t", header=None, engine="python", encoding="latin1")
    df_cog.columns = ["COG_ID", "Category", "Description", "Gene_Symbol", "Function", "Extra", "PDB_ID"]
    # ---------------------------
    # 2. Map to Canonical Metacategories
    # ---------------------------
    # Define canonical mapping for each letter.
    meta_map = {
        "S": "Poorly Characterized", "R": "Poorly Characterized",
        "Q": "Metabolism", "P": "Metabolism", "I": "Metabolism",
        "H": "Metabolism", "F": "Metabolism", "E": "Metabolism",
        "G": "Metabolism", "C": "Metabolism",
        "O": "Cellular Processes & Signaling", "U": "Cellular Processes & Signaling", "W": "Cellular Processes & Signaling",
        "N": "Cellular Processes & Signaling", "M": "Cellular Processes & Signaling", "T": "Cellular Processes & Signaling",
        "V": "Cellular Processes & Signaling", "D": "Cellular Processes & Signaling",
        "B": "Information Storage & Processing", "L": "Information Storage & Processing", "K": "Information Storage & Processing",
        "A": "Information Storage & Processing", "J": "Information Storage & Processing"
    }

    def assign_metacategory(cat_str, meta_map):
        """
        For a given category string (which may contain multiple letters),
        assign a canonical metacategory using majority rule.
        """
        counts = {}
        for letter in str(cat_str):
            if letter in meta_map:
                mc = meta_map[letter]
                counts[mc] = counts.get(mc, 0) + 1
        if counts:
            return max(counts, key=counts.get)
        else:
            return None

    # Process all rows: keep original Category and compute Meta_Category.
    functional_df = df_cog[["COG_ID", "Category"]].copy()
    functional_df["Meta_Category"] = functional_df["Category"].apply(lambda x: assign_metacategory(x, meta_map))
    functional_df = functional_df.dropna(subset=["Meta_Category"])
    functional_df = functional_df.sort_values("COG_ID").reset_index(drop=True)
    functional_df["COG_Index"] = functional_df.index
    print("First few rows of functional_df with Meta_Category:")
    return functional_df

def target_and_oredict_for_model(train_loader, model, DEVICE):
    # Storage for predictions and true labels
    all_preds, all_targets = [], []

    with torch.no_grad():
        for tokens, mask, targets in tqdm(train_loader, desc="Generating Predictions"):
            tokens, mask, targets = tokens.to(DEVICE), mask.to(DEVICE), targets.to(DEVICE)
            preds = (model(tokens, mask) > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all results across batches
    X_predict = torch.cat(all_preds, dim=0).numpy()
    X_true = torch.cat(all_targets, dim=0).numpy()

    return X_true, X_predict


def compute_metrics(tp, fp, fn, tn):
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return accuracy, precision, recall, f1


def accuracy_stats_per_cog_per_sample(X_true, X_predict, cog_names):
    # Initialize stats
    num_cogs = X_true.shape[1]
    num_samples = X_true.shape[0]

    cog_stats = {cog: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for cog in cog_names}
    sample_stats = []
    overall_stats = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    # Compute FP, FN, TP, TN per COG and overall
    for i in range(num_samples):
        sample_tp = (X_true[i] == 1) & (X_predict[i] == 1)
        sample_fp = (X_true[i] == 0) & (X_predict[i] == 1)
        sample_fn = (X_true[i] == 1) & (X_predict[i] == 0)
        sample_tn = (X_true[i] == 0) & (X_predict[i] == 0)
        
        tp, fp, fn, tn = sample_tp.sum(), sample_fp.sum(), sample_fn.sum(), sample_tn.sum()
        sample_stats.append(compute_metrics(tp, fp, fn, tn))
        
        overall_stats["TP"] += tp
        overall_stats["FP"] += fp
        overall_stats["FN"] += fn
        overall_stats["TN"] += tn
        
        for j in range(num_cogs):
            if sample_tp[j]: cog_stats[cog_names[j]]["TP"] += 1
            if sample_fp[j]: cog_stats[cog_names[j]]["FP"] += 1
            if sample_fn[j]: cog_stats[cog_names[j]]["FN"] += 1
            if sample_tn[j]: cog_stats[cog_names[j]]["TN"] += 1

    # Compute final metrics
    cog_metrics = {cog: compute_metrics(stats["TP"], stats["FP"], stats["FN"], stats["TN"]) for cog, stats in cog_stats.items()}
    overall_metrics = compute_metrics(
        overall_stats["TP"], 
        overall_stats["FP"], 
        overall_stats["FN"], 
        overall_stats["TN"]
    )
    # Print results
    print(f"Overall Metrics: accuracy = {round(overall_metrics[0],2)}, precision = {round(overall_metrics[1],2)}, recall = {round(overall_metrics[2],2)}, f1 = {round(overall_metrics[3],2)}")  #accuracy, precision, recall, f1

    # print("Per-COG Metrics:")
    # for cog, metrics in cog_metrics.items():
    #     print(f"{cog}: Accuracy={metrics[0]:.4f}, Precision={metrics[1]:.4f}, Recall={metrics[2]:.4f}, F1={metrics[3]:.4f}")

    # # Convert sample stats to numpy for easier processing
    # sample_stats = np.array(sample_stats)
    # print(f"Average Per-Sample Metrics: Accuracy={sample_stats[:,0].mean():.4f}, Precision={sample_stats[:,1].mean():.4f}, Recall={sample_stats[:,2].mean():.4f}, F1={sample_stats[:,3].mean():.4f}")

    to_plot_distributions_per_sample = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for sample_st in sample_stats:
        to_plot_distributions_per_sample["accuracy"].append(sample_st[0])
        if sample_st[1] > 0: 
            to_plot_distributions_per_sample["precision"].append(sample_st[1])
        if sample_st[2] > 0:     
            to_plot_distributions_per_sample["recall"].append(sample_st[2])
        if sample_st[3] > 0:    
            to_plot_distributions_per_sample["f1"].append(sample_st[3])

            
    to_plot_distributions_per_cog = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for cog in cog_metrics.keys():
        to_plot_distributions_per_cog["accuracy"].append(cog_metrics[cog][0])
        if cog_metrics[cog][1] > 0: 
            to_plot_distributions_per_cog["precision"].append(cog_metrics[cog][1])
        if cog_metrics[cog][2] > 0:     
            to_plot_distributions_per_cog["recall"].append(cog_metrics[cog][2])
        if cog_metrics[cog][3] > 0:    
            to_plot_distributions_per_cog["f1"].append(cog_metrics[cog][3])          

    return to_plot_distributions_per_sample, to_plot_distributions_per_cog


def find_miscalssif_cogs(model, train_loader, cog_names, device):
    # Initialize dictionaries for misclassifications and correct classifications
    misclassified_cogs_fp = {cog_name: 0 for cog_name in cog_names}  # False Positives
    misclassified_cogs_fn = {cog_name: 0 for cog_name in cog_names}  # False Negatives
    correct_cogs_tp = {cog_name: 0 for cog_name in cog_names}        # True Positives
    correct_cogs_tn = {cog_name: 0 for cog_name in cog_names}        # True Negatives

    num_samples_processed = 0
    with torch.no_grad():
        for tokens, mask, targets in tqdm(train_loader, desc="Processing Batches"):
            tokens, mask, targets = tokens.to(device), mask.to(device), targets.to(device)
            preds = (model(tokens, mask) > 0.5).float()  
            num_samples_processed += len(tokens)

            # Identify misclassifications and correct classifications
            false_positives = (targets == 0) & (preds == 1)  
            false_negatives = (targets == 1) & (preds == 0)  
            true_positives = (targets == 1) & (preds == 1)  
            true_negatives = (targets == 0) & (preds == 0) 

            # Get indices of False Positives, False Negatives, True Positives, and True Negatives
            fp_indices = false_positives.nonzero(as_tuple=True)[1].cpu().numpy()
            fn_indices = false_negatives.nonzero(as_tuple=True)[1].cpu().numpy()
            tp_indices = true_positives.nonzero(as_tuple=True)[1].cpu().numpy()
            tn_indices = true_negatives.nonzero(as_tuple=True)[1].cpu().numpy()

            for idx in fp_indices:
                misclassified_cogs_fp[cog_names[idx]] += 1  

            for idx in fn_indices:
                misclassified_cogs_fn[cog_names[idx]] += 1  

            for idx in tp_indices:
                correct_cogs_tp[cog_names[idx]] += 1  

            for idx in tn_indices:
                correct_cogs_tn[cog_names[idx]] += 1      

    return misclassified_cogs_fp, misclassified_cogs_fn, correct_cogs_tp, correct_cogs_tn, num_samples_processed           


def top_missclass(misclassified_cogs_fp, num_samples_processed, title, N):
    top_10_fp = sorted(misclassified_cogs_fp.items(), key=lambda x: x[1], reverse=True)[:N]
    
    # Extract COG names and counts separately
    top_10_fp_cogs, top_10_fp_counts = zip(*top_10_fp) if top_10_fp else ([], [])
    
    # Print results
    print(title)
    for cog, count in zip(top_10_fp_cogs, top_10_fp_counts):
        print(f"{cog}: misclassification rate {round(count/num_samples_processed,2)}")

def accuracy(tp, tn, fp, fn):
    return (tp + tn)/(tp + tn + fp + fn)
    
def precision (tp, fp):
    return tp /(tp + fp) if (tp + fp) > 0 else 0.0

def recall (tp, fn):
    return tp/(tp + fn)  if (tp + fn) > 0 else 0.0

def f1_score(pres, recall):
    return 2*(pres*recall)/(pres+recall)  if (pres+recall) > 0 else 0.0        

def plot_misclassification_histogram(counts, title, color, xlabel):
    plt.figure(figsize=(15, 1))
    plt.bar(range(len(counts)), counts, color=color)
    plt.xlabel(xlabel)
    plt.ylabel('misclassification rate')
    plt.title(title)
    #plt.xticks(rotation=90, fontsize=8)
  #  plt.yscale("log")  # Log scale for better visualization
    plt.show()        

def misclassif_metrics_per_posit(misclassified_cogs_fp, misclassified_cogs_fn, correct_cogs_tp, correct_cogs_tn, num_samples_processed):
    fp_cogs, fp_counts = zip(*misclassified_cogs_fp.items()) if misclassified_cogs_fp else ([], [])
    fn_cogs, fn_counts = zip(*misclassified_cogs_fn.items()) if misclassified_cogs_fn else ([], [])
    tp_cogs, tp_counts = zip(*correct_cogs_tp.items()) if correct_cogs_tp else ([], [])
    tn_cogs, tn_counts = zip(*correct_cogs_tn.items()) if correct_cogs_tn else ([], [])

    fp_rate = [f/num_samples_processed for f in fp_counts]
    fn_rate = [f/num_samples_processed for f in fn_counts]
    tp_rate = [f/num_samples_processed for f in tp_counts]
    tn_rate = [f/num_samples_processed for f in tn_counts]

    accurac_per_cog =  [accuracy(tp_rate[i], tn_rate[i], fp_rate[i], fn_rate[i]) for i in range(len(fn_rate))]
    precis_per_cog = [precision(tp_rate[i], fp_rate[i]) for i in range(len(fn_cogs))] 
    recall_per_cog = [recall(tp_rate[i], fn_rate[i]) for i in range(len(fn_cogs))]
    f1_score_per_cog = [f1_score(precis_per_cog[i], recall_per_cog[i]) for i in range(len(precis_per_cog))]

    return fp_rate, fn_rate, tp_rate, tn_rate, accurac_per_cog, precis_per_cog, recall_per_cog, f1_score_per_cog

