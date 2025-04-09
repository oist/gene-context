import torch
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# --- Helper Loss Functions ---
def soft_f1_loss(y_true, y_pred, eps=1e-7):
    """
    Computes a differentiable approximation of the F1 loss.
    y_true and y_pred are tensors of shape (B, vocab_size).
    """
    tp = (y_true * y_pred).sum(dim=1)
    fp = ((1 - y_true) * y_pred).sum(dim=1)
    fn = (y_true * (1 - y_pred)).sum(dim=1)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return 1 - f1.mean()  # We want to maximize F1, so minimize (1-F1)

def combined_loss(y_true, y_pred, alpha=0.5):
    """
    Combines standard BCE loss with the soft F1 loss.
    alpha determines the trade-off between BCE and soft F1.
    """
    bce = F.binary_cross_entropy(y_pred, y_true)
    f1  = soft_f1_loss(y_true, y_pred)
    return alpha * bce + (1 - alpha) * f1

def evaluate_metrics_extended(model, dataloader, device, threshold=0.5):
    """
    Evaluate the model over the given dataloader and compute:
      - Per-genome Accuracy, Precision, Recall, F1 based on thresholded predictions.
      - Average Genome Size Difference (absolute difference between number of positives in target and prediction).
      - FP Noise Removed Fraction: Among genes observed in the noisy input that are false positives
        (observed but truly absent), the fraction correctly predicted as absent.
      - FN Noise Recovered Fraction: Among genes that were missed in the noisy input (false negatives)
        (not observed but truly present), the fraction correctly predicted as present.
    """
    model.eval()
    all_preds = []         # predicted probabilities, shape (N, V)
    all_targets = []       # ground truth binary labels, shape (N, V)
    abs_genome_size_diff_list = []
    fp_removed_list = []   # List to store per-genome FP noise removal fraction
    fn_recovered_list = [] # List to store per-genome FN noise recovery fraction
    
    for tokens, mask, targets in tqdm(dataloader, desc="Evaluating the pre-trained model"):
        tokens = tokens.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            preds = model(tokens, mask)  # shape: (B, vocab_size)
        # Store predictions and targets for overall metrics.
        all_preds.append(preds.cpu().detach().numpy())
        all_targets.append(targets.cpu().detach().numpy())
        
        # Convert predictions to binary using the threshold.
        preds_np = (preds.cpu().detach().numpy() >= threshold).astype(int)
        targets_np = targets.cpu().detach().numpy().astype(int)
        batch_size = targets_np.shape[0]
        
        # Compute genome size difference per sample.
        for i in range(batch_size):
            true_size = np.sum(targets_np[i])
            pred_size = np.sum(preds_np[i])
            abs_diff = np.abs(true_size / pred_size)
            abs_genome_size_diff_list.append(abs_diff)
        
        # --- Compute noise correction metrics ---
        # For each sample in the batch, reconstruct the observed input (before noise correction)
        # from the tokens using the provided mask.
        tokens_np = tokens.cpu().detach().numpy()  # Shape: (B, N, 2)
        mask_np = mask.cpu().detach().numpy()        # Shape: (B, N)
        for i in range(batch_size):
            # Initialize an empty observed vector for this genome.
            observed = np.zeros(model.vocab_size, dtype=int)
            # For valid (unpadded) tokens, mark the gene as observed.
            valid_tokens = tokens_np[i][mask_np[i] == False]
            for token in valid_tokens:
                cog_id = int(token[0])
                # Make sure not to include pad tokens.
                if cog_id < model.vocab_size:
                    observed[cog_id] = 1
            # FP noise: observed but should be 0 in the target.
            fp_noise = (observed == 1) & (targets_np[i] == 0)
            # FN noise: not observed but should be 1 in the target.
            fn_noise = (observed == 0) & (targets_np[i] == 1)
            pred_binary = preds_np[i]
            # Compute fraction of FP noise that were correctly removed (predicted 0).
            if np.sum(fp_noise) > 0:
                fp_removed = np.sum((pred_binary == 0) & fp_noise) / np.sum(fp_noise)
                fp_removed_list.append(fp_removed)
            # Compute fraction of FN noise that were recovered (predicted 1).
            if np.sum(fn_noise) > 0:
                fn_recovered = np.sum((pred_binary == 1) & fn_noise) / np.sum(fn_noise)
                fn_recovered_list.append(fn_recovered)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    binary_preds = (all_preds >= threshold).astype(int)
    N, V = all_targets.shape
    
    per_sample_acc, per_sample_prec, per_sample_rec, per_sample_f1, per_sample_roc_auc  = [], [], [], [], []
    
    # Compute per-sample classification metrics.
    for i in range(N):
        TP = np.sum((all_targets[i] == 1) & (binary_preds[i] == 1))
        TN = np.sum((all_targets[i] == 0) & (binary_preds[i] == 0))
        FP = np.sum((all_targets[i] == 0) & (binary_preds[i] == 1))
        FN = np.sum((all_targets[i] == 1) & (binary_preds[i] == 0))
        acc = (TP + TN) / V
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        roc_auc = roc_auc_score(all_targets[i], all_preds[i])
                
        per_sample_acc.append(acc)
        per_sample_prec.append(prec)
        per_sample_rec.append(rec)
        per_sample_f1.append(f1)
        per_sample_roc_auc.append(roc_auc)
    
    # Average the noise correction metrics over samples that had any noise.
    avg_fp_removed = np.mean(fp_removed_list) if fp_removed_list else float('nan')
    avg_fn_recovered = np.mean(fn_recovered_list) if fn_recovered_list else float('nan')
    
    metrics = {
        "avg_accuracy": np.mean(per_sample_acc),
        "avg_precision": np.mean(per_sample_prec),
        "avg_recall": np.mean(per_sample_rec),
        "avg_f1": np.mean(per_sample_f1),
        "avg_genome_size_diff": np.mean(abs_genome_size_diff_list),
        "avg_fp_removed_fraction": avg_fp_removed,
        "avg_fn_recovered_fraction": avg_fn_recovered,
        "avg_roc_auc": np.mean(per_sample_roc_auc),
    }
    return metrics