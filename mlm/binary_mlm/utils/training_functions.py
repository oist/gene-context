import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

import gc
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_processing_utils.data_processing_functions import print_to_file

def validate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc="validate.."):
            # Transfer only the current batch to GPU.
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, loss = model(input_ids, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100
            all_preds.extend(preds[mask].cpu().numpy().tolist())
            all_labels.extend(labels[mask].cpu().numpy().tolist())
            
            torch.cuda.empty_cache()
    
    if len(all_labels) == 0:
        return None
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, pos_label=1)
    rec = recall_score(all_labels, all_preds, pos_label=1)
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    avg_loss = total_loss / len(dataloader)
    metrics = {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }
    return metrics

def train_model_simple(model, train_loader, val_loader, device, output_file, epochs=5, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Set up mixed precision training.
    scaler = torch.cuda.amp.GradScaler()
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        metrics = validate(model, val_loader, device)
        if metrics:
            print_to_file(output_file, f"Validation Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        else:
            print_to_file(output_file, "Warning: No masked tokens in validation set!")
        
        for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                _, loss = model(input_ids, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            del input_ids, labels, loss
            torch.cuda.empty_cache()
            
        avg_loss = total_loss / len(train_loader)
        print_to_file(output_file, f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")
        
        metrics = validate(model, val_loader, device)
        if metrics:
            print_to_file(output_file, f"Validation Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        else:
            print_to_file(output_file, "Warning: No masked tokens in validation set!")
        
        scheduler.step(avg_loss)
        
        torch.cuda.empty_cache()
        gc.collect()
        
    return model


def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    scaler = GradScaler()
    model.to(device)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        model.train()
        total_loss = 0
        for input_ids, labels in tqdm(train_loader, desc="Training"):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                _, loss = model(input_ids, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            del input_ids, labels, loss
            torch.cuda.empty_cache()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")
        
        combined_metrics = combined_validate(model, val_loader, device)
        std_met = combined_metrics["standard_metrics"]
        ext_met = combined_metrics["extended_metrics"]
        print("Post-training Validation Metrics (Standard masked):")
        print(f"  Loss: {std_met['loss']:.4f}, Accuracy: {std_met['accuracy']:.4f}, "
              f"Precision: {std_met['precision']:.4f}, Recall: {std_met['recall']:.4f}, F1: {std_met['f1']:.4f}")
        print("Extended Validation Metrics (Corrupted unmasked & masked performance):")
        for (fn_rate, fp_rate), metrics in ext_met.items():
            print(f"  Setting: false_negative={fn_rate}, false_positive={fp_rate}")
            print(f"    False negatives recovered: {metrics['false_negative_recovered']}")
            print(f"    False positives removed:  {metrics['false_positive_removed']}")
            print(f"    Masked token accuracy:      {metrics['masked_accuracy']}")
            print(f"    Masked token precision:     {metrics['masked_precision']}")
            print(f"    Masked token recall:        {metrics['masked_recall']}")
            print(f"    Masked token F1:            {metrics['masked_f1']}")
        
        scheduler.step(avg_loss)
        torch.cuda.empty_cache()
        gc.collect()
        
    return model



def combined_validate_todelete(model, dataloader, device, corruption_settings=None):
    """
    Compute standard masked validation metrics and extended metrics on corrupted data.
    
    Standard metrics are computed on positions where labels != -100.
    
    For extended validation, unmasked tokens (labels == -100) are corrupted by:
      - Flipping 1 -> 0 with probability false_negative.
      - Flipping 0 -> 1 with probability false_positive.
    Then for each setting, we compute:
      - false_negative_recovered: fraction of corrupted tokens originally 1 predicted as 1.
      - false_positive_removed: fraction of corrupted tokens originally 0 predicted as 0.
      - masked_accuracy: accuracy on masked tokens (which remain uncorrupted).
    
    Returns a dictionary with "standard_metrics" and "extended_metrics".
    """
    if corruption_settings is None:
        corruption_settings = [(0.1, 0.01), (0.2, 0.01), (0.3, 0.01), (0.5, 0.01)]
    
    # Standard masked validation metrics.
    all_masked_preds = []
    all_masked_labels = []
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, loss = model(input_ids, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            masked_mask = (labels != -100)
            all_masked_preds.extend(preds[masked_mask].cpu().numpy().tolist())
            all_masked_labels.extend(labels[masked_mask].cpu().numpy().tolist())
    standard_loss = total_loss / len(dataloader)
    standard_metrics = {
        'loss': standard_loss,
        'accuracy': accuracy_score(all_masked_labels, all_masked_preds),
        'precision': precision_score(all_masked_labels, all_masked_preds, pos_label=1),
        'recall': recall_score(all_masked_labels, all_masked_preds, pos_label=1),
        'f1': f1_score(all_masked_labels, all_masked_preds, pos_label=1)
    }
    
    # Extended validation metrics.
    ext_results = {setting: {"fn_total": 0, "fn_recovered": 0,
                             "fp_total": 0, "fp_removed": 0,
                             "masked_total": 0, "masked_correct": 0}
                   for setting in corruption_settings}
    
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masked_mask = (labels != -100)
            unmasked_mask = (labels == -100)
            ground_truth = input_ids.clone()
            for false_negative, false_positive in corruption_settings:
                corrupted_input = input_ids.clone()
                rand_vals = torch.rand(corrupted_input.shape, device=corrupted_input.device)
                flip_fn = (ground_truth == 1) & unmasked_mask & (rand_vals < false_negative)
                flip_fp = (ground_truth == 0) & unmasked_mask & (rand_vals < false_positive)
                corrupted_input[flip_fn] = 0
                corrupted_input[flip_fp] = 1
                
                logits, _ = model(corrupted_input, labels)
                preds = torch.argmax(logits, dim=-1)
                
                fn_flipped = flip_fn
                fn_total = fn_flipped.sum().item()
                fn_recovered = (preds[fn_flipped] == 1).sum().item() if fn_total > 0 else 0
                
                fp_flipped = flip_fp
                fp_total = fp_flipped.sum().item()
                fp_removed = (preds[fp_flipped] == 0).sum().item() if fp_total > 0 else 0
                
                ext_results[(false_negative, false_positive)]["fn_total"] += fn_total
                ext_results[(false_negative, false_positive)]["fn_recovered"] += fn_recovered
                ext_results[(false_negative, false_positive)]["fp_total"] += fp_total
                ext_results[(false_negative, false_positive)]["fp_removed"] += fp_removed
                
                masked_preds = preds[masked_mask]
############################################
# Combined Validation Function
############################################

def combined_validate(model, dataloader, device, corruption_settings=None):
    """
    Compute standard masked validation metrics and extended metrics on corrupted data.
    
    Standard metrics are computed on positions where labels != -100.
    
    For extended validation, unmasked tokens (labels == -100) are corrupted by:
      - Flipping 1 -> 0 with probability false_negative.
      - Flipping 0 -> 1 with probability false_positive.
    Then for each setting, we compute:
      - false_negative_recovered: fraction of corrupted tokens originally 1 predicted as 1.
      - false_positive_removed: fraction of corrupted tokens originally 0 predicted as 0.
      - Masked token performance on the corrupted input: accuracy, precision, recall, and F1 
        computed on the masked tokens (which remain uncorrupted).
    
    Returns a dictionary with two keys:
         "standard_metrics": standard masked metrics.
         "extended_metrics": a dict keyed by corruption setting (tuple) with the following keys:
            - "false_negative_recovered"
            - "false_positive_removed"
            - "masked_accuracy"
            - "masked_precision"
            - "masked_recall"
            - "masked_f1"
    """
    if corruption_settings is None:
        corruption_settings = [(0.1, 0.01), (0.2, 0.01), (0.3, 0.01), (0.5, 0.01)]
    
    model.eval()
    
    # ---- Standard Masked Metrics ----
    all_masked_preds = []
    all_masked_labels = []
    total_loss = 0
    for input_ids, labels in tqdm(dataloader, desc="Standard Validation"):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, loss = model(input_ids, labels)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        masked_mask = (labels != -100)
        all_masked_preds.extend(preds[masked_mask].cpu().numpy().tolist())
        all_masked_labels.extend(labels[masked_mask].cpu().numpy().tolist())
    standard_loss = total_loss / len(dataloader)
    standard_metrics = {
        'loss': standard_loss,
        'accuracy': accuracy_score(all_masked_labels, all_masked_preds),
        'precision': precision_score(all_masked_labels, all_masked_preds, pos_label=1),
        'recall': recall_score(all_masked_labels, all_masked_preds, pos_label=1),
        'f1': f1_score(all_masked_labels, all_masked_preds, pos_label=1)
    }
    
    # ---- Extended Metrics on Corrupted Unmasked Tokens ----
    # For each corruption setting, accumulate:
    # - Counts for false negatives (flipped tokens originally 1) and false positives (flipped tokens originally 0).
    # - Also, accumulate masked token predictions and labels from the corrupted input.
    ext_results = {
        setting: {
            "fn_total": 0, "fn_recovered": 0,
            "fp_total": 0, "fp_removed": 0,
            "masked_total": 0, "masked_correct": 0,
            "masked_preds_list": [],
            "masked_labels_list": []
        }
        for setting in corruption_settings
    }
    
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc="Extended Validation"):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masked_mask = (labels != -100)
            unmasked_mask = (labels == -100)
            ground_truth = input_ids.clone()  # true values for unmasked tokens
            
            for false_negative, false_positive in corruption_settings:
                # Create a corrupted copy.
                corrupted_input = input_ids.clone()
                rand_vals = torch.rand(corrupted_input.shape, device=corrupted_input.device)
                flip_fn = (ground_truth == 1) & unmasked_mask & (rand_vals < false_negative)
                flip_fp = (ground_truth == 0) & unmasked_mask & (rand_vals < false_positive)
                corrupted_input[flip_fn] = 0
                corrupted_input[flip_fp] = 1
                
                logits, _ = model(corrupted_input, labels)
                preds = torch.argmax(logits, dim=-1)
                
                # For false negatives: tokens originally 1 that were flipped to 0.
                fn_flipped = flip_fn
                fn_total = fn_flipped.sum().item()
                fn_recovered = (preds[fn_flipped] == 1).sum().item() if fn_total > 0 else 0
                
                # For false positives: tokens originally 0 that were flipped to 1.
                fp_flipped = flip_fp
                fp_total = fp_flipped.sum().item()
                fp_removed = (preds[fp_flipped] == 0).sum().item() if fp_total > 0 else 0
                
                ext_results[(false_negative, false_positive)]["fn_total"] += fn_total
                ext_results[(false_negative, false_positive)]["fn_recovered"] += fn_recovered
                ext_results[(false_negative, false_positive)]["fp_total"] += fp_total
                ext_results[(false_negative, false_positive)]["fp_removed"] += fp_removed
                
                # For masked tokens (which remain uncorrupted), accumulate predictions and labels.
                masked_preds = preds[masked_mask]
                masked_labels = labels[masked_mask]
                ext_results[(false_negative, false_positive)]["masked_total"] += masked_mask.sum().item()
                ext_results[(false_negative, false_positive)]["masked_correct"] += (masked_preds == masked_labels).sum().item()
                # Also accumulate the lists for further metric computation.
                ext_results[(false_negative, false_positive)]["masked_preds_list"].append(masked_preds.cpu().numpy().tolist())
                ext_results[(false_negative, false_positive)]["masked_labels_list"].append(masked_labels.cpu().numpy().tolist())
    
    extended_metrics = {}
    for setting, counts in ext_results.items():
        fn_total = counts["fn_total"]
        fp_total = counts["fp_total"]
        recovered_frac = counts["fn_recovered"] / fn_total if fn_total > 0 else None
        removed_frac = counts["fp_removed"] / fp_total if fp_total > 0 else None
        masked_acc = counts["masked_correct"] / counts["masked_total"] if counts["masked_total"] > 0 else None
        
        # Flatten the lists of masked predictions and labels.
        flat_masked_preds = sum(counts["masked_preds_list"], [])
        flat_masked_labels = sum(counts["masked_labels_list"], [])
        # Compute additional metrics.
        masked_prec = precision_score(flat_masked_labels, flat_masked_preds, pos_label=1) if len(flat_masked_labels) > 0 else None
        masked_rec = recall_score(flat_masked_labels, flat_masked_preds, pos_label=1) if len(flat_masked_labels) > 0 else None
        masked_f1 = f1_score(flat_masked_labels, flat_masked_preds, pos_label=1) if len(flat_masked_labels) > 0 else None
        
        extended_metrics[setting] = {
            "false_negative_recovered": recovered_frac,
            "false_positive_removed": removed_frac,
            "masked_accuracy": masked_acc,
            "masked_precision": masked_prec,
            "masked_recall": masked_rec,
            "masked_f1": masked_f1
        }
    
    return {
        "standard_metrics": standard_metrics,
        "extended_metrics": extended_metrics
    }