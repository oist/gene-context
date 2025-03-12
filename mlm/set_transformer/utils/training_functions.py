import torch
import torch.nn as nn
from tqdm import tqdm

from set_transformer.utils.metrics import evaluate_metrics_extended, combined_loss
from data_processing_utils.data_processing_functions import print_to_file

def initialize_weights(module):
    # Initialize Linear layers using Kaiming initialization for ReLU
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    # Optionally, initialize Embedding layers using Xavier initialization.
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)


############################################
# Modified Training and Validation Function
############################################
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Train and Validate Function ---
def train_and_validate(model, train_loader, val_loader, optimizer, num_epochs, device, output_file, 
                       threshold=0.5, max_iters=10, use_focal_loss=False, focal_alpha=1.0, focal_gamma=2.0,combined_alpha=0.5, 
                       use_combined_loss=False, criterion=None):
    """
    Train the model and evaluate using:
      1. Extended metrics: per-genome Accuracy, Precision, Recall, F1, and genome size difference.
      2. Iterative evaluation: refined predictions using iterative inference.
      
    Loss equation options:
      - Standard BCE Loss: L = -[ y*log(p) + (1-y)*log(1-p) ]
      - Focal Loss (if use_focal_loss=True):
          FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t)
      - Combined Loss (if use_combined_loss=True):
          combined_loss = alpha * BCE + (1 - alpha) * soft F1 loss.
    
    A ReduceLROnPlateau scheduler reduces the learning rate when the training loss plateaus.
    """
    model.to(device)
    
    # Set criterion if not provided.
    if criterion is None:
        if use_combined_loss:
            # Use combined loss (BCE + soft F1 loss)
            criterion = lambda preds, targets: combined_loss(targets, preds, alpha=combined_alpha)
            print_to_file(output_file, "Using combined loss (BCE + soft F1 loss)")
        elif use_focal_loss:
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print_to_file(output_file, "Using Focal Loss with alpha={} and gamma={}".format(focal_alpha, focal_gamma))
        else:
            criterion = nn.BCELoss()
            print_to_file(output_file, "Using standard BCE Loss")
    
    # Set up learning rate scheduler.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc="Epoch {} Training".format(epoch+1))
        for tokens, mask, targets in train_bar:
            tokens  = tokens.to(device)
            mask    = mask.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(tokens, mask)  # Expected shape: (B, vocab_size)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print_to_file(output_file, "Epoch [{}/{}] Training Loss: {:.4f}, Learning Rate: {:.6f}".format(
            epoch+1, num_epochs, avg_loss, current_lr))
        
        # Step the scheduler based on average training loss.
        scheduler.step(avg_loss)
        
        # Evaluate using extended metrics.
        extended_metrics = evaluate_metrics_extended(model, val_loader, device, threshold)
        print_to_file(output_file, "\nExtended Evaluation Metrics (Epoch {}):".format(epoch+1))
        print_to_file(output_file, "  Average per-genome Accuracy         : {:.4f}".format(extended_metrics["avg_accuracy"]))
        print_to_file(output_file, "  Average per-genome Precision (macro)  : {:.4f}".format(extended_metrics["avg_precision"]))
        print_to_file(output_file, "  Average per-genome Recall (macro)     : {:.4f}".format(extended_metrics["avg_recall"]))
        print_to_file(output_file, "  Average per-genome F1 (macro)         : {:.4f}".format(extended_metrics["avg_f1"]))
        print_to_file(output_file, "  Average Genome Size Difference (abs)  : {:.4f}".format(extended_metrics["avg_genome_size_diff"]))
        print_to_file(output_file, "  Average FP Noise Removed Fraction   : {:.4f}".format(extended_metrics["avg_fp_removed_fraction"]))
        print_to_file(output_file, "  Average FN Noise Recovered Fraction : {:.4f}".format(extended_metrics["avg_fn_recovered_fraction"]))    

    torch.save(model.state_dict(), "model_checkpoint_full.pth")
    print_to_file(output_file, "Model checkpoint saved to model_checkpoint_full.pth")        