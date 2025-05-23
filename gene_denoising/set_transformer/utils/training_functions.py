import torch
import torch.nn as nn
from tqdm import tqdm

from set_transformer.utils.metrics import evaluate_metrics_extended, combined_loss
from data_processing_utils.data_processing_functions import print_to_file, print_to_file_block

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
def train_and_validate(model, train_loader, val_loader, optimizer, num_epochs, device, output_file, full_model_filename,
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
        # Training step
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc="Epoch {} Training".format(epoch+1))
        for tokens, mask, targets in train_bar:
            tokens  = tokens.to(device)
            mask    = mask.to(device)
            targets = targets.to(device) #shape: (B, vocab_size)
            
            optimizer.zero_grad()
            preds = model(tokens, mask)  # Predicted probabilities. Expected shape: (B, vocab_size)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print_to_file(output_file, "Epoch [{}/{}] Training Loss: {:.4f}, Learning Rate: {:.6f}".format(
            epoch+1, num_epochs, avg_loss, current_lr))
        

        # Validation step
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Epoch {} Validation".format(epoch+1))
        for tokens, mask, targets in val_bar:
            tokens  = tokens.to(device)
            mask    = mask.to(device)
            targets = targets.to(device) 
        
            preds = model(tokens, mask)  # Predicted probabilities. Expected shape: (B, vocab_size)
            loss = criterion(preds, targets)
            val_running_loss += loss.item()

        aver_val_loss = val_running_loss / len(val_loader)      
        current_lr = optimizer.param_groups[0]['lr']
        print_to_file(output_file, "Epoch [{}/{}] Validation Loss: {:.4f}, Learning Rate: {:.6f}".format(epoch+1, num_epochs, aver_val_loss, current_lr))

        # Evaluate using extended metrics.
        extended_metrics = evaluate_metrics_extended(model, val_loader, device, threshold)
        print_to_file(output_file, "\nExtended Evaluation Metrics (Epoch {}):".format(epoch+1))
        print_to_file_block(output_file, extended_metrics)

    torch.save(model.state_dict(), full_model_filename)
    print_to_file(output_file, f"Model checkpoint saved to {full_model_filename}")        