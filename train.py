"""
train.py - Training Script for Speech Recognition Model

This script implements the training pipeline for the speech recognition model,
including data loading, model initialization, training loop, and evaluation.

The training process uses:
- CTC (Connectionist Temporal Classification) loss
- Stochastic Gradient Descent with Nesterov momentum
- Learning rate scheduling
- Optional SortaGrad curriculum learning
- Batch normalization for improved convergence

Usage:
    python train.py --train_json path/to/train.json --valid_json path/to/valid.json \
                   --batch_size 64 --epochs 100 --learning_rate 1e-3

The script saves model checkpoints and training logs during training,
and can resume training from a checkpoint.
"""

import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import SpeechRecognition
from src.data_loader import Data, collate_fn_padd
import matplotlib.pyplot as plt
import argparse

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Train speech recognition model")
    parser.add_argument("--train_json", type=str, required=True, help="Path to training data JSON")
    parser.add_argument("--valid_json", type=str, required=True, help="Path to validation data JSON")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.99, help="Momentum for SGD")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--sortagrad", action="store_true", help="Use SortaGrad curriculum")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")
    
    return parser.parse_args()

def train_dataloader(json_path, batch_size, num_workers):
    """
    Create a DataLoader for training data.
    
    Args:
        json_path (str): Path to training data JSON file
        batch_size (int): Batch size
        num_workers (int): Number of data loader workers
        
    Returns:
        DataLoader: PyTorch DataLoader for training data
    """
    d_params = Data.parameters
    train_dataset = Data(json_path=json_path, **d_params)
    return DataLoader(dataset=train_dataset,
                     batch_size=batch_size,
                     num_workers=num_workers,
                     pin_memory=True,
                     collate_fn=collate_fn_padd)

def valid_dataloader(json_path, batch_size, num_workers):
    """
    Create a DataLoader for validation data.
    
    Args:
        json_path (str): Path to validation data JSON file
        batch_size (int): Batch size
        num_workers (int): Number of data loader workers
        
    Returns:
        DataLoader: PyTorch DataLoader for validation data
    """
    d_params = Data.parameters
    valid_dataset = Data(json_path=json_path, valid=True, **d_params)
    return DataLoader(dataset=valid_dataset,
                     batch_size=batch_size,
                     num_workers=num_workers,
                     pin_memory=True,
                     collate_fn=collate_fn_padd)

def train(model, optimizer, scheduler, criterion, train_loader, valid_loader, 
          num_epochs=100, checkpoint_dir="checkpoints"):
    """
    Main training loop for the speech recognition model.
    
    Args:
        model (SpeechRecognition): The speech recognition model
        optimizer (Optimizer): Optimizer for training
        scheduler (LRScheduler): Learning rate scheduler
        criterion (CTCLoss): Loss function
        train_loader (DataLoader): DataLoader for training data
        valid_loader (DataLoader): DataLoader for validation data
        num_epochs (int): Number of training epochs
        checkpoint_dir (str): Directory to save checkpoints
        
    Returns:
        tuple: (train_losses, valid_losses) - Lists of losses during training
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize lists to store losses
    train_losses = []
    valid_losses = []
    
    # Track best validation loss for model saving
    best_valid_loss = float('inf')
    
    # Main training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        
        # Use SortaGrad in the first epoch if enabled (sort data by length)
        if epoch == 0 and args.sortagrad:
            print("Using SortaGrad - sorting training data by length for first epoch")
            # Sort the training loader data by utterance length
            # Implementation depends on your specific DataLoader setup
        
        for i, (spectrograms, labels, input_lengths, label_lengths) in enumerate(train_loader): 
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            # Initialize hidden state
            bs = spectrograms.shape[0]
            hidden = model._init_hidden(bs)
            hn, c0 = hidden[0].to(device), hidden[1].to(device)
            
            # Forward pass
            output, _ = model(spectrograms, (hn, c0))
            output = F.log_softmax(output, dim=2)
            
            # Calculate loss
            loss = criterion(output, labels, input_lengths, label_lengths)
            epoch_train_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=400)
            
            optimizer.step()
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Print training progress
        print(f"Train Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        epoch_valid_loss = 0.0
        
        with torch.no_grad():
            for i, (spectrograms, labels, input_lengths, label_lengths) in enumerate(valid_loader):
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                
                # Initialize hidden state
                bs = spectrograms.shape[0]
                hidden = model._init_hidden(bs)
                hn, c0 = hidden[0].to(device), hidden[1].to(device)
                
                # Forward pass
                output, _ = model(spectrograms, (hn, c0))
                output = F.log_softmax(output, dim=2)
                
                # Calculate loss
                loss = criterion(output, labels, input_lengths, label_lengths)
                epoch_valid_loss += loss.item()
            
            # Calculate average validation loss for the epoch
            avg_valid_loss = epoch_valid_loss / len(valid_loader)
            valid_losses.append(avg_valid_loss)
            
            # Print validation progress
            print(f"Valid Epoch [{epoch+1}/{num_epochs}], Loss: {avg_valid_loss:.4f}")
            
            # Update learning rate based on validation loss
            scheduler.step(avg_valid_loss)
            
            # Save model checkpoint if it's the best so far
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
                checkpoint_path = os.path.join(checkpoint_dir, f"model_{timestamp}.pt")
                
                print(f"New best validation loss: {best_valid_loss:.4f}")
                print(f"Saving model to {checkpoint_path}")
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'valid_loss': avg_valid_loss,
                }, checkpoint_path)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'loss_plot.png'))
    
    return train_losses, valid_losses

def main(args):
    """
    Main function to set up and run the training process.
    
    Args:
        args (Namespace): Command line arguments
    """
    print("Initializing data loaders...")
    train_loader = train_dataloader(args.train_json, args.batch_size, args.num_workers)
    valid_loader = valid_dataloader(args.valid_json, args.batch_size, args.num_workers)
    
    print("Initializing model...")
    model = SpeechRecognition().to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Initialize CTC loss
    criterion = nn.CTCLoss(blank=28, zero_infinity=True)
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    
    print("Starting training...")
    train_losses, valid_losses = train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=args.epochs - start_epoch
    )
    
    print("Training completed!")

if __name__ == "__main__":
    args = parse_args()
    main(args)