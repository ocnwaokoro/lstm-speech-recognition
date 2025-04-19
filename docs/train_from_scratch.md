# train_from_scratch.ipynb Documentation

## Overview

The `train_from_scratch.ipynb` notebook implements training for the speech recognition model from scratch (without pre-trained weights). This notebook provides a complete training pipeline including data loading, model initialization, training loop, and evaluation. It's ideal for users who want to train models with custom configurations or on new datasets without relying on previous weights.

## Notebook Structure

### 1. Environment Setup

```python
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SpeechRecognition
from dataLoader import Data, collate_fn_padd
import matplotlib.pyplot as plt
device = "cuda:0"

batchSize = 64
numWorkers = 8
```

This section:
- Imports required libraries (PyTorch, numpy, etc.)
- Sets computation device (GPU)
- Configures data loading parameters (batch size, worker threads)
- Prepares the environment for model training

### 2. Data Loaders

```python
def train_dataloader():
    d_params = Data.parameters
    train_dataset = Data(json_path="data/train.json", **d_params)
    return DataLoader(dataset=train_dataset,
                        batch_size=batchSize,
                        num_workers=numWorkers,
                        pin_memory=True,
                        collate_fn=collate_fn_padd)
                        
def valid_dataloader():
    d_params = Data.parameters
    valid_dataset = Data(json_path="data/test.json", **d_params)
    return DataLoader(dataset=valid_dataset,
                        batch_size=batchSize,
                        num_workers=numWorkers,
                        pin_memory=True,
                        collate_fn=collate_fn_padd)

trainLoader = train_dataloader()
validLoader = valid_dataloader()
```

This section:
- Creates data loaders for both training and validation sets
- Utilizes the `Data` class to process audio files and transcriptions
- Implements batching and padding with custom collation
- Uses training and testing JSONs created during data preparation

### 3. Training Function

```python
def train(model, optimizer, criterion, numEpochs=10, title="Model"):
    trnLoss = []
    valLoss = []
    for epoch in range(numEpochs):
        # Training phase
        model.train()
        tmpTrnLoss = []
        for i, (spectrograms, labels, input_lengths, label_lengths) in enumerate(trainLoader): 
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            bs = spectrograms.shape[0]
            hidden = model._init_hidden(bs)
            hn, c0 = hidden[0].to(device), hidden[1].to(device)
            output, _ = model(spectrograms, (hn, c0))
            output = F.log_softmax(output, dim=2)
            loss = criterion(output, labels, input_lengths, label_lengths)
            tmpTrnLoss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        trnLoss.append(torch.mean(torch.tensor(tmpTrnLoss)))
        print(f"Train Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        tmpValLoss = []
        with torch.no_grad():
            for i, (spectrograms, labels, input_lengths, label_lengths) in enumerate(validLoader):
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                bs = spectrograms.shape[0]
                hidden = model._init_hidden(bs)
                hn, c0 = hidden[0].to(device), hidden[1].to(device)
                output, _ = model(spectrograms, (hn, c0))
                output = F.log_softmax(output, dim=2)
                loss = criterion(output, labels, input_lengths, label_lengths)
                tmpValLoss.append(loss)
            print(f"Valid Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item():.4f}")
            valLoss.append(torch.mean(torch.tensor(tmpValLoss)))
            
    return trnLoss, valLoss
```

The training function:
- Implements a complete training and validation loop
- Tracks losses for both training and validation phases
- Prints progress for each epoch
- Handles the CTC loss computation with proper length masking
- Returns loss histories for visualization

### 4. Model Initialization and Training

```python
srmodel = SpeechRecognition().to(device)
optimizer = optim.AdamW(srmodel.parameters(), 1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min',
    factor=0.50, patience=6)
criterion = nn.CTCLoss(blank=28, zero_infinity=True)
trnLoss, valLoss = train(srmodel, optimizer, criterion, 100)
```

This section:
- Initializes a fresh speech recognition model from the `SpeechRecognition` class
- Creates an AdamW optimizer with a learning rate of 1e-3
- Configures a learning rate scheduler that reduces the rate by 50% after 6 epochs without improvement
- Sets up the CTC loss function with blank token index 28
- Trains the model for 100 epochs, collecting loss values for both training and validation

### 5. Result Visualization

```python
plt.plot([x.cpu().detach().numpy() for x in trnLoss], label = "Train Loss")
plt.plot([x.cpu().detach().numpy() for x in valLoss], label = "Valid Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

This section:
- Plots training and validation loss curves over epochs
- Visualizes the model's learning progress
- Helps identify overfitting or convergence issues
- Uses matplotlib for visualization with proper labeling

### 6. Testing with Transcription Examples

This final section tests the trained model on validation examples:
- Loads ten samples from the validation set
- Processes each example through the model
- Compares the ground truth transcription with the model prediction
- Provides a qualitative evaluation of model performance

## Technical Details

### CTC Loss Function

The notebook uses the Connectionist Temporal Classification (CTC) loss function, which:
- Allows variable-length input sequences to map to variable-length output sequences
- Does not require explicit alignment between audio frames and characters
- Introduces a blank token to handle repeated characters and alignment
- Measures the difference between predicted and actual transcriptions

### Data Augmentation

The training data loader applies SpecAugment, which:
- Randomly masks frequency bands in the spectrogram
- Randomly masks time steps in the spectrogram
- Increases model robustness to variations in audio
- Helps prevent overfitting

### Learning Rate Scheduling

The notebook implements a learning rate scheduler that:
- Monitors validation loss over epochs
- Reduces learning rate when no improvement is seen
- Helps escape plateaus during training
- Adapts optimization to different training phases

## Usage Guidelines

### Hardware Requirements

For optimal performance with this notebook:
- GPU with at least 8GB VRAM (16GB+ recommended)
- 16GB+ system RAM
- Fast storage for data loading

### Training Dataset Size

The notebook is designed to handle large datasets:
- Can process thousands of hours of audio
- Uses efficient batching for memory management
- Supports distributed training with minor modifications

### Customization Points

Key parameters you might want to adjust:
- `batchSize`: Change based on available GPU memory
- `numEpochs`: More epochs for complex datasets
- Learning rate: Adjust for different convergence patterns
- Model architecture: Modify the `SpeechRecognition` class instantiation

### Expected Outputs

After successful training:
- Model weights saved to disk
- Loss curves showing convergence pattern
- Example transcriptions from the validation set
- Readable output in the notebook cells

## Common Issues and Solutions

1. **Out of Memory Errors**
   - Reduce batch size
   - Decrease model complexity
   - Use gradient checkpointing

2. **Slow Convergence**
   - Adjust learning rate
   - Implement learning rate warmup
   - Check data quality and preprocessing

3. **Overfitting**
   - Increase dropout rate
   - Add more data augmentation
   - Implement early stopping

4. **Poor Transcription Quality**
   - Verify audio preprocessing
   - Check character set and preprocessing
   - Ensure proper CTC configuration