# Training Notebook Documentation

## Overview

The `train_from_pretrained.ipynb` notebook provides an interactive environment for training the speech recognition model with pre-trained weights. This notebook is ideal for users who want to fine-tune an existing model on new data or continue training from a checkpoint.

## Notebook Sections

### 1. Environment Setup and Data Loading

```python
# Import libraries and set device
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SpeechRecognition
from dataLoader import Data, collate_fn_padd
import matplotlib.pyplot as plt

device = "cuda:0"  # Use first GPU
batchSize = 64
numWorkers = 8
```

This section:
- Imports necessary libraries for training
- Sets the device to GPU if available
- Configures batch size and number of workers for data loading

### 2. Data Loading Functions

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
```

These functions:
- Create PyTorch DataLoader objects for training and validation
- Load data from JSON files created in the data preparation step
- Apply appropriate data parameters from the Data class
- Configure batching with custom collation function for padding

### 3. Training Function Definition

```python
def train(model, optimizer, scheduler, criterion, numEpochs=10, title="Model"):
    trnLoss = []
    valLoss = []
    for epoch in range(numEpochs):
        # Training phase
        model.train()
        tmpTrnLoss = []
        for i, (spectrograms, labels, input_lengths, label_lengths) in enumerate(trainLoader): 
            # Forward pass and loss calculation
            # Backward pass and optimization
            
        # Validation phase
        model.eval()
        tmpValLoss = []
        with torch.no_grad():
            # Evaluate on validation set
            
    return trnLoss, valLoss
```

The training function:
- Handles both training and validation phases
- Tracks loss for both phases across epochs
- Implements standard PyTorch training loop
- Uses learning rate scheduler based on validation performance
- Returns loss history for plotting

### 4. Model Initialization and Training

```python
srmodel = SpeechRecognition().to(device)
checkpoint = torch.load("speechrecognition.ckpt")
state_dict = checkpoint['state_dict']
# Load pre-trained weights
srmodel.load_state_dict(new_state_dict)

# Optimizer and scheduler setup
optimizer = optim.AdamW(srmodel.parameters(), 1e-3)
optimizer.load_state_dict(checkpoint["optimizer_states"][-1])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.50, patience=6)
scheduler.load_state_dict(checkpoint["lr_schedulers"][-1])

# Loss function
criterion = nn.CTCLoss(blank=28, zero_infinity=True)

# Train the model
trnLoss, valLoss = train(srmodel, optimizer, scheduler, criterion, 10)
```

This section:
- Initializes the speech recognition model and moves it to GPU
- Loads pre-trained weights from a checkpoint
- Configures optimizer with AdamW and learning rate scheduler
- Sets up CTC loss function with blank token index 28
- Runs training for 10 epochs

### 5. Model Saving

```python
timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
model_name = f"model_{timestamp}.pt"
torch.save(srmodel.state_dict(), f"savedModel/{model_name}")
```

This part:
- Creates a timestamped filename for the model
- Saves the trained model weights to disk
- Uses PyTorch's save functionality for model persistence

### 6. Visualization and Analysis

```python
# Plot training and validation loss
plt.plot([x.cpu().detach().numpy() for x in trnLoss], label = "Train Loss")
plt.plot([x.cpu().detach().numpy() for x in valLoss], label = "Valid Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

The visualization section:
- Plots training and validation loss curves
- Helps visualize model convergence
- Provides insight into potential overfitting

### 7. Inference Testing

The final section tests the trained model on sample data:
- Loads the saved model
- Processes several test utterances
- Prints both target and predicted transcriptions
- Allows qualitative assessment of model performance

## Usage Notes

1. **Prerequisites**:
   - Processed data in JSON format
   - Pre-trained checkpoint (`speechrecognition.ckpt`)
   - GPU with sufficient memory (recommended 8GB+)

2. **Customization Options**:
   - Adjust `batchSize` based on your GPU memory
   - Modify `numEpochs` in the training call
   - Change learning rate and scheduler parameters
   - Use different model architectures by modifying the `SpeechRecognition` class

3. **Output**:
   - Saved model in the `savedModel` directory
   - Loss curves showing training progress
   - Example transcriptions for qualitative assessment

4. **Potential Issues**:
   - Out of memory errors (reduce batch size)
   - Slow convergence (adjust learning rate)
   - Overfitting (add regularization or early stopping)

5. **Hardware Considerations**:
   - Training runs significantly faster on GPU
   - Multi-GPU training requires code modification
   - For very large models, gradient accumulation may be necessary