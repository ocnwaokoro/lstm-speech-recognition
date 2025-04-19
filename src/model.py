"""
model.py - Speech Recognition Neural Network Architecture

This module implements the core neural network architecture for speech recognition,
inspired by Deep Speech 2 (Baidu Research). The architecture consists of convolutional 
layers for feature extraction, followed by recurrent layers for temporal modeling,
and fully connected layers for character prediction.

The model takes log-mel spectrograms as input and outputs character probabilities
that are then decoded using CTC (Connectionist Temporal Classification).

Architecture overview:
1. Convolutional layers - Extract frequency and temporal features from spectrograms
2. Recurrent layers - Model temporal dependencies in the audio
3. Fully connected layers - Map features to character probabilities

The implementation supports both simple RNNs and GRUs (Gated Recurrent Units),
as well as unidirectional and bidirectional recurrent layers.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class ActDropNormCNN1D(nn.Module):
    """
    Combines activation, dropout and layer normalization for CNN outputs.
    
    This module applies the following operations in sequence:
    1. Layer normalization to stabilize training
    2. GELU activation function (clipped ReLU)
    3. Dropout for regularization
    
    Args:
        n_feats (int): Number of input features/channels
        dropout (float): Dropout probability
        keep_shape (bool): If True, restore original shape after operations
    """
    def __init__(self, n_feats, dropout, keep_shape=False):
        super(ActDropNormCNN1D, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, features, time)
        
        Returns:
            Tensor: Processed tensor, same shape as input if keep_shape=True,
                   otherwise transposed to (batch, time, features)
        """
        x = x.transpose(1, 2)
        x = self.dropout(F.gelu(self.norm(x)))
        if self.keep_shape:
            return x.transpose(1, 2)
        else:
            return x


class SpeechRecognition(nn.Module):
    """
    End-to-end speech recognition model using CNN and RNN/GRU layers.
    
    The model processes log-mel spectrograms and outputs character probabilities.
    It supports various configurations including different numbers of layers,
    hidden sizes, and recurrent cell types.
    
    Default hyperparameters are defined in the hyperParams dictionary.
    """
    
    hyperParams = {
        "num_classes": 29,  # Characters a-z, space, apostrophe, blank
        "n_feats": 81,      # Input feature dimensions (frequency bins)
        "dropout": 0.1,     # Dropout probability
        "hidden_size": 1024,  # Size of hidden layers
        "num_layers": 1     # Number of recurrent layers
    }

    def __init__(self, 
                 hidden_size=hyperParams["hidden_size"], 
                 num_classes=hyperParams["num_classes"], 
                 n_feats=hyperParams["n_feats"], 
                 num_layers=hyperParams["num_layers"], 
                 dropout=hyperParams["dropout"]):
        """
        Initialize the speech recognition model.
        
        Args:
            hidden_size (int): Size of the hidden layers
            num_classes (int): Number of output classes (characters + blank)
            n_feats (int): Number of input features (frequency bins)
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout probability
        """
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Convolutional feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 10, 2, padding=10//2),
            ActDropNormCNN1D(n_feats, dropout),
        )
        
        # Dense feature refinement
        self.dense = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Recurrent sequence modeling (bidirectional)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0.0,
                            bidirectional=False)
        
        # Final classification layers
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)

    def _init_hidden(self, batch_size):
        """
        Initialize hidden state for recurrent layers.
        
        Args:
            batch_size (int): Batch size
            
        Returns:
            tuple: (hidden state, cell state) for LSTM initialization
        """
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*1, batch_size, hs),
                torch.zeros(n*1, batch_size, hs))

    def forward(self, x, hidden):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor of shape (batch, channel, time, freq)
            hidden (tuple): Initial hidden state for recurrent layer
            
        Returns:
            tuple: (output, hidden_state)
                - output: Tensor of shape (time, batch, num_classes)
                - hidden_state: Final hidden state of recurrent layer
        """
        x = x.squeeze(1)  # Remove channel dimension, shape: (batch, freq, time)
        
        # Apply CNN feature extraction
        x = self.cnn(x)  # Shape: (batch, freq, time/2)
        
        # Apply dense layers for feature refinement
        x = self.dense(x)  # Shape: (batch, time/2, 128)
        
        # Prepare for recurrent processing
        x = x.transpose(0, 1)  # Shape: (time/2, batch, 128)
        
        # Apply recurrent layer
        out, (hn, cn) = self.lstm(x, hidden)
        
        # Final processing
        x = self.dropout2(F.gelu(self.layer_norm2(out)))  # Shape: (time/2, batch, hidden_size)
        
        # Project to output classes
        return self.final_fc(x), (hn, cn)