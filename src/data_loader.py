"""
data_loader.py - Audio Data Processing for Speech Recognition

This module implements data processing pipelines for speech recognition, including:
1. Audio feature extraction (spectrogram, log-mel features)
2. Data augmentation (SpecAugment with frequency and time masking)
3. Dataset loading and batch preparation

The primary components are:
- LogMelSpec: Transforms raw audio into log-mel spectrograms
- SpecAugment: Performs frequency and time masking for data augmentation
- Data: Custom PyTorch Dataset for loading and processing speech data
- collate_fn_padd: Custom collate function for handling variable-length sequences

This implementation is designed to work with the Mozilla Common Voice dataset 
and similar speech corpora that provide audio files with text transcriptions.
"""

import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import TextProcess

class SpecAugment(nn.Module):
    """
    Implements SpecAugment for data augmentation in speech recognition.
    
    SpecAugment applies masking operations to the spectrogram, specifically:
    1. Frequency masking: Masks contiguous frequency bands
    2. Time masking: Masks contiguous time steps
    
    This implementation supports three different masking policies with
    varying levels of augmentation strength.
    
    Reference:
    Park, D. S., et al. "SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition." (2019)
    
    Args:
        rate (float): Probability of applying augmentation
        policy (int): Augmentation policy (1=mild, 2=medium, 3=aggressive)
        freq_mask (int): Maximum width of frequency masks
        time_mask (int): Maximum width of time masks
    """
    def __init__(self, rate, policy=3, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()

        self.rate = rate

        # Policy 1: Apply one frequency and one time mask
        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        # Policy 2: Apply two frequency and two time masks
        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        # Map policy numbers to corresponding methods
        policies = { 1: self.policy1, 2: self.policy2, 3: self.policy3 }
        self._forward = policies[policy]

    def forward(self, x):
        """
        Apply SpecAugment based on the selected policy.
        
        Args:
            x (Tensor): Input spectrogram of shape (batch, channel, freq, time)
            
        Returns:
            Tensor: Augmented spectrogram with the same shape as input
        """
        return self._forward(x)

    def policy1(self, x):
        """
        Policy 1: Apply augmentation with probability self.rate
        Uses one frequency mask and one time mask
        """
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug(x)
        return x

    def policy2(self, x):
        """
        Policy 2: Apply augmentation with probability self.rate
        Uses two frequency masks and two time masks
        """
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug2(x)
        return x

    def policy3(self, x):
        """
        Policy 3: Randomly choose between policy1 and policy2
        This is the most aggressive augmentation policy
        """
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)

class LogMelSpec(nn.Module):
    """
    Converts raw audio waveforms to log-mel spectrograms.
    
    This transformation is commonly used as the input representation for
    speech recognition models, as it better represents the frequency characteristics
    relevant to human speech perception.
    
    Args:
        sample_rate (int): Audio sample rate in Hz
        n_mels (int): Number of mel bands
        win_length (int): Window length in samples
        hop_length (int): Hop length in samples (stride between windows)
    """
    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
                            sample_rate=sample_rate, n_mels=n_mels,
                            win_length=win_length, hop_length=hop_length)

    def forward(self, x):
        """
        Convert audio waveform to log-mel spectrogram.
        
        Args:
            x (Tensor): Audio waveform of shape (batch, time)
            
        Returns:
            Tensor: Log-mel spectrogram of shape (batch, n_mels, time)
        """
        x = self.transform(x)  # mel spectrogram
        x = np.log(x + 1e-14)  # logarithmic scaling with small offset to avoid log(0)
        return x

def get_featurizer(sample_rate, n_feats=81):
    """
    Factory function to create a mel spectrogram feature extractor.
    
    Args:
        sample_rate (int): Audio sample rate in Hz
        n_feats (int): Number of mel frequency bands
        
    Returns:
        LogMelSpec: Configured feature extractor
    """
    return LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)

class Data(torch.utils.data.Dataset):
    """
    Dataset for loading and processing speech data.
    
    This dataset:
    1. Loads audio files and transcriptions from a JSON file
    2. Applies feature extraction (log-mel spectrogram)
    3. Optionally applies data augmentation (SpecAugment)
    4. Handles text preprocessing
    
    Args:
        json_path (str): Path to JSON file with audio paths and transcriptions
        sample_rate (int): Audio sample rate
        n_feats (int): Number of mel frequency bands
        specaug_rate (float): Probability of applying SpecAugment
        specaug_policy (int): SpecAugment policy to use
        time_mask (int): Maximum width of time masks for SpecAugment
        freq_mask (int): Maximum width of frequency masks for SpecAugment
        valid (bool): If True, disables data augmentation (for validation set)
        shuffle (bool): If True, shuffles the dataset
        text_to_int (bool): If True, converts text to integer indices
        log_ex (bool): If True, logs exceptions during sample loading
    """
    
    parameters = {
        "sample_rate": 8000, "n_feats": 81,
        "specaug_rate": 0.5, "specaug_policy": 3,
        "time_mask": 70, "freq_mask": 15 
    }

    def __init__(self, 
                 json_path, 
                 sample_rate=parameters["sample_rate"], 
                 n_feats=parameters["n_feats"], 
                 specaug_rate=parameters["specaug_rate"], 
                 specaug_policy=parameters["specaug_policy"],
                 time_mask=parameters["time_mask"], 
                 freq_mask=parameters["freq_mask"], 
                 valid=False, 
                 shuffle=True, 
                 text_to_int=True, 
                 log_ex=True):
        
        self.log_ex = log_ex
        self.text_process = TextProcess()
        self.data = pd.read_json(json_path, lines=True)
        
        # Configure audio transformations - with or without augmentation
        if valid:
            # No augmentation for validation set
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)
            )
        else:
            # Apply augmentation for training set
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80),
                SpecAugment(specaug_rate, specaug_policy, freq_mask, time_mask)
            )
            
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            tuple: (spectrogram, label, spectrogram_length, label_length)
            
        Raises:
            Exception: If processing fails and returns the next item instead
        """
        if torch.is_tensor(idx):
            idx = idx.item()

        try:
            # Load audio file
            file_path = self.data.key.iloc[idx]
            waveform, _ = torchaudio.load(file_path)
            
            # Prepare label sequence
            label = self.text_process.text_to_int_sequence(str.lower(self.data.text.iloc[idx]))
            
            # Convert audio to spectrogram features
            spectrogram = self.audio_transforms(waveform)  # (channel, feature, time)
            
            # Calculate lengths
            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)
            
            # Validation checks
            if spec_len < label_len:
                raise Exception('spectrogram len is smaller than label len')
            if spectrogram.shape[0] > 1:
                raise Exception('dual channel, skipping audio file %s' % file_path)
            if spectrogram.shape[2] > 1650:
                pass
                # raise Exception('spectrogram too big. size %s' % spectrogram.shape[2])
            if label_len == 0:
                raise Exception('label len is zero... skipping %s' % file_path)
                
        except Exception as e:
            if self.log_ex:
                print(str(e), file_path)  
            # Recursively try the next item on error
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)  
            
        return spectrogram, label, spec_len, label_len

    def describe(self):
        """Return descriptive statistics about the dataset"""
        return self.data.describe()


def collate_fn_padd(data):
    """
    Custom collate function for handling variable-length sequences in batches.
    
    This function:
    1. Pads spectrograms and labels to have the same length within a batch
    2. Transposes spectrograms to the format expected by the model
    3. Collects length information for CTC loss calculation
    
    Args:
        data (list): List of tuples (spectrogram, label, spec_len, label_len)
        
    Returns:
        tuple: (padded_spectrograms, padded_labels, input_lengths, label_lengths)
    """
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    
    # Process each sample in the batch
    for (spectrogram, label, input_length, label_length) in data:
        if spectrogram is None:
            continue
            
        # Squeeze channel dimension and transpose to (time, frequency)
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
        labels.append(torch.Tensor(label))
        input_lengths.append(input_length)
        label_lengths.append(label_length)

    # Pad sequences to the same length
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return spectrograms, labels, input_lengths, label_lengths