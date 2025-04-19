"""
utils.py - Text Processing Utilities for Speech Recognition

This module implements text processing utilities for the speech recognition system,
including character-to-index mapping, text normalization, and sequence conversion
between text and integer representations.

The main component is the TextProcess class which handles:
1. Character set definition (alphabet, space, apostrophe, blank)
2. Bidirectional mapping between characters and integer indices
3. Text normalization (lowercase, punctuation removal)
4. Conversion between text and integer sequences for model training and inference

These utilities are essential for preparing text data for CTC (Connectionist Temporal
Classification) training and for decoding model outputs back to readable text.
"""

import string
import torch
import numpy as np

class TextProcess:
    """
    Handles text processing for speech recognition, including character-to-index mapping
    and sequence conversion.
    
    The class defines a character set consisting of:
    - Lowercase English alphabet (a-z)
    - Space character
    - Apostrophe
    - Blank token (for CTC)
    
    It provides bidirectional mapping between characters and integer indices,
    which is used for encoding text labels for training and decoding model outputs.
    """
    
    def __init__(self):
        """
        Initialize the text processor with character set and mappings.
        
        Defines the character set and creates mappings between characters and indices.
        The character set includes lowercase letters, space, apostrophe, and blank token.
        """
        # Define the character set
        self.char_map = {}
        self.index_map = {}
        
        # Add space
        self.char_map[' '] = 0
        self.index_map[0] = ' '
        
        # Add alphabet (a-z)
        for c in string.ascii_lowercase:
            self.char_map[c] = len(self.char_map)
            self.index_map[len(self.index_map)] = c
            
        # Add apostrophe
        self.char_map["'"] = len(self.char_map)
        self.index_map[len(self.index_map)] = "'"
        
        # Add blank token (used by CTC)
        self.blank_index = len(self.char_map)
        
    def text_to_int_sequence(self, text):
        """
        Convert a text string to a sequence of integer indices.
        
        This is used to convert text transcriptions to the format expected by the model
        during training.
        
        Args:
            text (str): Input text string to convert
            
        Returns:
            list: Sequence of integer indices corresponding to each character
        """
        int_sequence = []
        
        # Convert each character to its corresponding index
        for c in text:
            if c in self.char_map:
                int_sequence.append(self.char_map[c])
            else:
                # Skip characters not in our character set
                pass
                
        return int_sequence
        
    def int_to_text_sequence(self, int_sequence):
        """
        Convert a sequence of integer indices to a text string.
        
        This is used to convert model predictions back to readable text.
        Handles CTC decoding by removing repeated characters that are not
        separated by the blank token.
        
        Args:
            int_sequence (list): Sequence of integer indices
            
        Returns:
            str: Decoded text string
        """
        text_sequence = ""
        previous_index = -1
        
        # Convert each index to its corresponding character
        for index in int_sequence:
            # Skip if it's the blank token
            if index == self.blank_index:
                previous_index = -1
                continue
                
            # Skip repeated characters (CTC decoding)
            if index != previous_index:
                if index in self.index_map:
                    text_sequence += self.index_map[index]
                    
            previous_index = index
            
        return text_sequence
    
    def normalize_text(self, text):
        """
        Normalize text for processing.
        
        Performs operations like:
        - Converting to lowercase
        - Removing punctuation (except apostrophes)
        - Standardizing whitespace
        
        Args:
            text (str): Input text to normalize
            
        Returns:
            str: Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Keep only characters in our character set
        result = ""
        for c in text:
            if c in self.char_map:
                result += c
                
        return result