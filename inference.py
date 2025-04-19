"""
inference.py - Speech Recognition Inference Script

This script loads a trained speech recognition model and performs inference on an audio file.
It converts the audio to log-mel spectrograms, runs it through the model, and transcribes
the speech to text.

Usage:
    python inference.py --audio-path /path/to/audio.wav --checkpoint-path /path/to/model.pt

Arguments:
    --audio-path: Path to the WAV audio file to transcribe
    --checkpoint-path: Path to the trained model checkpoint

Example:
    python inference.py --audio-path examples/hello.wav --checkpoint-path models/speech_model_v1.pt
"""

import argparse
import torch
import torchaudio
import torch.nn.functional as F
from src.model import SpeechRecognition
from src.data_loader import LogMelSpec
from utils import TextProcess

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(filepath, featurizer):
    """
    Load and process an audio file for inference.
    
    Args:
        filepath (str): Path to the audio file
        featurizer (LogMelSpec): Feature extractor instance
        
    Returns:
        torch.Tensor: Processed audio tensor with batch dimension
    """
    waveform, _ = torchaudio.load(filepath)
    features = featurizer(waveform)
    return features.unsqueeze(0)  # add batch dimension

def predict(model, audio_tensor):
    """
    Run inference on processed audio using the model.
    
    Args:
        model (SpeechRecognition): Trained speech recognition model
        audio_tensor (torch.Tensor): Processed audio tensor
        
    Returns:
        list: Sequence of predicted character indices
    """
    model.eval()
    with torch.no_grad():
        bs = audio_tensor.size(0)
        hidden = model._init_hidden(bs)
        hidden = (hidden[0].to(device), hidden[1].to(device))
        output, _ = model(audio_tensor.to(device), hidden)
        output = F.log_softmax(output, dim=2)
        prediction = torch.argmax(output, dim=2).squeeze().tolist()
        return prediction

def main(audio_path, checkpoint_path):
    """
    Main inference function.
    
    Args:
        audio_path (str): Path to the audio file
        checkpoint_path (str): Path to the model checkpoint
    """
    print(f"🔍 Loading model from: {checkpoint_path}")
    model = SpeechRecognition().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print(f"🎧 Processing audio file: {audio_path}")
    featurizer = LogMelSpec()
    audio_tensor = load_audio(audio_path, featurizer)

    print("📤 Generating prediction...")
    pred_indices = predict(model, audio_tensor)
    # Filter out blank tokens (index 28)
    decoded = TextProcess().int_to_text_sequence([i for i in pred_indices if i != 28])
    
    print("✅ Transcription:")
    print(decoded)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single audio file.")
    parser.add_argument("--audio-path", type=str, required=True, help="Path to .wav audio file")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model .pt checkpoint")
    args = parser.parse_args()
    main(args.audio_path, args.checkpoint_path)