# Practical Guide to Speech Recognition System

This guide provides practical information and examples for working with the speech recognition system, covering common use cases, troubleshooting, and best practices.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Model Selection](#model-selection)
4. [Training Best Practices](#training-best-practices)
5. [Inference Guide](#inference-guide)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ocnwaokoro/speech-recognition.git
cd speech-recognition

# Install dependencies
pip install -r requirements.txt
```

### Run Inference on an Audio File

```bash
python inference.py --audio-path samples/audio.wav --checkpoint-path models/model_latest.pt
```

### Train a Model (using pre-trained weights)

```bash
python train.py --train-json data/train.json --valid-json data/test.json --checkpoint models/pretrained.pt
```

## Data Preparation

### Data Format

The system expects data in JSON format with the following structure:

```json
{"key": "/path/to/audio.wav", "text": "transcription of the audio"}
```

### Preparing Common Voice Dataset

Use the `preprocess_common_voice.ipynb` notebook or `split_data.py` script:

```bash
python scripts/split_data.py --file_path path/to/validated.tsv --percent 20 --save_json_path data/
```

This will:
1. Process the Common Voice TSV file
2. Split data into training (80%) and test (20%) sets
3. Save JSON files in the specified directory

### Audio Format Considerations

- Sample rate: 8 kHz (will be resampled if different)
- Format: 16-bit PCM WAV files
- Duration: Ideally 1-15 seconds per audio file
- For best results, normalize audio volume before processing

### Data Augmentation

The system implements SpecAugment for data augmentation during training:
- Frequency masking: Masks random frequency bands
- Time masking: Masks random time steps

To enable/disable augmentation, modify the data loader settings in `data_loader.py`.

## Model Selection

### Available Architectures

The system supports several architectural variations:

| Architecture | Description | Strengths | Use Case |
|--------------|-------------|-----------|----------|
| 5-layer, 1 RNN | Smallest model (18M parameters) | Fast inference, limited accuracy | Resource-constrained environments |
| 9-layer, 7 RNN | Medium model (70M parameters) | Good balance of speed/accuracy | General purpose |
| 11-layer, 7 RNN + 2D Conv | Large model (100M parameters) | Best accuracy, slower inference | When accuracy is critical |

### Choosing Based on Your Needs

- **Speed is critical**: Use 5-layer model with simple RNN
- **Accuracy is critical**: Use 11-layer model with 2D convolution
- **Balanced approach**: Use 9-layer model with 7 RNN layers
- **Deployment to edge devices**: Use 5-layer model with unidirectional RNN
- **Handling noisy audio**: Use models with 2D convolution

## Training Best Practices

### Hardware Requirements

- **Minimum**: Single GPU with 8GB VRAM
- **Recommended**: Multi-GPU system with 16GB+ VRAM per GPU
- **CPU RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: Fast SSD for data loading

### Hyperparameter Selection

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| Batch Size | 32-128 | Larger is better, limited by GPU memory |
| Learning Rate | 1e-4 to 6e-4 | Start with 3e-4 and adjust |
| Optimizer | AdamW | Better convergence than SGD |
| LR Schedule | ReduceLROnPlateau | patience=6, factor=0.5 |

### Training Process

1. **Start with SortaGrad**: Enable curriculum learning for first epoch
2. **Monitor validation loss**: Watch for signs of overfitting
3. **Save checkpoints**: Save periodically to prevent data loss
4. **Use BatchNorm**: Significantly improves convergence
5. **Track metrics**: Monitor WER/CER on validation set

### Example Training Command

```bash
python train.py \
  --train-json data/train.json \
  --valid-json data/test.json \
  --batch-size 64 \
  --epochs 100 \
  --learning-rate 3e-4 \
  --checkpoint models/pretrained.pt \
  --save-dir models/
```

## Inference Guide

### Basic Usage

```bash
python inference.py --audio-path audio.wav --checkpoint-path model.pt
```

### Batch Processing

Process multiple files at once:

```python
from inference import load_audio, predict
import torch
import glob
import os

# Load model
model = SpeechRecognition().to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Process files
audio_files = glob.glob("audio_dir/*.wav")
for audio_path in audio_files:
    audio_tensor = load_audio(audio_path, LogMelSpec())
    pred_indices = predict(model, audio_tensor)
    decoded = TextProcess().int_to_text_sequence([i for i in pred_indices if i != 28])
    print(f"{os.path.basename(audio_path)}: {decoded}")
```

### Language Model Integration

The system supports n-gram language model integration during inference:

```bash
python inference.py --audio-path audio.wav --checkpoint-path model.pt --lm-path lm/5gram.binary --alpha 0.75 --beta 0.1
```

Where:
- `--lm-path`: Path to KenLM language model
- `--alpha`: Language model weight
- `--beta`: Word insertion bonus

### Optimizing for Speed vs Accuracy

- **Speed focus**: Use smaller beam size (e.g., 50)
- **Accuracy focus**: Use larger beam size (e.g., 500)
- **Balanced**: Use beam size of 200 with pruned language model

## Performance Optimization

### Training Optimizations

1. **Data loading**: Use SSD storage and increase `num_workers`
2. **Mixed precision**: Enable AMP for faster training
3. **Gradient accumulation**: For larger effective batch sizes
4. **Synchronous SGD**: For multi-GPU training

### Inference Optimizations

1. **Half precision**: Use FP16 for faster inference
2. **Batch Dispatch**: Process multiple requests together
3. **Beam search pruning**: Limit active tokens
4. **Row convolution**: For streaming inference

### Memory Optimization

1. **Stride factor**: Increase stride to reduce sequence length
2. **Gradient checkpointing**: Trade compute for memory
3. **Model pruning**: Remove unnecessary weights
4. **Memory-efficient attention**: For long audio files

## Troubleshooting

### Common Issues

1. **Out of memory during training**
   - Reduce batch size
   - Increase stride in first convolutional layer
   - Use gradient checkpointing

2. **Poor transcription quality**
   - Check audio quality and format
   - Ensure model was trained on similar data
   - Adjust language model weights
   - Use models with 2D convolution for noisy audio

3. **Slow inference**
   - Use half precision (FP16)
   - Reduce beam search width
   - Use faster language model implementation
   - Implement Batch Dispatch for multiple requests

4. **Model fails to converge**
   - Verify data quality and format
   - Implement SortaGrad curriculum learning
   - Use BatchNorm in all layers
   - Start with smaller learning rate

### Debugging Tips

1. Test on simple examples first
2. Check audio preprocessing pipeline
3. Visualize spectrograms to verify input
4. Trace common errors with sample utterances
5. Compare output distributions to expected values

## Advanced Usage

### Customizing the Architecture

Modify `model.py` to customize the architecture:

```python
# Example: Deeper network with more hidden units
model = SpeechRecognition(
    hidden_size=2048,  # Larger hidden size
    num_layers=3,      # More recurrent layers
    dropout=0.2        # Increased regularization
)
```

### Streaming Inference

For low-latency applications, use row convolution:

```python
# In model.py
self.row_conv = RowConvolution(hidden_size, context=19)
```

### Transfer Learning

Fine-tune a pre-trained model on new data:

```python
# Load pre-trained model
model = SpeechRecognition()
model.load_state_dict(torch.load("pretrained.pt"))

# Freeze feature extraction layers
for param in model.cnn.parameters():
    param.requires_grad = False
    
# Train with new data
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
```

### Multi-Language Support

For multi-language models:

1. Increase output classes to include all character sets
2. Train on combined dataset with language tags
3. Use language-specific language models during inference