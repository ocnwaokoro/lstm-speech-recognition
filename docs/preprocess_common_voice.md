# Test-Train Split Notebook Documentation

## Overview

The `preprocess_common_voice.ipynb` notebook provides utilities for processing the Common Voice dataset and preparing it for model training. This notebook handles the conversion of audio files from MP3 to WAV format and creates training and testing JSON files with appropriate data splitting. It's designed to work with the Mozilla Common Voice dataset structure and prepare it for use with the speech recognition model.

## Notebook Structure

### 1. Data Processing Configuration

```python
import os
import json
import random
import csv
from pydub import AudioSegment

data = []
args = {
    "file_path": "cv-corpus-sample/en/validated.tsv",
    "percent": 20,
    "convert": False,
    "save_json_path": "data/",
}
```

This section:
- Imports necessary libraries for file handling and audio conversion
- Creates a data list to store processed entries
- Configures processing parameters including:
  - Path to the TSV file containing transcription metadata
  - Percentage split for test set (20% by default)
  - Flag to control audio conversion
  - Output directory for JSON files

### 2. File Processing Logic

```python
directory = args["file_path"].rpartition('/')[0]
percent = args["percent"]

with open(args["file_path"]) as f:
    length = sum(1 for line in f)

with open(args["file_path"], newline='') as csvfile: 
    reader = csv.DictReader(csvfile, delimiter='\t')
    index = 1
    if(args["convert"]):
        print(str(length) + "files found")
    for row in reader:  
        file_name = row['path']
        filename = file_name.rpartition('.')[0] + ".wav"
        text = row['sentence']
        if(args["convert"]):
            data.append({
            "key": directory + "/clips/" + filename,
            "text": text
            })
            print("converting file " + str(index) + "/" + str(length) + " to wav", end="\r")
            src = directory + "/clips/" + file_name
            dst = directory + "/clips/" + filename
            sound = AudioSegment.from_mp3(src)
            sound.export(dst, format="wav")
            index = index + 1
        else:
            data.append({
            "key": directory + "/clips/" + file_name,
            "text": text
            })
```

This section:
- Extracts the directory from the file path
- Counts total files for progress tracking
- Reads the TSV file containing metadata
- Processes each entry:
  - Extracts file path and transcript text
  - Creates data entry with file path and transcript
  - Optionally converts MP3 to WAV format if conversion flag is enabled
  - Displays progress during conversion

### 3. Data Shuffling and Splitting

```python
random.shuffle(data)

f = open(args["save_json_path"] +"/"+  "train.json", "w")

with open(args["save_json_path"] +"/"+  'train.json','w') as f:
    d = len(data)
    i=0
    while(i<int(d - d/percent)):
        r=data[i]
        line = json.dumps(r)
        f.write(line + "\n")
        i = i+1
        
f = open(args["save_json_path"] +"/"+  "test.json", "w")
with open(args["save_json_path"] +"/"+  'test.json','w') as f:
    d = len(data)
    i=int(d-d/percent)
    while(i<d):
        r=data[i]
        line = json.dumps(r)
        f.write(line + "\n")
        i = i+1
```

This section:
- Shuffles the data to ensure random distribution
- Calculates split point based on configured percentage
- Creates and writes to the training JSON file:
  - Writes each data entry as a JSON line
  - Includes all entries up to the split point
- Creates and writes to the testing JSON file:
  - Writes remaining data entries as JSON lines
  - Includes all entries from the split point to the end

## Technical Details

### Data Format

The notebook processes Common Voice data which follows this structure:
- TSV file containing metadata and transcriptions
- Audio clips stored in MP3 format in a "clips" subdirectory
- Each entry contains file path and transcription text

### Output JSON Format

The created JSON files contain one entry per line with this structure:
```json
{"key": "path/to/audio/file.wav", "text": "transcription of the audio"}
```

This format is compatible with the `Data` class used in the training notebooks:
- The "key" points to the audio file location
- The "text" contains the ground truth transcription

### Audio Conversion

When the conversion flag is enabled:
- MP3 files are converted to WAV format
- WAV files maintain the same base filename in the same directory
- The PyDub library handles the conversion process
- Progress is displayed during conversion

## Usage Guidelines

### Dataset Preparation

1. Download the Common Voice dataset
2. Extract the dataset to a directory
3. Update the `args` dictionary with appropriate paths
4. Run the notebook to process the data

### Configuration Options

- `file_path`: Path to the validated.tsv file in the Common Voice dataset
- `percent`: Percentage of data to allocate to test set (20% recommended)
- `convert`: Boolean flag to control audio conversion (set to True for first run)
- `save_json_path`: Directory to save the output JSON files

### Execution Flow

1. Set the configuration parameters in the `args` dictionary
2. Run the notebook cells in sequence
3. Check the output directory for the created JSON files
4. Verify that files are created with proper entry counts

### Expected Outputs

After successful execution:
- `train.json`: Contains approximately 80% of the data entries
- `test.json`: Contains approximately 20% of the data entries
- If conversion was enabled, WAV files will be created in the clips directory

## Common Issues and Solutions

1. **Missing Audio Files**
   - Ensure the Common Voice dataset is fully extracted
   - Verify the path structure matches the expected format
   - Check file permissions for the clips directory

2. **Conversion Errors**
   - Install the PyDub dependencies (FFmpeg)
   - Ensure sufficient disk space for WAV files (larger than MP3)
   - Check if source MP3 files are valid and not corrupted

3. **Memory Issues**
   - For very large datasets, consider processing in batches
   - Increase available system memory
   - Process a subset of the dataset by modifying the reader loop

4. **Path Issues**
   - Use absolute paths if relative paths cause problems
   - Ensure proper directory separators for your operating system
   - Create the output directory if it doesn't exist

5. **Performance Considerations**
   - Audio conversion is CPU-intensive
   - Converting large datasets may take significant time
   - Once converted, set the `convert` flag to False for subsequent runs