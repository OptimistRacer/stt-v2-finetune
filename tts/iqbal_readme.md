---
dataset_info:
  features:
  - name: filename
    dtype: string
  - name: text
    dtype: string
  - name: audio
    dtype: audio
  splits:
  - name: train
    num_bytes: 58271682.0
    num_examples: 353
  download_size: 58203806
  dataset_size: 58271682.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
# Urdu Audio Dataset

## Overview

This dataset contains Urdu audio files along with their transcriptions. It is designed to be used for training and evaluating automatic speech recognition (ASR) models. The audio files are processed and segmented based on silence to facilitate training and fine-tuning of speech models.

## Dataset Details

- **Dataset Name**: urdu-tts
- **Username**: muhammadsaadgondal
- **Version**: 1.1
- **Languages**: Urdu
- **Content**: Audio files and their corresponding transcriptions

## Usage

To load the dataset using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset('muhammadsaadgondal/urdu-tts')

```

# Urdu Audio Dataset

## Overview

This dataset contains Urdu audio files along with their transcriptions. It is designed to be used for training and evaluating automatic speech recognition (ASR) models. The audio files are processed and segmented based on silence to facilitate training and fine-tuning of speech models.

## Dataset Details

- **Dataset Name**: urdu-tts
- **Username**: muhammadsaadgondal
- **Version**: 1.1
- **Languages**: Urdu
- **Content**: Audio files and their corresponding transcriptions


## Citation

If you use this dataset in your research or projects, please cite it as follows:
@dataset{your_citation_key,
- **author**    = Muhammad Saad Gondal,
- **title**     = urdu-tts,
- **year**      = {2024},
- **publisher** = muhammadsaadgondal,
- **url**       = {https://huggingface.co/datasets/muhammadsaadgondal/urdu-tts}
  }