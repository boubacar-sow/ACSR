# ACSR: Automatic Cued Speech Recognition

This repository contains the code for an Automatic Cued Speech Recognition (ACSR) system. The system is designed to recognize cued speech, a visual communication system that combines hand shapes and positions with lip movements to represent speech sounds.

## Project Structure

The project is organized as follows:

```
src/acsr/
├── config.py           # Configuration settings
├── data/               # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset.py      # Dataset classes and functions
│   └── dataloader.py   # DataLoader functions
├── decode.py           # Decoding functions
├── main.py             # Main entry point
├── models/             # Model definitions
│   ├── __init__.py
│   ├── acoustic_model.py  # Acoustic model classes
│   ├── conformer.py    # Conformer block implementation
│   └── language_model.py  # Language model classes
├── train.py            # Training functions
├── utils/              # Utility functions
│   ├── __init__.py
│   ├── decoding_utils.py  # Decoding utilities
│   ├── metrics.py      # Evaluation metrics
│   └── text_processing.py  # Text processing utilities
└── variables.py        # Constants and mappings
```

## Features

- Multi-modal fusion of hand shape, hand position, and lip features
- Joint CTC-Attention model for sequence prediction
- Language model rescoring for improved recognition
- Evaluation metrics for phoneme error rate (PER)

## Usage

### Training

To train a new model:

```bash
python -m src.acsr.main --train
```

### Evaluation

To evaluate a trained model:

```bash
python -m src.acsr.main --evaluate
```

### Rescoring

To evaluate a trained model with language model rescoring:

```bash
python -m src.acsr.main --rescore
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- pandas
- numpy
- jiwer
- wandb

## License

This project is licensed under the MIT License - see the LICENSE file for details.
