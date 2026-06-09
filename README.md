# wordLSTM

A LSTM model for word-level text generation.

This repository is still in a crude state. Further documentation and simplification will be provided soon. 
## Setup

```bash
pip install -r requirements.txt
```

Parameters for various tasks are stored in their respective files.

- `train.py` for training related variables

- `wordLSTM.py` for model configuration

## Usage

**Train the model:**
```bash
python train.py
```

In default configuration, this tends to use around 1.6Gi on an RTX 2060 Super.

On keyboard interrupt or next epoch the model automatically saves to `model.pth`

**Chat/inference:**

Weights by default are loaded from `model.pth`.

```bash
python chat.py
```
Enter 'exit' to exit the chat.

## Requirements

- PyTorch 2.10.0+
- NumPy 2+
- [griot](https://github.com/vertigoaway/griot) - vocabulary utilities
- [trinketbox](https://github.com/vertigoaway/trinketbox) - training/data utilities
- \>2Gi of freed RAM
- ~2Gi of freed VRAM for accelerated training
## Data

Training data should be in CSV format (default: `discordData.csv`). All messages should be confined to a single column.

Modify csvPos variable in train.py to the column number that contains all data.

This data is required to be loaded to initialize the vocabulary currently. 

There is no persistent save/load functionality for vocabulary as of writing this. Altering training data will alter the generated vocabulary making it potentially incompatible with models trained on different data.