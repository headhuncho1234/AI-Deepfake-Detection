# AI-Deepfake-Detection

## Project Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Configuration 

This project uses a specific directory structure for image data. **Note:** The `data/` directory is excluded from version control to save space.

1. Create a `data/` directory in the root of the project.
2. Organize your dataset as follows:

```
data/
├── raw/                # Raw dataset
│   ├── fake/           # Deepfake images
│   └── real/           # Real images
├── pool/               # Pooled images for active learning
│   ├── fake/
│   └── real/
├── validation/         # Validation set images
├── train_labels.csv    # Training labels
├── val_labels.csv      # Validation labels
└── test_labels.csv     # Test labels
```

## Model Management

Trained models are stored in `models/trained_models/`. These files are large binaries and are excluded from git.

- **Saving Models**: Training scripts will automatically save models to this directory.
- **Loading Models**: If you have downloaded pre-trained models or are restoring from a backup, place your `.keras` or `.h5` files into `models/trained_models/`.