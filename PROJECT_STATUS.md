# Project Status & Architecture

## Overview
This project implements a Deepfake Detection System using Transfer Learning. The current implementation features a fully functional baseline classifier, a robust data processing pipeline, and specific optimizations for high-end consumer GPUs (RTX 3090).

## 1. Core Detection Model
**Location:** `models/train_cnn.py`

The system uses a Transfer Learning approach based on **EfficientNetB0**:
- **Base:** EfficientNetB0 (pre-trained on ImageNet), frozen during initial training.
- **Head:** Custom classification head designed for binary classification (Real vs. Fake):
  - `GlobalAveragePooling2D`
  - `Dropout` (0.5) for regularization
  - `Dense` (512 units, ReLU activation)
  - `Dropout` (0.3)
  - `Output` (1 unit, Sigmoid activation)
- **Optimization:** 
  - Uses TensorFlow's `mixed_float16` policy for ~2x training speedup on RTX series GPUs.
  - Configured for memory growth to prevent VRAM allocation issues.

## 2. Data Pipeline
**Location:** `scripts/` & `data/`

The project utilizes a structured data pipeline to manage datasets for Active Learning:

### Scripts
- **`split_dataset.py`**: Partitions raw data into three distinct sets:
  - `seed`: Initial training set.
  - `validation`: Set for model tuning and early stopping.
  - `pool`: Large pool of unlabeled images intended for future Active Learning selection.
- **`analyze_dataset.py`**: Analyzes class distribution to ensure balance between 'Real' and 'Fake' samples.
- **`check_indices.py`**: Validates data integrity.

### Directory Structure
Data is organized for Keras `flow_from_directory`:
```
data/
├── seed/       # Used for training baseline
├── validation/ # Used for validation
└── pool/       # Candidate images for RL Agent
```

## 3. Training & Evaluation
The training workflow is automated with the following features:
- **Data Augmentation:** Rotation (20°) and horizontal flips applied during training.
- **Callbacks:**
  - `EarlyStopping`: Monitors validation loss (patience=3).
  - `ModelCheckpoint`: Saves only the best performing model to `models/trained_models/baseline_best.keras`.
- **Inference:**
  - `evaluate_model.py`: Calculates aggregate metrics on test sets.
  - `test_prediction.py`: Performs single-image inference for sanity checks.

## 4. Roadmap & Future Modules
The "RL" (Reinforcement Learning) and Explainability components are currently initialized but not implemented:

- **RL Agent** (`rl_agent/`): 
  - *Goal:* Implement an agent to select the most informative samples from the `pool` directory to retrain the model (Active Learning).
  - *Status:* Placeholder created.

- **Explainability** (`explainability/`):
  - *Goal:* Implement visualization tools (e.g., Grad-CAM) to understand model focus areas.
  - *Status:* Placeholder created.

