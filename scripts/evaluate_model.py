import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate():
    model_path = 'models/trained_models/baseline_best.keras'
    val_dir = 'data/validation'

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    if not os.path.exists(val_dir):
        print(f"Validation data not found at {val_dir}")
        return

    print("Loading model...")
    model = keras.models.load_model(model_path)

    print("Preparing validation data...")
    # Important: shuffle=False to match predictions with filenames/labels
    val_gen = keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    print("Running predictions...")
    # Get probabilities
    predictions = model.predict(val_gen, verbose=1)
    
    # Convert to binary classes (0 or 1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    # Get true labels
    true_classes = val_gen.classes
    class_labels = list(val_gen.class_indices.keys()) # ['fake', 'real']

    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"(True Neg: {cm[0][0]}, False Pos: {cm[0][1]})")
    print(f"(False Neg: {cm[1][0]}, True Pos: {cm[1][1]})")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels, digits=4))

if __name__ == "__main__":
    evaluate()

