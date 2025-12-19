import sys
import os

# Import TensorFlow with a helpful fallback if it's not installed.
try:
    import tensorflow as tf
    from tensorflow import keras
except ModuleNotFoundError:
    sys.stderr.write("\nERROR: TensorFlow is not installed in the active Python environment.\n\n")
    sys.stderr.write("Quick steps to get started (zsh):\n")
    sys.stderr.write("  python3 -m venv .venv\n")
    sys.stderr.write("  source .venv/bin/activate\n")
    sys.stderr.write("  pip install -U pip\n")
    sys.stderr.write("  pip install -r requirements.txt\n\n")
    sys.stderr.write("If you're on Apple Silicon (M1/M2/Pro/Max), prefer the macOS-specific builds:\n")
    sys.stderr.write("  pip install tensorflow-macos tensorflow-metal\n\n")
    sys.stderr.write("Note: macOS 12+ and Python 3.8–3.11 are commonly required for these wheels.\n")
    sys.stderr.write("Using a virtual environment or Miniforge/conda is recommended to avoid\n")
    sys.stderr.write("system-level package conflicts. See the TensorFlow macOS docs for details.\n\n")
    sys.stderr.write("If you plan to run on a Linux machine with an NVIDIA GPU, install the appropriate\n")
    sys.stderr.write("TensorFlow build and GPU drivers (CUDA/cuDNN) for best performance.\n\n")
    sys.exit(1)

# RTX 3090 Optimization (only applicable on machines with an NVIDIA GPU)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    keras.mixed_precision.set_global_policy('mixed_float16') # 2x Speedup on RTX cards

def create_model():
    base_model = keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    return keras.Model(inputs, outputs)

def train_baseline():
    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Batch size 64 for 24GB VRAM
    # Note: EfficientNetB0 expects [0, 255] inputs, so we do NOT use rescale=1./255
    train_gen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20, horizontal_flip=True
    ).flow_from_directory('data/seed', target_size=(224,224), batch_size=64, class_mode='binary')
    
    val_gen = keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
        'data/validation', target_size=(224,224), batch_size=64, class_mode='binary')
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('models/trained_models/baseline_best.keras', save_best_only=True)
    ]
    
    # workers argument removed for Keras 3 compatibility
    model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=callbacks)
    model.save('models/trained_models/baseline_final.keras')

if __name__ == "__main__":
    os.makedirs('models/trained_models', exist_ok=True)
    train_baseline()