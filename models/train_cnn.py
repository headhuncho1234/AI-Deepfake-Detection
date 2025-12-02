import tensorflow as tf
from tensorflow import keras
import os

# RTX 3090 Optimization
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