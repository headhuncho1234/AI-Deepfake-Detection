import os
from tensorflow import keras

# Create a dummy generator to check class indices
if os.path.exists('data/seed'):
    datagen = keras.preprocessing.image.ImageDataGenerator()
    generator = datagen.flow_from_directory(
        'data/seed', 
        target_size=(224,224), 
        batch_size=1, 
        class_mode='binary'
    )
    print(f"Class Indices: {generator.class_indices}")
else:
    print("data/seed directory not found")

