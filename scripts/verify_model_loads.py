from tensorflow import keras

model = keras.models.load_model('models/trained_models/baseline_best.keras')
print(f"Model loaded successfully!")
print(f"Number of layers: {len(model.layers)}")