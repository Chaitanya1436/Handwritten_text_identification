import os
import numpy as np
from keras.models import load_model
from mltu.utils.text_utils import ctc_decoder
from configs import ModelConfigs
from PIL import Image
import keras

# ======== CONFIGS ========
configs = ModelConfigs()
configs.model_path = r"Models/04_sentence_recognition\202510262029"
model_path = os.path.join(configs.model_path, "model.keras")

# Enable unsafe deserialization because model has Lambda layers
keras.config.enable_unsafe_deserialization()

# Load model
model = load_model(model_path, compile=False)

# ======== TEST FOLDER ========
test_folder = "my_test_images"

# Get all image files
image_files = sorted([
    f for f in os.listdir(test_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

# ======== PREDICTION ========
for image_name in image_files:
    image_path = os.path.join(test_folder, image_name)
    
    # Open image as PIL.Image
    pil_image = Image.open(image_path).convert("RGB")
    
    # Convert to numpy array and normalize to [0,1]
    image_array = np.array(pil_image).astype(np.float32) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict
    prediction = model.predict(image_array)
    
    # Decode CTC output
    decoded_text = ctc_decoder(prediction, configs.vocab)[0]
    
    # Print result
    print(f"Image: {image_path}")
    print(f"Prediction: {decoded_text}")
    print("-" * 50)
