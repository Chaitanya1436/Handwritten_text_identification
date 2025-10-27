import os
import cv2
import numpy as np
import yaml
from keras.models import load_model
from mltu.utils.text_utils import ctc_decoder
from keras import config

# CONFIGURATION
timestamp = "202510262029"
model_path = f"Models/04_sentence_recognition/{timestamp}/model.keras"
configs_path = f"Models/04_sentence_recognition/{timestamp}/configs.yaml"
test_images_folder = "my_test_images"

with open(configs_path, "r") as f:
    configs = yaml.safe_load(f)
vocab = configs["vocab"]

config.enable_unsafe_deserialization()
model = load_model(model_path, safe_mode=False, compile=False)

def preprocess_image(image_path, width=1408, height=96):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    scale = height / h
    new_w = int(w * scale)
    if new_w > width:
        new_w = width
    resized = cv2.resize(image, (new_w, height))
    padded = np.ones((height, width, 3), dtype=np.uint8) * 255
    padded[:, :new_w] = resized
    image = np.expand_dims(padded, axis=0).astype(np.float32)
    return image

def predict_image(image_path):
    image = preprocess_image(image_path)
    pred = model.predict(image)
    pred_text = ctc_decoder(pred, vocab)[0]
    return pred_text

if __name__ == "__main__":
    for fname in sorted(os.listdir(test_images_folder)):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            image_path = os.path.join(test_images_folder, fname)
            pred_text = predict_image(image_path)
            print(f"\nImage: {image_path}\nPrediction: {pred_text}\n")
