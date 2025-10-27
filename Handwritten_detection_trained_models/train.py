import os
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from PIL import Image

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import TrainLogger, Model2onnx
from mltu.tensorflow.metrics import CERMetric, WERMetric
from mltu.annotations.images import CVImage


from model import train_model
from configs import ModelConfigs


# Load Hugging Face IAM-line dataset as streaming
dataset = load_dataset('Teklia/IAM-line', streaming=True)
train_dataset = dataset['train']

print("Preparing dataset from Hugging Face streaming source...")

# Create a temporary directory to save images for the data provider (required by mltu DataProvider)
temp_image_dir = "temp_hf_images"
os.makedirs(temp_image_dir, exist_ok=True)

dataset_list = []
vocab = set()
max_len = 0

# For demonstration, we will load max of 1000 samples (you can increase this)
max_samples = 200
count = 0

for item in tqdm(train_dataset):
    if count >= max_samples:
        break

    # PIL Image from dataset
    pil_img = item['image']
    label = item['text'].replace('|', ' ').strip()

    # Save image locally as PNG with a unique name
    image_path = os.path.join(temp_image_dir, f"hf_sample_{count}.png")
    pil_img.save(image_path)

    dataset_list.append([image_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))
    count += 1

print(f"Loaded {count} samples. Vocabulary size: {len(vocab)}, Max label length: {max_len}")

# Initialize configs
configs = ModelConfigs()
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
configs.save()

# Prepare DataProvider with the locally saved images
data_provider = DataProvider(
    dataset=dataset_list,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],  # pass CVImage correctly!
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
    ],
)

# Split into train and validation sets
train_data_provider, val_data_provider = data_provider.split(split=0.9)

# Set augmentors for training data
train_data_provider.augmentors = [
    # Recommended augmentations to improve generalization
    # Add more if needed
]

# Build the TensorFlow model using model.py
model = train_model(
    input_dim=(configs.height, configs.width, 3),
    output_dim=len(configs.vocab),
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab),
    ],
    run_eagerly=False
)

model.summary(line_length=110)

# Define callbacks including TensorBoard
early_stopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode="min")
checkpoint = ModelCheckpoint(
    f"{configs.model_path}/model.keras",
    monitor="val_CER",
    verbose=1,
    save_best_only=True,
    mode="min",
)
# train_logger = TrainLogger(configs.model_path)
tensorboard_callback = TensorBoard(log_dir=f"{configs.model_path}/logs", update_freq=1)
reduce_lr = ReduceLROnPlateau(
    monitor="val_CER", factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode="min"
)
# model2onnx = Model2onnx(f"{configs.model_path}/model.keras")

# Train the model with progress and callbacks
model.fit(
    train_data_provider, 
    validation_data=val_data_provider,
    epochs=configs.train_epochs, 
    callbacks=[early_stopper, checkpoint, reduce_lr, tensorboard_callback]
)

# Save train/validation splits as CSV for inferenceModel.py use
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))
