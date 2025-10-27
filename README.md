# ✍️ Handwritten Sentence Recognition using TensorFlow

This repository implements a **Handwritten Sentence Recognition (HSR)** system using **TensorFlow (TF)**.  
The model is trained on the **IAM offline handwriting dataset** and is capable of recognizing **complete handwritten sentences** (multiple words in a line) from image input.

---

## 📘 Overview

The system takes an image containing a **handwritten sentence** and outputs the **recognized text** using a deep learning model based on a **CNN–RNN–CTC** architecture.

- Model is trained on **sentence-level IAM dataset images**.  
- Achieves strong accuracy for real-world handwritten text.  
- Designed for research and experimentation in **OCR**, **document analysis**, and **computer vision**.

---

## 🚀 Features

- 🧠 Recognizes **entire sentences** from handwritten text line images.  
- 🔡 Uses **CTC (Connectionist Temporal Classification)** decoding.  
- ⚙️ Modular and extendable TensorFlow implementation.  
- 🧪 Ready-to-use scripts for **training**, **validation**, and **inference**.  
- 🧩 Compatible with **IAM offline handwriting dataset**.  

---

## 🗂️ Project Structure

Handwritten_text_identification/
│
├── data/ # Sample images and dataset folder
│ ├── sample_sentence.png
│ └── ...
│
├── model/ # Saved model checkpoints
│ └── snapshot-xx
│
├── Handwritten_detection_trained_models/   #No need this becasue i trained my model here from another source so ignore this folder
│
├── src/ # Source code
│ ├── main.py # Main entry for training/inference
│ ├── model.py # Model architecture (CNN + RNN + CTC)
│ ├── data_loader.py # Dataset loading and preprocessing
│ ├── decoder.py # CTC decoding logic
│ ├── config.py # Configurations and hyperparameters
│ └── utils.py # Helper functions
│
├── requirements.txt # Dependencies
├── .gitignore # Ignore files (venv, checkpoints, etc.)
├── LICENSE.md # License (MIT)
└── README.md # Documentation


---

## ⚙️ Installation and Setup

### 1️⃣ Clone the repository
git clone https://github.com/Chaitanya1436/Handwritten_text_identification.git
cd Handwritten_text_identification

### 2️⃣ Create and activate a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate


macOS/Linux:

python3 -m venv venv
source venv/bin/activate

### 3️⃣ Install dependencies
pip install -r requirements.txt

### ▶️ How to Run
Run inference on a sample handwritten sentence image
cd src
python main.py --img_file ../data/sample_sentence.png

### Example Output
Init with stored values from ../model/snapshot-13
Recognized: "this is a handwritten sentence example"
Probability: 0.872

### 🧩 Command-Line Options
Argument	Description	Default
--mode	Select operation mode: train, validate, infer	infer
--decoder	CTC decoder type (bestpath, beamsearch)	bestpath
--batch_size	Training batch size	100
--data_dir	Path to IAM dataset (contains img and gt folders)	../data/iam/
--line_mode	Enables sentence (text-line) mode	True
--img_file	Image path for inference	None
🧠 Training the Model (Optional)

## If you want to train from scratch using the IAM dataset:

### 1️⃣ Prepare dataset

Register and download the IAM dataset (words and ascii annotations).

Organize into the following structure:

data/
  ├── img/
  └── gt/

### 2️⃣ Start training
cd src
python main.py --mode train --data_dir ../data/iam --line_mode --batch_size 250

### 🧰 Example Workflow

# Clone repository
git clone https://github.com/Chaitanya1436/Handwritten_text_identification.git
cd Handwritten_text_identification

# Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run sentence recognition
cd src
python main.py --img_file ../data/sample_sentence.png

# 📝 Notes

Designed only for sentence-level recognition, not isolated words.
Can be extended for custom datasets with similar handwriting formats.
Virtual environments (venv/, .venv/) are excluded via .gitignore.