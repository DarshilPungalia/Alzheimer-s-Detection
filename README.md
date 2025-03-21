# Alzheimer's Detection using Deep Learning

## Overview
This project aims to classify MRI brain scans to detect Alzheimer's disease using deep learning. It processes 3D MRI scans, extracts 2D slices, and uses a Convolutional Neural Network (CNN) for classification.

## Features
- **Preprocessing:** Converts 3D MRI scans into 2D slices.
- **CNN Model:** Trained to classify MRI slices into four categories.

## Requirements
Install dependencies using:
```bash
pip install tensorflow numpy nibabel matplotlib
```

## File Structure
```
├── Alzheimer's Detection
│   ├── preprocessing.py   # Data Preprocessing
│   ├── alzeihmers.ipynb   # Model training and saving
│   ├── trained_model/     # Saved trained model
│   ├── README.md          # Project documentation
```

## Usage

### 1. Preprocess MRI Scans
Run the following command to process a 3D MRI scan into 2D slices:
```bash
python preprocessing.py
```
## Dataset
This project assumes a dataset with MRI slices categorized into four labels:
- Mild Dementia
- Moderate Dementia
- Non Demented
- Very Mild Dementia

## Model
The CNN model consists of:
- Convolutional layers with ReLU activation
- Max-pooling layers
- Fully connected layers with softmax activation


