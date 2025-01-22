# Swallowing-Detection-with-Machine-Learning

This repository provides a pipeline for detecting swallowing events in audio recordings using machine learning. It includes scripts for preprocessing, feature extraction, model training, and event detection. This tool is designed for researchers and professionals in biosignal processing and dysphagia analysis.


# Features

-   **Audio Preprocessing**:
    -   Bandpass filtering (50–1000 Hz) for noise removal.
    -   Signal normalization and segmentation.
-   **Feature Extraction**:
    -   Time-domain features: Energy, RMS, Zero-Crossing Rate (ZCR).
    -   Frequency-domain features: Spectral Centroid, Dominant Frequency, MFCCs.
    -   Time-frequency features: Wavelet Transform, Spectrogram Analysis.
-   **Machine Learning**:
    -   Models: Logistic Regression, Random Forest, SVM.
    -   Custom training with labeled datasets.
-   **Visualization**:
    -   Spectrograms and detected swallowing events on waveforms.

## Getting started
### Prerequisites

Ensure you have the following installed:

-   Python 3.7 or later
-   Anaconda

## Repository Structure

       swallow-detection/
│
├── data/                # Example audio and labeled datasets
├── preprocess.py        # Audio preprocessing script
├── feature_extraction.py# Feature extraction script
├── train_model.py       # Machine learning model training
├── detect_swallow.py    # Swallow event detection
├── utils/               # Utility functions
├── models/              # Pretrained models
├── README.md            # Project description
└── requirements.txt     # Python dependenciesenter code here

## Workflow
1. Preprocessing
2. Feature extraction
3. Model training
4. Event detection

## Example workflow

1. Record or input raw swallowing audio.
2. Preprocess the audio (e.g., filter out noise).
3. Extract meaningful features (e.g., RMS, MFCCs).
4. Train a model with labeled swallowing and non-swallowing events.
5. Detect swallowing events in new audio recordings using the trained model.


