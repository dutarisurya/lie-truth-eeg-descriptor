# Structured Multi-Domain EEG Descriptors with Phase-Based Connectivity for Lie and Truth Detection

This repository provides the implementation scripts for the MethodsX article:

"Structured Multi-Domain EEG Descriptors with Phase-Based Connectivity for Lie and Truth Detection"

## Overview

This code implements a deterministic EEG descriptor-engineering pipeline for binary Lie and Truth classification using the public LieWaves dataset. The pipeline includes:

- 50 Hz notch filtering
- 0.5–45 Hz Butterworth bandpass filtering
- Overlapping sliding-window segmentation
- Temporal, fractal, spectral, and phase-connectivity descriptor extraction
- Meta-correlation descriptor construction
- Fold-wise z-score normalization
- Stratified 5-fold cross-validation
- Random Forest, XGBoost, and SVM classification
- Performance evaluation and computational profiling

## Dataset

The experiments use the public LieWaves dataset:

Aslan, M., Baykara, M., & Alakus, T. B. (2024). LieWaves: Dataset for lie detection based on EEG signals and wavelets. Medical & Biological Engineering & Computing.

Dataset DOI: 10.17632/5gzxb2bzs2.2

The dataset is not redistributed in this repository. Users should download it from the official Mendeley Data repository.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
