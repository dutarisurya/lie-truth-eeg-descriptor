# Structured Multi-Domain EEG Descriptor Pipeline 

This repository provides the implementation code for the MethodsX manuscript:

**Structured Multi-Domain EEG Descriptors with Phase-Based Connectivity for Lie and Truth Detection**

## Overview

This code implements a deterministic EEG descriptor-engineering pipeline for binary Lie and Truth classification using the public LieWaves dataset.

The pipeline includes:

- 50 Hz notch filtering
- 0.5–45 Hz Butterworth bandpass filtering
- overlapping sliding-window segmentation
- temporal, fractal, spectral, phase-connectivity, and meta-correlation descriptors
- fold-wise z-score normalization
- stratified 5-fold cross-validation
- optional subject-wise grouped validation
- optional leave-one-subject-out validation
- meta-correlation ablation analysis
- performance metrics and confusion matrix generation

## Dataset

The LieWaves dataset is not redistributed in this repository. Users should download it from the official Mendeley Data repository.

Dataset DOI: 10.17632/5gzxb2bzs2.2

## Expected Dataset Structure

After downloading the LieWaves dataset, arrange the files as follows:

```text
data/LieWaves/
├── Raw/
│   ├── S1S1.csv
│   ├── S1S2.csv
│   └── ...
└── Subject_Stimuli.xlsx
```

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline using:

```bash
python run_liewaves_descriptor_pipeline.py
```

## Outputs

The results will be saved in:

```text
results/
├── figures/
└── tables/
```

## Validation Note

The main evaluation protocol uses window-level stratified 5-fold cross-validation with fold-wise z-score normalization. Normalization parameters are estimated only from the training fold and applied to the corresponding test fold.

Because highly overlapping windows may introduce temporal dependency between adjacent EEG segments, the window-level results should be interpreted as controlled descriptor separability under a subject-dependent setting, not as strict subject-independent generalization.

The script also includes optional subject-wise GroupKFold and Leave-One-Subject-Out settings for stricter generalization analysis.