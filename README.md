# Multi-View Crowd Anomaly Detection

A deep learning system for detecting anomalous behavior in crowd scenes using multiple camera views. The system utilizes a Spatio-Temporal Convolutional Network (STCN) to process and analyze video feeds from multiple cameras simultaneously.

## Project Overview

This project implements a multi-view anomaly detection system that can:
- Process video feeds from multiple cameras
- Extract spatio-temporal features from each view
- Combine information from all views
- Detect unusual or anomalous behavior in crowds

### Key Features

- Multi-camera video processing
- Real-time anomaly detection
- Spatio-temporal feature extraction
- Fusion of multiple camera views
- CPU and GPU support

## Architecture

The system consists of three main components:

1. **Feature Extraction (SpatioTemporalBlock)**
   - 3D Convolution layers for spatio-temporal processing
   - Batch Normalization for stable training
   - ReLU activation and MaxPooling for feature selection

2. **Multi-View Fusion**
   - Combines features from multiple camera views
   - Global Average Pooling for feature summarization
   - Dense layers for feature fusion

3. **Anomaly Detection**
   - Final classification layers
   - Outputs anomaly probability score

## Requirements

```python
# requirements.txt
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.3
numpy>=1.19.5
Pillow>=8.3.1
matplotlib>=3.4.3
tqdm>=4.62.2
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crowd-anomaly-detection.git
cd crowd-anomaly-detection
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Link

https://www.kaggle.com/datasets/angelchi56/multiview-highdensity-anomalous-crowd/data 

Thanks to the Dataset Provider

Author Name
Samar Mahmoud

Bio
https://www.gre.ac.uk/people/rep/fach/samar-mahmoud2

