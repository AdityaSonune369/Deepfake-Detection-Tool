# Technical Documentation - Deepfake Detection Tool

**Creator:** Aditya Sonune 
**Last Updated:** November 23, 2025

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Models & Datasets](#models--datasets)
4. [Detection Algorithms](#detection-algorithms)
5. [Technology Stack](#technology-stack)
6. [Performance Metrics](#performance-metrics)
7. [System Configuration](#system-configuration)

---

## Overview

This deepfake detection tool is a web-based application that leverages state-of-the-art deep learning models and forensic analysis techniques to identify manipulated images and videos. The system uses a hybrid approach combining neural network-based classification with heuristic analysis for robust detection.

---

## Architecture

### System Design

```
┌─────────────────┐
│   Web Browser   │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   FastAPI       │
│   Server        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      DeepfakeDetector               │
│  ┌──────────────┬────────────────┐  │
│  │ Face         │ Vision         │  │
│  │ Detector     │ Transformer    │  │
│  │ (OpenCV)     │ (Hugging Face) │  │
│  └──────────────┴────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ Error Level Analysis (ELA)   │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Component Breakdown

1. **Frontend**: Single Page Application (SPA) with modern UI
2. **Backend**: FastAPI REST API
3. **Detection Engine**: Multi-modal deepfake detector
4. **Video Processor**: Frame-by-frame analysis system

---

## Models & Datasets

### Primary Detection Model

**Model Name:** `dima806/deepfake_vs_real_image_detection`

**Architecture:** Vision Transformer (ViT)
- **Base Model**: Google's Vision Transformer
- **Fine-tuning**: Specialized for deepfake detection
- **Source**: Hugging Face Model Hub
- **Repository**: https://huggingface.co/dima806/deepfake_vs_real_image_detection

**Training Dataset:**
- Trained on curated datasets of real and manipulated images
- Includes various deepfake generation techniques (face swap, face reenactment, etc.)

**Model Characteristics:**
- Input Size: 224x224 pixels
- Output: Binary classification (Real/Fake)
- Parameters: ~86M (ViT-Base)
- Confidence Scores: Provides probability distributions

### Face Detection Model

**Model Name:** Haar Cascade Frontal Face Detector

**Architecture:** Viola-Jones Algorithm
- **Source**: OpenCV pre-trained cascade
- **File**: `haarcascade_frontalface_default.xml`
- **Repository**: OpenCV library

**Training Dataset:**
- Trained on thousands of positive and negative face samples
- Original paper: Viola-Jones (2001)

**Parameters:**
- Scale Factor: 1.1
- Min Neighbors: 5
- Min Size: 30x30 pixels
- Margin: 20% padding around detected faces

### Alternative Models Evaluated

**Model:** `prithivMLmods/Deep-Fake-Detector-v2-Model`
- **Accuracy:** 92.12% on validation set
- **Result:** Rejected due to high false positive rate (68% on real images)
- **Architecture:** Vision Transformer (ViT)

---

## Detection Algorithms

### 1. Face Detection Workflow

```python
Input Image
    ↓
Convert to Grayscale
    ↓
Apply Haar Cascade Detection
    ↓
Extract Face Regions (with 20% margin)
    ↓
Return Face Crops
```

### 2. Error Level Analysis (ELA)

**Algorithm:**
1. Save original image as JPEG with quality=90
2. Reload compressed image
3. Calculate pixel-wise difference
4. Normalize differences to 0-255 range
5. Compute average brightness as manipulation score

**Mathematical Formula:**
```
ELA_score = min(mean(|original - compressed|) / 50.0, 1.0)
```

**Principle:**
- Manipulated regions show different compression artifacts
- Inconsistent error levels indicate tampering

### 3. Neural Network Classification

**Process:**
1. **Preprocessing:**
   - Resize to 224x224
   - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
2. **Inference:**
   - Forward pass through Vision Transformer
   - Softmax activation for probability distribution
3. **Post-processing:**
   - Extract "Fake" class probability
   - If multiple faces: take maximum probability

### 4. Score Fusion

**Weighted Combination:**

When faces detected:
```
final_score = (model_score × 0.85) + (ELA_score × 0.15)
```

When no faces:
```
final_score = (model_score × 0.70) + (ELA_score × 0.30)
```

**Decision Threshold:** 70%
- Score > 0.70 → FAKE
- Score ≤ 0.70 → REAL

---

## Technology Stack

### Backend
- **Framework**: FastAPI 0.104.0+
- **Server**: Uvicorn (ASGI)
- **Python Version**: 3.8+

### Machine Learning
- **Deep Learning**: PyTorch 2.9.1
- **Computer Vision**: torchvision 0.24.1
- **Transformers**: Hugging Face Transformers 4.57.1+
- **Image Processing**: Pillow (PIL)
- **Face Detection**: OpenCV 4.12.0

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with glassmorphism effects
- **JavaScript**: ES6+ features
- **Fonts**: Google Fonts (Inter)

### Development Tools
- **Version Control**: Git
- **Package Manager**: pip
- **Environment**: Python virtual environment

---

## Performance Metrics

### Model Performance

**Observed Accuracy (Test Set):**

| Test Case | Score | Label | Result |
|-----------|-------|-------|--------|
| Real Image (Portrait) | 24% | REAL | ✓ Correct |
| Face-Swap Deepfake | 86% | FAKE | ✓ Correct |
| Real Image (Outdoor) | ~27% | REAL | ✓ Correct |

**Performance Characteristics:**
- **True Positive Rate**: High (86%+ confidence on obvious deepfakes)
- **False Positive Rate**: Low (24-27% on real images, well below 70% threshold)
- **Threshold Calibration**: 70% provides optimal balance

### Inference Speed

- **Face Detection**: ~50-100ms per image
- **Model Inference**: ~200-500ms per face (CPU)
- **ELA Analysis**: ~100-200ms per image
- **Total Processing**: ~500-1000ms per image (single face)

### Resource Usage

- **Memory**: ~500MB (model loaded)
- **CPU**: Single-threaded inference
- **GPU**: Optional CUDA support (not tested)

---

## System Configuration

### Detection Parameters

```python
# Face Detection
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (30, 30)
FACE_MARGIN = 0.2  # 20% padding

# Model Fusion Weights
FACE_DETECTED_MODEL_WEIGHT = 0.85
FACE_DETECTED_ELA_WEIGHT = 0.15
NO_FACE_MODEL_WEIGHT = 0.70
NO_FACE_ELA_WEIGHT = 0.30

# Decision Threshold
FAKE_THRESHOLD = 0.70

# ELA Parameters
ELA_QUALITY = 90  # JPEG compression quality
ELA_NORMALIZATION = 50.0
```

### Video Processing

```python
FRAME_INTERVAL = 30  # Process every 30th frame (~1 second at 30fps)
```

---

## Dependencies

### Python Packages

```
fastapi>=0.104.0
uvicorn>=0.38.0
torch>=2.9.1
torchvision>=0.24.1
transformers>=4.57.1
opencv-python>=4.12.0
pillow>=10.0.0
numpy>=1.26.0
python-multipart>=0.0.6
huggingface_hub>=0.36.0
```

---

## Research & References

### Key Papers
1. **Vision Transformer (ViT)**: "An Image is Worth 16x16 Words" - Dosovitskiy et al., 2020
2. **Face Detection**: "Rapid Object Detection using Boosted Cascade" - Viola & Jones, 2001
3. **Error Level Analysis**: Krawetz, N. (2007) "A Picture's Worth"

### Model Sources
- Hugging Face Model Hub: https://huggingface.co/models
- OpenCV Pre-trained Models: https://github.com/opencv/opencv

---

## Limitations & Future Work

### Current Limitations
1. CPU-only inference (slower than GPU)
2. No fine-tuning on specific deepfake datasets
3. Haar Cascade has limited face detection accuracy
4. Single-model approach (no ensemble)

### Potential Improvements
1. **GPU Acceleration**: CUDA support for faster inference
2. **Better Face Detection**: MTCNN or RetinaFace
3. **Model Fine-tuning**: Train on FaceForensics++ dataset
4. **Ensemble Methods**: Combine multiple detection models
5. **XAI**: Explainability features (Grad-CAM visualization)
6. **Real-time Processing**: Optimize for video streaming

---

## License & Attribution

**Creator:** Aditya Sonune  
**Model Credits:**
- Vision Transformer: dima806 (Hugging Face)
- Haar Cascade: OpenCV contributors
- FastAPI: Sebastián Ramírez

**License:** Educational/Personal Use

---

*For technical support or contributions, refer to the main README.md*
