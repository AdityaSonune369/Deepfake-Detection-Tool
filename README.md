# Deepfake Detection Tool

**Creator:** Aditya

## Overview

The **Deepfake Detection Tool** is an advanced AI-powered web application designed to verify the authenticity of digital media. It utilizes state-of-the-art deep learning models and forensic analysis techniques to detect manipulation in images and videos.

## Features

-   **Hybrid Detection**: Combines Vision Transformer (ViT) neural networks with Error Level Analysis (ELA) for robust results
-   **Smart Face Detection**: Automatically detects and isolates faces using OpenCV for focused analysis
-   **High Accuracy**: 70% threshold calibrated for optimal balance (24% on real images, 86%+ on fakes)
-   **Video Support**: Analyzes video files by processing sampled frames
-   **Modern UI**: Responsive, glassmorphism-inspired web interface with drag-and-drop
-   **Real-time Analysis**: Fast inference with detailed confidence scores

## Technology Stack

### Backend
-   **Framework**: Python 3.11, FastAPI
-   **AI/ML**: PyTorch, Hugging Face Transformers
-   **Computer Vision**: OpenCV, torchvision
-   **Image Processing**: Pillow, NumPy

### Frontend
-   **UI**: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript
-   **Fonts**: Google Fonts (Inter)

### Model
-   **Primary**: `dima806/deepfake_vs_real_image_detection` (Vision Transformer)
-   **Face Detection**: Haar Cascade (OpenCV)

## Installation & Deployment

### Prerequisites

-   Python 3.8 or higher
-   pip (Python package manager)

### 1. Clone/Download the Repository

Ensure you have the project files in a local directory.

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install fastapi uvicorn torch torchvision transformers opencv-python numpy pillow python-multipart huggingface_hub
```

### 3. Run the Application

Start the server:

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload
```

### 4. Access the Tool

Open your web browser and navigate to:

```
http://127.0.0.1:8000
```

## Usage

1.  **Upload Media**: Drag and drop an image or video file, or click "browse files"
2.  **Analyze**: Click the "Analyze Media" button
3.  **View Results**: See the verdict (REAL/FAKE) with confidence score and detailed metrics

## Performance

-   **Real Images**: ~24% fake probability → Correctly labeled as REAL
-   **Deepfakes**: ~86%+ fake probability → Correctly labeled as FAKE
-   **Threshold**: 70% (requires strong evidence before labeling as fake)
-   **Processing Time**: 500-1000ms per image (CPU)

## Technical Details

For comprehensive technical information including:
- Model architectures and training datasets
- Detection algorithms and mathematical formulas
- Complete dependency list
- Performance benchmarks
- System configuration parameters

**See:** [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)

## Project Structure

```
deepfake_detection/
├── main.py                 # FastAPI server
├── detector.py             # Core detection engine
├── video_processor.py      # Video frame processing
├── static/                 # Frontend files
│   ├── index.html
│   ├── script.js
│   └── style.css
├── temp/                   # Temporary file storage
├── README.md               # This file
└── TECHNICAL_DOCUMENTATION.md  # Detailed technical docs
```

## License

This project is created by **Aditya**. All rights reserved.

## Acknowledgments

-   **Vision Transformer Model**: dima806 (Hugging Face)
-   **FastAPI Framework**: Sebastián Ramírez
-   **OpenCV**: OpenCV contributors

