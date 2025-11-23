# Deepfake Detection Tool

**Creator:** Aditya

## Overview

The **Deepfake Detection Tool** is an advanced AI-powered application designed to verify the authenticity of digital media. It utilizes state-of-the-art deep learning models and forensic analysis techniques to detect manipulation in images and videos.

## Features

-   **Dual-Layer Detection**: Combines Error Level Analysis (ELA) with a deep learning classifier for robust results.
-   **Advanced AI Model**: Powered by **EfficientNet-B0**, a highly accurate convolutional neural network.
-   **Smart Face Detection**: Automatically detects and isolates faces using **OpenCV** to focus analysis on the most critical regions.
-   **Video Support**: Analyzes video files by processing sampled frames.
-   **Modern UI**: A responsive, glassmorphism-inspired web interface for easy interaction.

## Technology Stack

-   **Backend**: Python, FastAPI
-   **AI/ML**: PyTorch, torchvision, OpenCV, NumPy, Pillow
-   **Frontend**: HTML5, CSS3, Vanilla JavaScript

## Installation & Deployment

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python 3.8 or higher
-   pip (Python package manager)

### 1. Clone/Download the Repository

Ensure you have the project files in a local directory.

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install fastapi uvicorn torch torchvision opencv-python numpy pillow python-multipart
```

### 3. Run the Application

Start the server using the main script:

```bash
python main.py
```

Alternatively, you can run it directly with uvicorn:

```bash
uvicorn main:app --reload
```

### 4. Access the Tool

Open your web browser and navigate to:

```
http://127.0.0.1:8000
```

## Usage

1.  **Upload Media**: Drag and drop an image or video file onto the upload area, or click "browse files".
2.  **Analyze**: Click the "Analyze Media" button.
3.  **View Results**: The tool will display a "REAL" or "FAKE" verdict along with a confidence score and detailed metrics (ELA score, Model confidence).

## License

This project is created by **Aditya**. All rights reserved.
