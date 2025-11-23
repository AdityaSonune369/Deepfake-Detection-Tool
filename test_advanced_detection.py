import cv2
import torch
from detector import DeepfakeDetector
from PIL import Image
import numpy as np

def test_detector():
    print("Initializing DeepfakeDetector...")
    detector = DeepfakeDetector(use_cuda=False)
    
    # Create a dummy image with a face (simulated with a rectangle for Haar Cascade)
    # Haar cascades are sensitive to specific patterns, so a simple rectangle might not work.
    # We will try to load the existing 'test_image.jpg' if it exists, otherwise create a random one.
    
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        print(f"Creating dummy image at {image_path}")
        # Create a blank image
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw a face-like structure (very basic, might not trigger Haar)
        cv2.circle(img, (150, 150), 60, (255, 255, 255), -1)
        cv2.imwrite(image_path, img)
    
    print(f"Testing prediction on {image_path}...")
    try:
        result = detector.predict(image_path)
        print("Prediction Result:")
        print(result)
        
        if result['details']['faces_detected']:
            print("SUCCESS: Faces were detected.")
        else:
            print("NOTE: No faces detected (this might be expected for dummy images).")
            
        print("Test passed!")
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import os
    test_detector()
