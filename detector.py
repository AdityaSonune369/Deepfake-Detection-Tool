import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import os
import cv2
from transformers import pipeline

class FaceDetector:
    def __init__(self):
        # Load the pre-trained Haar Cascade classifier for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("Warning: Could not load Haar Cascade classifier.")

    def detect_faces(self, image):
        """
        Detects faces in a PIL Image.
        Returns a list of PIL Image crops of the faces.
        """
        # Convert PIL image to OpenCV format (BGR)
        img_np = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        face_crops = []
        for (x, y, w, h) in faces:
            # Add a small margin
            margin = int(w * 0.2)
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(img_cv.shape[1], x + w + margin)
            y_end = min(img_cv.shape[0], y + h + margin)
            
            face_crop = img_np[y_start:y_end, x_start:x_end]
            face_crops.append(Image.fromarray(face_crop))
            
        return face_crops

class DeepfakeDetector:
    def __init__(self, use_cuda=False):
        self.device = 0 if use_cuda and torch.cuda.is_available() else -1
        print(f"Using device: {'cuda' if self.device == 0 else 'cpu'}")
        
        self.face_detector = FaceDetector()
        
        # Load Hugging Face Pipeline
        # Using dima806/deepfake_vs_real_image_detection (better calibration)
        # This model has lower false positive rate
        try:
            print("Loading Hugging Face model...")
            self.pipe = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection", device=self.device)
            self.model_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False

    def perform_ela(self, image_path, quality=90):
        """
        Performs Error Level Analysis (ELA) to detect manipulation artifacts.
        Returns a score based on the amount of high-frequency noise difference.
        """
        try:
            original = Image.open(image_path).convert('RGB')
            
            # Save compressed version to a temporary buffer/file
            temp_filename = "temp_ela.jpg"
            original.save(temp_filename, 'JPEG', quality=quality)
            compressed = Image.open(temp_filename)
            
            # Calculate difference
            diff = ImageChops.difference(original, compressed)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            
            if max_diff == 0:
                max_diff = 1
                
            scale = 255.0 / max_diff
            diff = ImageEnhance.Brightness(diff).enhance(scale)
            
            # Calculate a simple "fake" score based on the average brightness of the ELA
            np_diff = np.array(diff)
            avg_diff = np.mean(np_diff)
            
            # Clean up
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except:
                pass
                
            # Normalize score roughly to 0-1 (heuristic)
            score = min(avg_diff / 50.0, 1.0) 
            return score
        except Exception as e:
            print(f"ELA failed: {e}")
            return 0.0

    def predict(self, image_path):
        """
        Predicts if an image is a deepfake.
        Returns a dictionary with 'score' (0-1, where 1 is fake) and 'label'.
        """
        # 1. Heuristic Score (ELA)
        ela_score = self.perform_ela(image_path)
        
        # 2. Model Score
        model_score = 0.5 # Default uncertainty
        faces_found = False
        
        if self.model_loaded:
            try:
                img = Image.open(image_path).convert('RGB')
                
                # Detect faces
                face_crops = self.face_detector.detect_faces(img)
                
                images_to_process = []
                if face_crops:
                    faces_found = True
                    images_to_process = face_crops
                else:
                    images_to_process = [img]
                
                # Run pipeline
                # Pipeline returns a list of dicts: [{'label': 'REAL', 'score': 0.9}, {'label': 'FAKE', 'score': 0.1}]
                # We need to find the score for 'FAKE' (or '1')
                
                max_fake_score = 0.0
                
                for inp_img in images_to_process:
                    results = self.pipe(inp_img)
                    # The model returns labels like "Deepfake" or "Realism" (or "FAKE"/"REAL")
                    # We need to find the score for the "fake" class
                    
                    fake_prob = 0.0
                    for res in results:
                        label = res['label'].upper()
                        # Check for various fake labels: FAKE, DEEPFAKE, 1
                        if 'FAKE' in label or 'DEEPFAKE' in label or label == '1':
                            fake_prob = res['score']
                        # Check for real labels: REAL, REALISM, 0
                        elif 'REAL' in label or label == '0':
                            # If it's real, the fake prob is 1 - real_prob
                            pass
                    
                    # If we didn't find an explicit FAKE label, maybe it only returned the top class
                    # If top class is REAL, fake_prob is low. If top class is FAKE, fake_prob is high.
                    # Let's handle the case where we only get one label
                    if len(results) == 1:
                         label = results[0]['label'].upper()
                         score = results[0]['score']
                         if 'FAKE' in label or 'DEEPFAKE' in label or label == '1':
                             fake_prob = score
                         else:
                             fake_prob = 1.0 - score
                    
                    # Update max fake score found across all faces
                    if fake_prob > max_fake_score:
                        max_fake_score = fake_prob
                        
                model_score = max_fake_score
                        
            except Exception as e:
                print(f"Model inference failed: {e}")
                import traceback
                traceback.print_exc()

        # Combine scores
        # Balance between the model and ELA to reduce both false positives and false negatives
        # The model is powerful but can have biases, ELA provides a sanity check
        
        if faces_found:
            # If faces are found, trust the model heavily but use ELA as a sanity check
            final_score = (model_score * 0.85) + (ela_score * 0.15)
        else:
            # If no faces found, be more conservative and balance the scores
            final_score = (model_score * 0.70) + (ela_score * 0.30)
        
        # Use a higher threshold (70%) to reduce false positives on real images
        # This requires even stronger evidence before labeling something as fake
        label = "FAKE" if final_score > 0.70 else "REAL"
        
        return {
            "score": float(final_score),
            "label": label,
            "details": {
                "ela_score": float(ela_score),
                "model_confidence": float(model_score),
                "faces_detected": faces_found
            }
        }
