import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import os
import cv2

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
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.face_detector = FaceDetector()
        
        # Load EfficientNet-B0
        try:
            # efficientnet_b0 is available in newer torchvision versions
            # If not available, we might need a fallback, but assuming standard environment
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.model = models.efficientnet_b0(weights=weights)
            
            # Modify classifier for binary classification
            # EfficientNet's classifier is a Sequential block, final layer is [1]
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, 2)
            
            self.model.eval()
            self.model.to(self.device)
            self.model_loaded = True
            print("EfficientNet-B0 loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load EfficientNet model: {e}")
            # Fallback to ResNet18 if EfficientNet fails (e.g. old torchvision)
            try:
                self.model = models.resnet18(pretrained=True)
                self.model.fc = nn.Linear(self.model.fc.in_features, 2)
                self.model.eval()
                self.model.to(self.device)
                self.model_loaded = True
                print("Fallback: ResNet18 loaded.")
            except Exception as e2:
                print(f"Critical: Could not load fallback model: {e2}")
                self.model_loaded = False

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

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
                
                if face_crops:
                    faces_found = True
                    batch_t = []
                    for face in face_crops:
                        batch_t.append(self.transform(face))
                    
                    # Stack into a batch
                    batch_t = torch.stack(batch_t).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(batch_t)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        # Get the max probability of "Fake" (class 1) across all faces
                        # If ANY face is fake, the image is fake
                        fake_probs = probs[:, 1]
                        model_score = torch.max(fake_probs).item()
                else:
                    # No faces found, run on whole image as fallback
                    # (Though EfficientNet trained on faces might not perform well on scenes)
                    img_t = self.transform(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(img_t)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        model_score = probs[0][1].item()
                        
            except Exception as e:
                print(f"Model inference failed: {e}")

        # Combine scores
        # If faces are found, rely heavily on the model (90%)
        # If no faces, rely more on ELA or keep balanced
        if faces_found:
            final_score = (model_score * 0.9) + (ela_score * 0.1)
        else:
            final_score = (model_score * 0.4) + (ela_score * 0.6)
        
        label = "FAKE" if final_score > 0.5 else "REAL"
        
        return {
            "score": float(final_score),
            "label": label,
            "details": {
                "ela_score": float(ela_score),
                "model_confidence": float(model_score),
                "faces_detected": faces_found
            }
        }
