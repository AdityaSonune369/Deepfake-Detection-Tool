import cv2
import os
import numpy as np
from detector import DeepfakeDetector

class VideoProcessor:
    def __init__(self, detector: DeepfakeDetector):
        self.detector = detector

    def process_video(self, video_path, frame_interval=30):
        """
        Process a video file, extracting frames and running detection.
        frame_interval: Process every Nth frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file"}

        frames_processed = 0
        total_score = 0
        frame_scores = []
        
        frame_count = 0
        
        temp_frame_path = "temp_frame.jpg"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Save frame to disk temporarily to use the existing detector logic
                # (Optimization: Refactor detector to accept in-memory images)
                cv2.imwrite(temp_frame_path, frame)
                
                result = self.detector.predict(temp_frame_path)
                score = result['score']
                frame_scores.append(score)
                total_score += score
                frames_processed += 1

            frame_count += 1

        cap.release()
        try:
            os.remove(temp_frame_path)
        except:
            pass

        if frames_processed == 0:
            return {"score": 0, "label": "UNKNOWN", "frames_analyzed": 0}

        avg_score = total_score / frames_processed
        label = "FAKE" if avg_score > 0.5 else "REAL"

        return {
            "score": avg_score,
            "label": label,
            "frames_analyzed": frames_processed,
            "frame_scores": frame_scores # Can be used for a timeline graph
        }
