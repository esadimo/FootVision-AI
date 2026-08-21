"""
src.detection.ball_detector — Specialized ball detection heuristics.
"""

from typing import Tuple, List, Optional
import numpy as np

class BallDetector:
    def __init__(self, conf_threshold: float = 0.05, max_radius: int = 35):
        """
        Specialized ball detector filter for YOLO results.
        
        Parameters
        ----------
        conf_threshold : float
            Lower confidence threshold to maximize recall for the small/blurred ball.
        max_radius : int
            Maximum acceptable pixel radius for a ball (rejects large false positives).
        """
        self.conf_threshold = conf_threshold
        self.max_radius = max_radius

    def extract_best_ball(self, yolo_boxes) -> Optional[Tuple[float, Tuple[float, float, float, float]]]:
        """
        Extracts the most likely ball detection from YOLO results.
        
        Parameters
        ----------
        yolo_boxes : ultralytics.engine.results.Boxes
            The boxes attribute from a YOLO result object.
            
        Returns
        -------
        Tuple[float, Tuple[float, float, float, float]] or None
            (confidence, (x1, y1, x2, y2)) of the best candidate, or None.
        """
        if yolo_boxes is None or len(yolo_boxes) == 0:
            return None

        ball_cands = []
        for box in yolo_boxes:
            cls_id = int(box.cls[0])
            if cls_id != 32:  # 32 is 'sports ball' in COCO
                continue
                
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            
            # Filter 1: Size constraint
            if w > self.max_radius * 2 or h > self.max_radius * 2:
                continue
                
            # Filter 2: Aspect ratio constraint
            # A moving ball can be elongated by motion blur, but shouldn't be extreme
            aspect_ratio = max(w, h) / max(min(w, h), 1.0)
            if aspect_ratio > 3.0:
                continue
                
            ball_cands.append((conf, (x1, y1, x2, y2)))

        if not ball_cands:
            return None

        # Sort by confidence descending and return the best one
        ball_cands.sort(key=lambda x: x[0], reverse=True)
        return ball_cands[0]
