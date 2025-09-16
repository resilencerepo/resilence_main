# detector.py
import os
import cv2
import random
import torch
from collections import deque
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PersonDetector using device: {self.device}")
        self.model = YOLO(model_name).to(self.device)  # Ensure model is on GPU
        if self.device.type == 'cuda':
            self.model.model.half()  # Use FP16 for speed
        
        # Cache FRAME_SIZE as int, provide default if not set
        frame_size_env = os.getenv('FRAME_SIZE', '640')
        self.target_width = int(frame_size_env)
        
        # Use deque for track history with maxlen to limit memory
        self.track_history = {}
        self.max_history_len = 10  # example limit
        
    def detect_and_track(self, frame):
        # Resize frame once, convert to RGB and to tensor on device
        resized_frame = cv2.resize(frame, (self.target_width, self.target_width))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(rgb_frame).to(self.device)
        img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0  # C,H,W and normalize
        
        if self.device.type == 'cuda':
            img_tensor = img_tensor.half()  # FP16
        
        img_tensor = img_tensor.unsqueeze(0)  # batch dimension
        
        results = self.model.track(img_tensor, persist=True, classes=[0])
        
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        
        for box, track_id in zip(boxes, track_ids):
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.max_history_len)
            self.track_history[track_id].append(box)
        
        return results[0].plot(), track_ids
    
    def get_random_person(self, frame, track_ids):
        if not track_ids:
            return None, None
        
        selected_id = random.choice(track_ids)
        person_box = self.track_history[selected_id][-1]
        
        x, y, w, h = person_box
        # Expand bounding box to include full person or upper body, prioritizing head
        expansion_factor_y = 1.5  # Expand height by 1.5x
        expansion_factor_x = 1.2  # Expand width slightly
        new_h = h * expansion_factor_y
        new_w = w * expansion_factor_x
        # Shift y upward to prioritize head and upper body
        new_y = y - (new_h - h) * 0.7  # Move up to focus on head
        x1 = int(x - new_w / 2)
        y1 = int(new_y - new_h / 2)
        x2 = int(x + new_w / 2)
        y2 = int(new_y + new_h / 2)
        
        # Ensure coordinates are within frame boundaries
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)
        
        # Check if the cropped area is valid
        if x2 <= x1 or y2 <= y1:
            print("Invalid bounding box after expansion")
            return None, None
        
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


