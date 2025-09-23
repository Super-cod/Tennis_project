from ultralytics import YOLO
import cv2
import numpy as np

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.ball_positions = []
    
    def detect_frames(self, frames):
        """Detect tennis ball in all frames"""
        ball_detections = []
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        return ball_detections
    
    def detect_frame(self, frame):
        """Detect tennis ball in a single frame"""
        results = self.model.track(frame, persist=True)
        
        ball_dict = {}
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    bbox = box.xyxy.tolist()[0]
                    confidence = float(box.conf.tolist()[0])
                    
                    # Only consider high confidence detections
                    if confidence > 0.5:
                        if box.id is not None:
                            track_id = int(box.id.tolist()[0])
                            ball_dict[track_id] = {
                                'bbox': bbox,
                                'confidence': confidence
                            }
                        else:
                            
                            ball_dict[0] = {
                                'bbox': bbox,
                                'confidence': confidence
                            }
        
        return ball_dict
    
    def interpolate_ball_positions(self, ball_detections):
        """Interpolate missing ball positions for smoother tracking"""
        interpolated_detections = ball_detections.copy()
        
        
        detected_positions = []
        for i, detection in enumerate(ball_detections):
            if detection:
                for track_id, ball_data in detection.items():
                    bbox = ball_data['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    detected_positions.append((i, center_x, center_y, bbox))
        
        if len(detected_positions) >= 2:
            for i in range(len(ball_detections)):
                if not ball_detections[i]:  # Missing detection
                    # Find nearest detected positions
                    prev_pos = None
                    next_pos = None
                    
                    for pos in detected_positions:
                        if pos[0] < i:
                            prev_pos = pos
                        elif pos[0] > i and next_pos is None:
                            next_pos = pos
                            break
                    
                    # Interpolate if we have both previous and next positions
                    if prev_pos and next_pos:
                        frame_diff = next_pos[0] - prev_pos[0]
                        current_diff = i - prev_pos[0]
                        ratio = current_diff / frame_diff
                        
                        # Interpolate position
                        interp_x = prev_pos[1] + (next_pos[1] - prev_pos[1]) * ratio
                        interp_y = prev_pos[2] + (next_pos[2] - prev_pos[2]) * ratio
                        
                        # Create interpolated bbox
                        bbox_width = prev_pos[3][2] - prev_pos[3][0]
                        bbox_height = prev_pos[3][3] - prev_pos[3][1]
                        
                        interp_bbox = [
                            interp_x - bbox_width/2,
                            interp_y - bbox_height/2,
                            interp_x + bbox_width/2,
                            interp_y + bbox_height/2
                        ]
                        
                        interpolated_detections[i] = {
                            0: {
                                'bbox': interp_bbox,
                                'confidence': 0.3,  # Lower confidence for interpolated
                                'interpolated': True
                            }
                        }
        
        return interpolated_detections
    
    def draw_ball_tracking(self, video_frames, ball_detections):
        """Draw ball tracking on video frames"""
        output_frames = []
        ball_trail = []  # Store ball positions for trail effect
        
        for frame, detection in zip(video_frames, ball_detections):
            frame_copy = frame.copy()
            
            if detection:
                for track_id, ball_data in detection.items():
                    bbox = ball_data['bbox']
                    confidence = ball_data['confidence']
                    is_interpolated = ball_data.get('interpolated', False)
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Add to trail
                    ball_trail.append((center_x, center_y))
                    if len(ball_trail) > 30:  # Keep last 30 positions
                        ball_trail.pop(0)
                    
                    # Draw ball detection
                    if is_interpolated:
                        color = (0, 255, 255)  # Yellow for interpolated
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(frame_copy, f'Ball (Interp): {confidence:.2f}', 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        color = (0, 0, 255)  # Red for detected ball
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_copy, f'Ball: {confidence:.2f}', 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw center point
                    cv2.circle(frame_copy, (center_x, center_y), 3, color, -1)
            
            # Draw ball trail
            if len(ball_trail) > 1:
                for i in range(1, len(ball_trail)):
                    alpha = i / len(ball_trail)  # Fade effect
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame_copy, ball_trail[i-1], ball_trail[i], 
                           (0, 150, 255), thickness)
            
            output_frames.append(frame_copy)
        
        return output_frames