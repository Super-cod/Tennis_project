from ultralytics import YOLO
import cv2
import numpy as np

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.player_positions = {}
    
    def detect_frames(self, frames):
        """Detect players in all frames"""
        player_detections = []
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        return player_detections

    def detect_frame(self, frame):
        """Detect players in a single frame"""
        results = self.model.track(frame, persist=True)
        
        player_dict = {}
        if results and len(results) > 0:
            result = results[0]
            id_name_dict = result.names
            
            if result.boxes is not None:
                for box in result.boxes:
                    if box.id is not None:  # Check if tracking ID exists
                        track_id = int(box.id.tolist()[0])
                        bbox = box.xyxy.tolist()[0]
                        confidence = float(box.conf.tolist()[0])
                        object_cls_id = int(box.cls.tolist()[0])
                        object_cls_name = id_name_dict[object_cls_id]
                        
                        if object_cls_name == "person" and confidence > 0.5:
                            player_dict[track_id] = {
                                'bbox': bbox,
                                'confidence': confidence,
                                'class': object_cls_name
                            }
                            
                            # Store position history for analysis
                            if track_id not in self.player_positions:
                                self.player_positions[track_id] = []
                            
                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2
                            self.player_positions[track_id].append((center_x, center_y))
                            
                            # Keep only last 50 positions
                            if len(self.player_positions[track_id]) > 50:
                                self.player_positions[track_id].pop(0)

        return player_dict
    
    def classify_players(self, player_detections):
        """Classify players as Player 1 and Player 2 based on court position"""
        if not player_detections:
            return player_detections
        
        # Get all unique track IDs
        all_track_ids = set()
        for detection in player_detections:
            all_track_ids.update(detection.keys())
        
        # If we have exactly 2 players, classify them
        if len(all_track_ids) == 2:
            track_ids = list(all_track_ids)
            
            # Calculate average positions for classification
            avg_positions = {}
            for track_id in track_ids:
                positions = []
                for detection in player_detections:
                    if track_id in detection:
                        bbox = detection[track_id]['bbox']
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        positions.append((center_x, center_y))
                
                if positions:
                    avg_x = sum(pos[0] for pos in positions) / len(positions)
                    avg_y = sum(pos[1] for pos in positions) / len(positions)
                    avg_positions[track_id] = (avg_x, avg_y)
            
            # Classify based on court position (top/bottom or left/right)
            if len(avg_positions) == 2:
                sorted_by_y = sorted(avg_positions.items(), key=lambda x: x[1][1])
                player1_id = sorted_by_y[0][0]  # Top player
                player2_id = sorted_by_y[1][0]  # Bottom player
                
                # Update detections with player labels
                for detection in player_detections:
                    for track_id in list(detection.keys()):
                        if track_id == player1_id:
                            detection[track_id]['player_label'] = 'Player 1'
                        elif track_id == player2_id:
                            detection[track_id]['player_label'] = 'Player 2'
        
        return player_detections

    def draw_player_tracking(self, video_frames, player_detections):
        """Draw player tracking with enhanced visualizations"""
        output_video_frames = []
        
        for frame, detection in zip(video_frames, player_detections):
            frame_copy = frame.copy()
            
            for track_id, player_data in detection.items():
                bbox = player_data['bbox']
                confidence = player_data['confidence']
                player_label = player_data.get('player_label', f'Person {track_id}')
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Different colors for different players
                if 'Player 1' in player_label:
                    color = (255, 0, 0)  # Blue for Player 1
                elif 'Player 2' in player_label:
                    color = (0, 255, 0)  # Green for Player 2
                else:
                    color = (0, 255, 255)  # Yellow for other persons
                
                # Draw bounding box
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f'{player_label}: {confidence:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame_copy, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame_copy, (center_x, center_y), 5, color, -1)
                
                # Draw movement trail if available
                if track_id in self.player_positions and len(self.player_positions[track_id]) > 1:
                    positions = self.player_positions[track_id][-10:]  # Last 10 positions
                    for i in range(1, len(positions)):
                        pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                        pt2 = (int(positions[i][0]), int(positions[i][1]))
                        alpha = i / len(positions)
                        thickness = max(1, int(3 * alpha))
                        cv2.line(frame_copy, pt1, pt2, color, thickness)
            
            output_video_frames.append(frame_copy)
        
        return output_video_frames