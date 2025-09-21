from ultralytics import YOLO
import cv2
import numpy as np
from .player_tracker import PlayerTracker
from .ball_tracker import BallTracker

class TennisTracker:
    def __init__(self, player_model_path="yolov8n.pt", ball_model_path="models/best.pt"):
        self.player_tracker = PlayerTracker(player_model_path)
        self.ball_tracker = BallTracker(ball_model_path)
        self.match_stats = {
            'ball_hits': 0,
            'rally_length': 0,
            'player_distances': {'Player 1': 0, 'Player 2': 0}
        }
    
    def track_tennis_match(self, video_frames):
        """Complete tennis match tracking with players and ball"""
        print("Tracking players...")
        player_detections = self.player_tracker.detect_frames(video_frames)
        player_detections = self.player_tracker.classify_players(player_detections)
        
        print("Tracking tennis ball...")
        ball_detections = self.ball_tracker.detect_frames(video_frames)
        ball_detections = self.ball_tracker.interpolate_ball_positions(ball_detections)
        
        print("Analyzing match...")
        self.analyze_match(player_detections, ball_detections)
        
        return player_detections, ball_detections
    
    def analyze_match(self, player_detections, ball_detections):
        """Analyze tennis match for statistics"""
        previous_ball_pos = None
        previous_player_positions = {}
        
        for frame_idx, (players, ball) in enumerate(zip(player_detections, ball_detections)):
            # Analyze ball movement for hit detection
            if ball:
                for ball_id, ball_data in ball.items():
                    bbox = ball_data['bbox']
                    current_ball_pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    
                    if previous_ball_pos:
                        # Detect sudden direction changes (potential hits)
                        velocity = (current_ball_pos[0] - previous_ball_pos[0], 
                                  current_ball_pos[1] - previous_ball_pos[1])
                        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                        
                        if speed > 50:  # Threshold for ball hit detection
                            self.match_stats['ball_hits'] += 1
                    
                    previous_ball_pos = current_ball_pos
            
            # Track player movement distances
            for track_id, player_data in players.items():
                player_label = player_data.get('player_label', f'Person {track_id}')
                bbox = player_data['bbox']
                current_pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                
                if track_id in previous_player_positions:
                    prev_pos = previous_player_positions[track_id]
                    distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + 
                                     (current_pos[1] - prev_pos[1])**2)
                    
                    if player_label in self.match_stats['player_distances']:
                        self.match_stats['player_distances'][player_label] += distance
                
                previous_player_positions[track_id] = current_pos
        
        # Calculate rally length (frames with ball visible)
        self.match_stats['rally_length'] = sum(1 for detection in ball_detections if detection)
    
    def draw_complete_analysis(self, video_frames, player_detections, ball_detections):
        """Draw complete tennis analysis with players and ball tracking in one video"""
        output_frames = []
        ball_trail = []  # Store ball positions for trail effect
        player_trails = {}  # Store player positions for trails
        
        for frame_idx, (frame, players, ball) in enumerate(zip(video_frames, player_detections, ball_detections)):
            frame_copy = frame.copy()
            
            # Draw player tracking with enhanced visuals
            for track_id, player_data in players.items():
                bbox = player_data['bbox']
                confidence = player_data['confidence']
                player_label = player_data.get('player_label', f'Person {track_id}')
                
                x1, y1, x2, y2 = map(int, bbox)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Player colors - more vibrant
                if 'Player 1' in player_label:
                    color = (255, 100, 0)  # Bright Blue
                elif 'Player 2' in player_label:
                    color = (0, 255, 100)  # Bright Green
                else:
                    color = (0, 255, 255)  # Yellow
                
                # Store player trail
                if track_id not in player_trails:
                    player_trails[track_id] = []
                player_trails[track_id].append((center_x, center_y))
                if len(player_trails[track_id]) > 20:  # Keep last 20 positions
                    player_trails[track_id].pop(0)
                
                # Draw player trail
                if len(player_trails[track_id]) > 1:
                    for i in range(1, len(player_trails[track_id])):
                        alpha = i / len(player_trails[track_id])
                        thickness = max(1, int(4 * alpha))
                        pt1 = player_trails[track_id][i-1]
                        pt2 = player_trails[track_id][i]
                        cv2.line(frame_copy, pt1, pt2, color, thickness)
                
                # Draw player bounding box with rounded corners effect
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
                
                # Draw player center point
                cv2.circle(frame_copy, (center_x, center_y), 6, color, -1)
                cv2.circle(frame_copy, (center_x, center_y), 8, (255, 255, 255), 2)
                
                # Enhanced player label with background
                label = f'{player_label}: {confidence:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                # Background rectangle
                cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 15), 
                             (x1 + label_size[0] + 10, y1), color, -1)
                # Text
                cv2.putText(frame_copy, label, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw ball tracking with enhanced trail
            if ball:
                for ball_id, ball_data in ball.items():
                    bbox = ball_data['bbox']
                    confidence = ball_data['confidence']
                    is_interpolated = ball_data.get('interpolated', False)
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Add to ball trail
                    ball_trail.append((center_x, center_y))
                    if len(ball_trail) > 40:  # Keep last 40 positions for longer trail
                        ball_trail.pop(0)
                    
                    # Ball colors and styles
                    if is_interpolated:
                        ball_color = (0, 255, 255)  # Yellow for interpolated
                        cv2.circle(frame_copy, (center_x, center_y), 12, ball_color, 2)
                        cv2.circle(frame_copy, (center_x, center_y), 6, ball_color, -1)
                        label_text = f'Ball (Est): {confidence:.2f}'
                    else:
                        ball_color = (0, 0, 255)  # Red for detected ball
                        cv2.circle(frame_copy, (center_x, center_y), 15, ball_color, -1)
                        cv2.circle(frame_copy, (center_x, center_y), 18, (255, 255, 255), 2)
                        label_text = f'Ball: {confidence:.2f}'
                    
                    # Ball label
                    cv2.putText(frame_copy, label_text, (x1, y1 - 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 2)
            
            # Draw enhanced ball trail
            if len(ball_trail) > 1:
                for i in range(1, len(ball_trail)):
                    alpha = i / len(ball_trail)
                    thickness = max(1, int(5 * alpha))
                    # Gradient color effect
                    trail_color = (int(255 * alpha), int(150 * alpha), 255)
                    cv2.line(frame_copy, ball_trail[i-1], ball_trail[i], trail_color, thickness)
            
            # Draw enhanced match statistics overlay
            self.draw_enhanced_stats_overlay(frame_copy, frame_idx, len(players), bool(ball))
            
            output_frames.append(frame_copy)
        
        return output_frames
    
    def draw_enhanced_stats_overlay(self, frame, frame_idx, player_count, ball_detected):
        """Draw enhanced match statistics overlay on frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Main stats panel background - larger and more prominent
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title with decorative border
        cv2.rectangle(frame, (10, 10), (500, 45), (0, 255, 255), 2)
        cv2.putText(frame, "🎾 TENNIS MATCH ANALYSIS", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Current frame info
        frame_time = frame_idx / 30.0  # Assuming 30fps
        minutes = int(frame_time // 60)
        seconds = int(frame_time % 60)
        
        # Draw enhanced statistics
        stats_text = [
            f"Time: {minutes:02d}:{seconds:02d} (Frame {frame_idx + 1})",
            f"Players Detected: {player_count}/2",
            f"Ball Detected: {'✅ YES' if ball_detected else '❌ NO'}",
            f"Ball Hits Detected: {self.match_stats['ball_hits']}",
            f"Rally Duration: {self.match_stats['rally_length']} frames",
            f"Player 1 Movement: {self.match_stats['player_distances'].get('Player 1', 0):.0f}px",
            f"Player 2 Movement: {self.match_stats['player_distances'].get('Player 2', 0):.0f}px"
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = 65 + i * 20
            # Add icons and colors
            if i == 0:  # Time
                cv2.putText(frame, "⏱️", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, text, (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            elif i == 1:  # Players
                color = (0, 255, 0) if player_count == 2 else (0, 255, 255)
                cv2.putText(frame, "👥", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, text, (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            elif i == 2:  # Ball
                color = (0, 255, 0) if ball_detected else (0, 0, 255)
                cv2.putText(frame, "🎾", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, text, (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            else:
                cv2.putText(frame, "📊", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, text, (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add legend for colors
        legend_y = frame.shape[0] - 80
        cv2.rectangle(frame, (10, legend_y - 10), (300, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, legend_y - 10), (300, frame.shape[0] - 10), (255, 255, 255), 1)
        
        cv2.putText(frame, "LEGEND:", (15, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "🔵 Player 1", (15, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
        cv2.putText(frame, "🟢 Player 2", (100, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)
        cv2.putText(frame, "🔴 Ball", (185, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, "🟡 Ball (Est)", (15, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, "--- Trails", (130, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def get_match_summary(self):
        """Get complete match analysis summary"""
        return {
            'total_ball_hits': self.match_stats['ball_hits'],
            'rally_duration_frames': self.match_stats['rally_length'],
            'player_movement_distances': self.match_stats['player_distances'],
            'average_hits_per_rally': self.match_stats['ball_hits'] / max(1, self.match_stats['rally_length'] / 30)  # Assuming 30fps
        }