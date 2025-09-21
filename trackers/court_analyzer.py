import cv2
import numpy as np

class CourtDetector:
    def __init__(self):
        self.court_lines = []
        self.court_corners = []
        self.court_template = None
    
    def detect_court_lines(self, frame):
        """Detect tennis court lines using edge detection and line detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=10)
        
        court_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Filter lines based on length and angle
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 50:  # Minimum line length
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    # Keep mostly horizontal and vertical lines
                    if abs(angle) < 30 or abs(angle) > 150 or (80 < abs(angle) < 100):
                        court_lines.append((x1, y1, x2, y2))
        
        return court_lines
    
    def detect_court_area(self, frame):
        """Detect the main court area"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for court color (typically green, blue, or red)
        # Green court
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        # Blue court
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create masks
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combine masks
        court_mask = cv2.bitwise_or(mask_green, mask_blue)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, kernel)
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_OPEN, kernel)
        
        # Find the largest contour (likely the court)
        contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 10000:  # Minimum area threshold
                return largest_contour
        
        return None
    
    def get_court_zones(self, frame_shape):
        """Define court zones for analysis"""
        height, width = frame_shape[:2]
        
        zones = {
            'baseline_top': (0, 0, width, height//4),
            'service_top': (0, height//4, width, height//2),
            'service_bottom': (0, height//2, width, 3*height//4),
            'baseline_bottom': (0, 3*height//4, width, height),
            'left_side': (0, 0, width//2, height),
            'right_side': (width//2, 0, width, height)
        }
        
        return zones
    
    def draw_court_analysis(self, frame, court_lines=None, court_area=None):
        """Draw court analysis overlay"""
        overlay = frame.copy()
        
        # Draw court lines
        if court_lines:
            for line in court_lines:
                x1, y1, x2, y2 = line
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw court area
        if court_area is not None:
            cv2.drawContours(overlay, [court_area], -1, (255, 255, 0), 2)
        
        # Draw court zones
        zones = self.get_court_zones(frame.shape)
        for zone_name, (x1, y1, x2, y2) in zones.items():
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 128, 128), 1)
            cv2.putText(overlay, zone_name, (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        return result

class MatchAnalyzer:
    def __init__(self):
        self.court_detector = CourtDetector()
        self.rally_count = 0
        self.player_zones = {'Player 1': [], 'Player 2': []}
        self.ball_trajectory = []
    
    def analyze_player_positions(self, player_detections, frame_shape):
        """Analyze player positions relative to court zones"""
        zones = self.court_detector.get_court_zones(frame_shape)
        
        for detection in player_detections:
            for track_id, player_data in detection.items():
                player_label = player_data.get('player_label', f'Person {track_id}')
                bbox = player_data['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Determine which zone the player is in
                current_zone = None
                for zone_name, (x1, y1, x2, y2) in zones.items():
                    if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                        current_zone = zone_name
                        break
                
                if current_zone and player_label in self.player_zones:
                    self.player_zones[player_label].append(current_zone)
    
    def analyze_ball_trajectory(self, ball_detections):
        """Analyze ball movement patterns"""
        for detection in ball_detections:
            if detection:
                for ball_id, ball_data in detection.items():
                    bbox = ball_data['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    self.ball_trajectory.append((center_x, center_y))
    
    def get_advanced_stats(self):
        """Get advanced match statistics"""
        stats = {
            'rally_count': self.rally_count,
            'ball_trajectory_length': len(self.ball_trajectory),
            'player_zone_distribution': {}
        }
        
        # Calculate player zone distribution
        for player, zones in self.player_zones.items():
            if zones:
                zone_counts = {}
                for zone in zones:
                    zone_counts[zone] = zone_counts.get(zone, 0) + 1
                stats['player_zone_distribution'][player] = zone_counts
        
        return stats