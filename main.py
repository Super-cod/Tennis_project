from ultralytics import YOLO
from utils import (read_video, save_video)
from trackers.tennis_tracker import TennisTracker
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
import os

def main():
    print("🎾 Tennis Match Analysis System")
    print("=" * 50)
    
    # Input video path
    input_video_path = 'input_video.mp4'
    
    # Check if video exists
    if not os.path.exists(input_video_path):
        print(f"❌ Error: Video file '{input_video_path}' not found!")
        return
    
    # Check if models exist
    player_model = "yolov8n.pt" 
    ball_model = "models/best.pt"
    
    if not os.path.exists(ball_model):
        print(f"❌ Error: Ball detection model '{ball_model}' not found!")
        return
    
    print(f"📹 Loading video: {input_video_path}")
    video_frames = read_video(input_video_path)
    print(f"✅ Loaded {len(video_frames)} frames")
    
    # COMBINED TENNIS ANALYSIS - ALL IN ONE VIDEO
    print("\n🔄 Starting COMPLETE tennis match analysis...")
    print("🎾 Tracking players and ball simultaneously...")
    
    tennis_tracker = TennisTracker(player_model_path=player_model, ball_model_path=ball_model)
    
    # Track both players and ball together
    player_detections, ball_detections = tennis_tracker.track_tennis_match(video_frames)
    
    # Create ONE comprehensive analysis video with EVERYTHING
    print("🎨 Creating ULTIMATE tennis analysis video...")
    print("   📍 2 Player tracking with trails")
    print("   🎾 Ball tracking with trajectory")
    print("   📊 Live match statistics")
    print("   🏆 Complete analysis overlay")
    
    output_video_frames = tennis_tracker.draw_complete_analysis(video_frames, player_detections, ball_detections)
    
    # Save the ultimate combined video
    ultimate_output_path = 'Output_videos/ULTIMATE_tennis_analysis.avi'
    os.makedirs('Output_videos', exist_ok=True)
    save_video(output_video_frames, ultimate_output_path)
    print(f"✅ ULTIMATE analysis saved: {ultimate_output_path}")
    
    # Optional: 
    create_separate = input("\n🔄 Do you want separate player/ball videos too? (y/n): ").lower().strip()
    
    if create_separate == 'y':
        print("\n🔄 Creating additional separate analysis videos...")
        
        # Player-only tracking
        player_tracker = PlayerTracker(player_model)
        player_detections_only = player_tracker.detect_frames(video_frames)
        player_detections_only = player_tracker.classify_players(player_detections_only)
        player_output_frames = player_tracker.draw_player_tracking(video_frames, player_detections_only)
        save_video(player_output_frames, 'Output_videos/tennis_players_only.avi')
        print("✅ Player tracking saved: Output_videos/tennis_players_only.avi")
        
        # Ball-only tracking
        ball_tracker = BallTracker(ball_model)
        ball_detections_only = ball_tracker.detect_frames(video_frames)
        ball_detections_interpolated = ball_tracker.interpolate_ball_positions(ball_detections_only)
        ball_output_frames = ball_tracker.draw_ball_tracking(video_frames, ball_detections_interpolated)
        save_video(ball_output_frames, 'Output_videos/tennis_ball_only.avi')
        print("✅ Ball tracking saved: Output_videos/tennis_ball_only.avi")
    
    # Print match summary
    print("\n📊 MATCH ANALYSIS SUMMARY")
    print("=" * 50)
    summary = tennis_tracker.get_match_summary()
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n🎉 Analysis complete! Check the Output_videos folder for results.")

def analyze_models():
    """Test and compare model performance"""
    print("\n🔍 MODEL ANALYSIS")
    print("=" * 30)
    
    models_to_test = {
        "YOLOv8n (General)": "yolov8n.pt",
        "Best Model (Tennis Ball)": "models/best.pt",
        "Last Model (Tennis Ball)": "models/last.pt"
    }
    
    for name, path in models_to_test.items():
        if os.path.exists(path):
            try:
                model = YOLO(path)
                print(f"✅ {name}: {path}")
                print(f"   Classes: {list(model.names.values())}")
            except Exception as e:
                print(f"❌ {name}: Error loading - {e}")
        else:
            print(f"❌ {name}: File not found - {path}")

if __name__ == "__main__":
    # Analyze available models
    analyze_models()
    
    # Run main analysis
    main()