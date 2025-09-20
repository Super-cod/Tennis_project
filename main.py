from ultralytics import YOLO
from utils import (read_video,

            save_video)

from trackers import player_tracker


def main():
    input_video_path="input_video.mp4"
    video_frame=read_video(input_video_path)
    save_video(video_frame, "Output_videos/output_video.avi")

if __name__ == "__main__":
    main()