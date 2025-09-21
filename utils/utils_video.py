import cv2

def read_video(path):
    cap = cv2.VideoCapture(path)
    frames=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, path):
    if not frames:
        return
    
    # Get dimensions from first frame instead of hardcoding
    height, width = frames[0].shape[:2]
    fourcc=cv2.VideoWriter_fourcc(*'MJPG')
    out=cv2.VideoWriter(path, fourcc, 24, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
