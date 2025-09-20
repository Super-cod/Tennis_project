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
    fourcc=cv2.VideoWriter_fourcc(*'MJPG')
    out=cv2.VideoWriter(path, fourcc, 24, (640, 480))
    for frame in frames:
        out.write(frame)
    out.release()
