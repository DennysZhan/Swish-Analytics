import cv2
import os

def read_video(video_path):
    cap=cv2.VideoCapture(video_path)
    frames = []
    while True:
        returned,frame = cap.read()
        if not returned:
            break
        frames.append(frame)
    return frames

def save_video(frames, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, 24.0, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()