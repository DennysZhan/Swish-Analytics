from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from utils.stubs_utils import read_stub, save_stub


class BallTracking:
    def __init__(self, model_path):
        """
        Initializes the BallTracking class with a YOLO model for ball detection.
        """
        self.model = YOLO(model_path)
    
    def detect_frames(self, frames):
        """
        Detects ball in the provided video frames.

        Args:
            frames: List of video frames to process.

        Returns:
            List of detected players in each frame.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_detections = self.model.predict(batch_frames, conf=0.5)
            detections += batch_detections           
        return detections
    
    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):
        """
        Tracks objects in the provided video frames.

        Args:
            frames: List of video frames to process.
            read_from_stub: Boolean indicating whether to read from a stub file.
            stub_path: Path to the stub file.
        
        Returns:
            List of tracks for each frame, where each track is a dictionary mapping track IDs to bounding
        """

        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for frame_number, frame_detections in enumerate(detections):
            class_names = frame_detections.names
            class_names_inverse = {value : key for key, value in class_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(frame_detections)

            tracks.append({})
            chosen_bbox = None
            max_confidence = 0


            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                confidence = frame_detection[2]

                if class_id == class_names_inverse['Ball']:
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_number][1] = {"bbox": chosen_bbox}
        
        save_stub(stub_path, tracks)

        return tracks

    def remove_wrong_detections(self, ball_positions):
        """
        Removes detections that are not valid ball positions.
        Args:
            ball_positions: List of ball positions to filter.
        Returns:
            List of valid ball positions.
        """

        maximum_distance_allowed = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_bbox = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_bbox) == 0:
                continue

            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            last_good_bbox = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            frame_gap =i - last_good_frame_index
            adjusted_max_distance = maximum_distance_allowed * frame_gap

            if np.linalg.norm(np.array(last_good_bbox[:2]) - np.array(current_bbox[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i
        
        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates missing ball positions in the provided list of ball positions.
        
        Args:
            ball_positions: List of ball positions to interpolate.
        
        Returns:
            List of ball positions with interpolated values.
        """
        
        ball_positions = [x.get(1,{}).get('bbox', []) for x in ball_positions ]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        #Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:{"bbox" : x }} for x in df_ball_positions.to_numpy().tolist() ]

        return ball_positions







        
    

