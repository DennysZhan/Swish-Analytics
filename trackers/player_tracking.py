from ultralytics import YOLO
import supervision as sv
import sys
sys.path.append("../")
from utils.stubs_utils import read_stub, save_stub

class PlayerTracking:
    def __init__(self, model_path):
        """
        Initializes the PlayerTracking class with a YOLO model for player detection.

        Args:
            model_path: Path to the YOLO model file.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        """
        Detects players in the provided video frames.

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

            detection_with_tracking = self.tracker.update_with_detections(detection_supervision)

            tracks.append({})

            for frame_detection in detection_with_tracking:
                track_id = frame_detection[4]
                class_id = frame_detection[3]
                bbox = frame_detection[0].tolist()

                if class_id == class_names_inverse['Player']:
                    tracks[frame_number][track_id] = bbox
        
        save_stub(stub_path, tracks)

        return tracks
