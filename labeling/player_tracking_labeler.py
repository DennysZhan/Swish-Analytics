from labeling.utils import draw_ellipse

class PlayerTrackingLabeler:
    """
    A class to label player tracking data in video frames.
    This class extends the PlayerTracking class to include labeling functionality.
    """

    def __init__(self):
        """
        Initializes the PlayerTrackingLabeler with a model path.
        """
        pass

    def label(self, frames, tracks):

        output_video_frames = []
        for frame_number, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks[frame_number]

            #Label Players tracks
            for track_id, player in player_dict.items():
                frame = draw_ellipse(frame, player, (0,0,255), track_id)

            output_video_frames.append(frame)
            
        return output_video_frames
        
