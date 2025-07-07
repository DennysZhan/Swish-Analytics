from labeling.utils import draw_arrow

class BallTrackingLabeler:
    def __init__(self):
        self.ball_pointer_color = (0, 255, 0)

    def label(self, frames, tracks):
        """
        Label ball tracks in the provided video frames.
        Args:
            frames: List of video frames to process.
            tracks: List of dictionaries containing ball track information for each frame.
        Returns:
            List of labeled video frames with ball tracks drawn.
        """
        output_video_frames = []
        for frame_number, frame in enumerate(frames):
            frame = frame.copy()
            ball_dict = tracks[frame_number]

            # Label Ball tracks
            for _, track in ball_dict.items():
                bbox = track['bbox']
                if bbox is None:
                    continue
                frame = draw_arrow(frame, bbox, self.ball_pointer_color)

            output_video_frames.append(frame)

        return output_video_frames   