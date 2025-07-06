from labeling.utils import draw_ellipse, draw_arrow

class PlayerTrackingLabeler:
    """
    A class to label player tracking data in video frames.
    This class extends the PlayerTracking class to include labeling functionality.
    """

    def __init__(self, team_1_color = [255, 245, 238], team_2_color = [128,0,0]): #bgr colors
        """
        Initializes the PlayerTrackingLabeler with a model path.
        """
        self.default_player_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color



    def label(self, frames, tracks, player_assignments, ball_acquisition_events):

        """        Draws player tracks on the video frames.
        Args:
            frames: List of video frames to process.
            tracks: List of tracks for each frame, where each track is a dictionary mapping track IDs to bounding boxes.
            player_assignments: List of team assignments for each player in each frame.
        Returns:
            List of video frames with player tracks drawn.
        """

        output_video_frames = []
        for frame_number, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks[frame_number]

            player_assignment_for_frame = player_assignments[frame_number]

            player_id_has_ball = ball_acquisition_events[frame_number]

            #Label Players tracks
            for track_id, player in player_dict.items():
                team_id = player_assignment_for_frame.get(track_id, self.default_player_team_id)

                if team_id == 1:
                    color = self.team_1_color
                else:
                    color = self.team_2_color

                if track_id == player_id_has_ball:
                    frame = draw_arrow(frame, player['bbox'], (0,0,255))

                frame = draw_ellipse(frame, player['bbox'], color, track_id)

            output_video_frames.append(frame)
            
        return output_video_frames
        
