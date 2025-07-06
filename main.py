from utils.video_utils import read_video, save_video
from trackers import PlayerTracking , BallTracking
from labeling import PlayerTrackingLabeler, BallTrackingLabeler, TeamBallControlLabeler
from team_assigner import TeamAssigner
from ball_acquisition import BallAquisitionDetector



def main():
    
    #Read video frames from a file
    video_frames = read_video("input_videos/video_1.mp4")

    #Initialize the tracking models
    player_tracker = PlayerTracking("models/player_detection.pt")
    ball_tracker = BallTracking("models/basketball_detection.pt")

    #Run trackers
    tracked_players = player_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/player_tracking_stub.pkl")
    tracked_ball = ball_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/ball_tracking_stub.pkl")

    #Remove wrong tracked ball positions
    tracked_ball = ball_tracker.remove_wrong_detections(tracked_ball)

    #Interpolate missing ball positins
    tracked_ball = ball_tracker.interpolate_ball_positions(tracked_ball)

    #Assign teams to players
    team_assigner = TeamAssigner()
    player_assignments = team_assigner.get_player_teams_across_frames(video_frames, tracked_players, read_from_stub=True, stub_path="stubs/player_assignments_stub.pkl")

    #Detect ball acquisition events
    ball_acquisition_detector = BallAquisitionDetector()
    ball_acquisition_events = ball_acquisition_detector.detect_ball_possession(tracked_players, tracked_ball)

    #Label the video frames with player and ball tracking information
    #Initialize the labelers
    player_labeler = PlayerTrackingLabeler()
    ball_labeler = BallTrackingLabeler()
    team_ball_control_labeler = TeamBallControlLabeler()

    #Label the video frames
    output_video_frames = player_labeler.label(video_frames, tracked_players, player_assignments, ball_acquisition_events)
    output_video_frames = ball_labeler.label(output_video_frames, tracked_ball)

    #Label the team ball control information
    output_video_frames = team_ball_control_labeler.label(output_video_frames, player_assignments, ball_acquisition_events)

    #Save the video frames to a new file
    save_video(output_video_frames, "output_videos/output_video.mp4")

if __name__ == "__main__":
    main()

