from utils.video_utils import read_video, save_video
from trackers import PlayerTracking , BallTracking
from labeling import PlayerTrackingLabeler, BallTrackingLabeler


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

    #Label the tracked players in the video frames
    player_labeler = PlayerTrackingLabeler()
    ball_labeler = BallTrackingLabeler()

    output_video_frames = player_labeler.label(video_frames, tracked_players)
    output_video_frames = ball_labeler.label(output_video_frames, tracked_ball)

    #Save the video frames to a new file
    save_video(output_video_frames, "output_videos/output_video.mp4")

if __name__ == "__main__":
    main()

