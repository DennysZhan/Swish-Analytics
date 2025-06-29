from utils.video_utils import read_video, save_video
from trackers.player_tracking import PlayerTracking
from labeling import PlayerTrackingLabeler

def main():
    
    #Read video frames from a file
    video_frames = read_video("input_videos/video_1.mp4")

    #Initialize the player tracking model
    player_tracker = PlayerTracking("models/player_detection.pt")

    #Run trackers
    tracked_players = player_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/player_tracking_stub.pkl")

    #Label the tracked players in the video frames
    player_labeler = PlayerTrackingLabeler()
    output_video_frames = player_labeler.label(video_frames, tracked_players)

    #Save the video frames to a new file
    save_video(output_video_frames, "output_videos/output_video_1.mp4")

if __name__ == "__main__":
    main()

