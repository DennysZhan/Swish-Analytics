import argparse
import os

from utils.video_utils import read_video, save_video
from trackers import PlayerTracking , BallTracking
from labeling import PlayerTrackingLabeler, BallTrackingLabeler, TeamBallControlLabeler, PassesAndInterceptionsLabeler, CourtKeypointLabeler, TacticalViewLabeler, SpeedAndDistanceLabeler
from team_assigner import TeamAssigner
from ball_acquisition import BallAquisitionDetector
from passes_and_interceptions import PassAndInterceptionDetector
from court_keypoint_detector import CourtKeypointDetector
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from tactical_view_converter import TacticalViewConverter
from configs import STUBS_DEFAULT_PATH, PLAYER_DETECTION_MODEL_PATH, BALL_DETECTION_MODEL_PATH, COURT_KEYPOINT_DETECTION_MODEL_PATH, OUTPUT_VIDEO_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Swish Analytics - Basketball Video Analysis")
    parser.add_argument('input_video', type=str, help='Path to the input video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH, help='Path to the stubs directory')
    parser.add_argument('--output_video_path', type=str, default=OUTPUT_VIDEO_PATH, help='Path to save the output video')
    return parser.parse_args()

def main():

    #Parse command line arguments
    args = parse_args()
    
    #Read video frames from a file
    video_frames = read_video(args.input_video)

    #Initialize the tracking models
    player_tracker = PlayerTracking(PLAYER_DETECTION_MODEL_PATH)
    ball_tracker = BallTracking(BALL_DETECTION_MODEL_PATH)

    #Initialize the court keypoint detector
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTION_MODEL_PATH)

    #Run trackers
    tracked_players = player_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=os.path.join(args.stub_path, "player_tracking_stub.pkl"))
    tracked_ball = ball_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=os.path.join(args.stub_path, "ball_tracking_stub.pkl"))
    court_keypoints = court_keypoint_detector.get_court_keypoints(video_frames, read_from_stub=True, stub_path=os.path.join(args.stub_path, "court_keypoints_stub.pkl"))
    
    print(court_keypoints)
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

    #Detect passes and interceptions
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_acquisition_events, player_assignments)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_acquisition_events, player_assignments)

    #Tactical view converter
    tactical_view_converter = TacticalViewConverter(court_image_path="./images/basketball_court.png")
    court_keypoints = tactical_view_converter.validate_keypoints(court_keypoints)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints, tracked_players)
    
    #Speed and distance calculator
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters,
    )

    player_distance_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distance_per_frame)
    
    #Label the video frames with player and ball tracking information
    #Initialize the labelers
    player_labeler = PlayerTrackingLabeler()
    ball_labeler = BallTrackingLabeler()
    team_ball_control_labeler = TeamBallControlLabeler()
    passes_and_interceptions_labeler = PassesAndInterceptionsLabeler()
    court_keypoint_labeler = CourtKeypointLabeler()
    speed_and_distance_labeler = SpeedAndDistanceLabeler()
    tactical_view_labeler = TacticalViewLabeler()

    #Label the video frames
    output_video_frames = player_labeler.label(video_frames, tracked_players, player_assignments, ball_acquisition_events)
    output_video_frames = ball_labeler.label(output_video_frames, tracked_ball)

    #Label the team ball control information
    output_video_frames = team_ball_control_labeler.label(output_video_frames, player_assignments, ball_acquisition_events)

    #Label passes and interceptions
    output_video_frames = passes_and_interceptions_labeler.label(output_video_frames, passes, interceptions)

    #Label court keypoints
    output_video_frames = court_keypoint_labeler.label(output_video_frames, court_keypoints)

    #Label tactical view
    output_video_frames = tactical_view_labeler.label(output_video_frames, 
                                                        tactical_view_converter.court_image_path, 
                                                        tactical_view_converter.width, 
                                                        tactical_view_converter.height, 
                                                        tactical_view_converter.key_points,
                                                        tactical_player_positions,
                                                        player_assignments,
                                                        ball_acquisition_events
                                                        )
    #Label speed and distance
    output_video_frames = speed_and_distance_labeler.label(output_video_frames,
                                                           tracked_players,
                                                           player_distance_per_frame,
                                                           player_speed_per_frame
                                                           )       

    #Save the video frames to a new file
    save_video(output_video_frames, args.output_video_path)

if __name__ == "__main__":
    main()

