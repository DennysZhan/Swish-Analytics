import cv2

class SpeedAndDistanceLabeler:
    def __init__(self):
        pass

    def label(self, video_frames, tracked_players, player_distance_per_frame, player_speed_per_frame):
        
        """
        Annotate video frames with player speed and distance traveled.
        Args:
            video_frames: List of video frames to process.
            tracked_players: Dictionary of player tracks, where each key is a player ID and the value is a dictionary containing the bounding box and other attributes.
            player_distance_per_frame: Dictionary mapping player IDs to their distance traveled in each frame.
            player_speed_per_frame: Dictionary mapping player IDs to their speed in each frame.
        Returns:
            List of annotated video frames with player speed and distance drawn."""
        
        
        output_video_frames = []
        total_distance = {}

        for frame, tracked_players, player_distance, player_speed in zip(video_frames, tracked_players, player_distance_per_frame, player_speed_per_frame):
            output_frame = frame.copy()

            for player_id, distance in player_distance.items():
                if player_id not in total_distance:
                    total_distance[player_id] = 0
                total_distance[player_id] += distance

            for player_id, bbox in tracked_players.items():
                x1, y1, x2, y2 = bbox['bbox']
                position = [int((x1 + x2) / 2), int(y2)]
                position[1] += 40

                distance = total_distance.get(player_id, None)
                speed = player_speed.get(player_id, None)

                if speed is not None:
                    cv2.putText(output_frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                if distance is not None:
                    cv2.putText(output_frame, f"{distance:.2f} m", [position[0], position[1] + 20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_video_frames.append(output_frame)

        return output_video_frames