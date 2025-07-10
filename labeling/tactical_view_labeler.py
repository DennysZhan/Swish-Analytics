import cv2

class TacticalViewLabeler:

    def __init__(self, team_1_color = [255, 245, 238], team_2_color=[128, 0, 0]):
        self.start_x = 20
        self.start_y = 40
        self.team_1_color = team_1_color 
        self.team_2_color = team_2_color 

    def label(self, frames, court_image_path, width, height, tactical_court_keypoints, tactical_player_positions=None, player_assignments=None, ball_acquisition_point = None):

        court_image = cv2.imread(court_image_path)
        court_image = cv2.resize(court_image, (width, height))

        output_frames = []
        for frame_index, frame in enumerate(frames):
            frame = frame.copy()
            
            y1 = self.start_y
            x1 = self.start_x
            y2 = y1 + height
            x2 = x1 + width

            alpha = 0.6
            
            overlay = frame[y1:y2,x1:x2].copy()
            cv2.addWeighted(court_image, alpha, overlay, 1 - alpha, 0, overlay)

            for keypoint_index, keypoint in enumerate(tactical_court_keypoints):
                x, y = keypoint
                x+= self.start_x
                y+= self.start_y
                cv2.circle(frame, (x,y), 5, (0,0,255), -1)
                cv2.putText(frame, str(keypoint_index), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            if tactical_player_positions and player_assignments and frame_index < len(tactical_player_positions):
                frame_positions = tactical_player_positions[frame_index]
                frame_assignment = player_assignments[frame_index] if frame_index < len(player_assignments) else {}
                player_with_ball = ball_acquisition_point[frame_index] if ball_acquisition_point and frame_index < len(ball_acquisition_point) else None

                for player_id, position in frame_positions.items():
                    team_id = frame_assignment.get(player_id, 1)
                    color = self.team_1_color if team_id == 1 else self.team_2_color

                    x,y = int(position[0] + self.start_x), int(position[1] + self.start_y)

                    player_radius = 8
                    cv2.circle(frame, (x,y), player_radius, color, -1)

                    if player_id == player_with_ball:
                        cv2.circle(frame, (x,y), player_radius + 3, (0, 0, 255), 2)

            output_frames.append(frame)
        
        return output_frames


