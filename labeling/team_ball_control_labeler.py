import numpy as np
import cv2

class TeamBallControlLabeler:

    def __init__(self):
        pass

    def get_team_ball_control(self, player_assignments, ball_acquisition_events):

        team_ball_control = []
        for player_assignment_frame, ball_acqusition_frame in zip(player_assignments, ball_acquisition_events):
            if ball_acqusition_frame == -1:
                team_ball_control.append(-1)
                continue

            if ball_acqusition_frame not in player_assignment_frame:
                team_ball_control.append(-1)
                continue

            if player_assignment_frame[ball_acqusition_frame] == 1:
                team_ball_control.append(1)
            else:
                team_ball_control.append(2)

        team_ball_control = np.array(team_ball_control)
        return team_ball_control
    


    def label(self, frames, player_assignments, ball_acquisition_events):

        team_ball_control = self.get_team_ball_control(player_assignments, ball_acquisition_events)

        output_video_frames = []
        for frame_number, frame in enumerate(frames):
            if frame_number == 0:
                continue

            frame_drawn = self.draw_frame(frame, frame_number, team_ball_control)
            output_video_frames.append(frame_drawn)

        return output_video_frames
    
    def draw_frame(self, frame, frame_number, team_ball_control):

        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2

        #O verlay position
        frame_height, frame_width = overlay.shape[0:2]
        rectangle_x1 = int(frame_width * 0.60)
        rectangle_y1 = int(frame_height * 0.75)
        rectangle_x2 = int(frame_width * 0.99)
        rectangle_y2 = int(frame_height * 0.90)

        # Text position
        text_x = int(frame_width * 0.63)
        text_y1 = int(frame_height * 0.80)
        text_y2 = int(frame_height * 0.88)

        cv2.rectangle(overlay, 
                      (rectangle_x1, rectangle_y1), 
                      (rectangle_x2, rectangle_y2), 
                      (255, 255, 255), 
                      -1)
       
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_number + 1]
        team_1_number_of_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_number_of_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        team_1_percentage = team_1_number_of_frames / (team_ball_control_till_frame.shape[0]) * 100
        team_2_percentage = team_2_number_of_frames / (team_ball_control_till_frame.shape[0]) * 100

        cv2.putText(frame, 
                    f"Team 1 Ball Control: {team_1_percentage:.2f}%", 
                    (text_x, text_y1), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (0, 0, 0), 
                    font_thickness)
        cv2.putText(frame, 
                    f"Team 2 Ball Control: {team_2_percentage:.2f}%", 
                    (text_x, text_y2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (0, 0, 0), 
                    font_thickness)
        
        return frame



    


