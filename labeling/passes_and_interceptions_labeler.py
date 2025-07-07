import cv2

class PassesAndInterceptionsLabeler:

    def __init__ (self):
        pass

    def get_stats(self, passes, interceptions):
        """
        Calculate the number of passes and interceptions for each team.
        Args:
            passes: List of pass events, where each event is represented by an integer (1 for team 1, 2 for team 2).
            interceptions: List of interception events, where each event is represented by an integer (1 for team 1, 2 for team 2).
        Returns:
            Tuple containing the number of passes and interceptions for each team:
        """

        team1_passes = []
        team2_passes = []
        team1_interceptions = []
        team2_interceptions = []

        for frame_number, (pass_frame, interception_frame) in enumerate(zip(passes, interceptions)):
            if pass_frame == 1:
                team1_passes.append(frame_number)
            elif pass_frame == 2:
                team2_passes.append(frame_number)
                    
            if interception_frame == 1:
                team1_interceptions.append(frame_number)
            elif interception_frame == 2:
                team2_interceptions.append(frame_number)
                    
        return len(team1_passes), len(team2_passes), len(team1_interceptions), len(team2_interceptions)
        

    def label(self, frames, passes, interceptions):
        """
        Annotate video frames with the number of passes and interceptions for each team.
        Args:
            frames: List of video frames to process.
            passes: List of pass events, where each event is represented by an integer (1 for team 1, 2 for team 2).
            interceptions: List of interception events, where each event is represented by an integer (1 for team 1, 2 for team 2).
        Returns:
            List of annotated video frames with passes and interceptions drawn.
        """
        
        output_video_frames = []

        for frame_number, frame in enumerate(frames):
            if frame_number == 0:
                continue

            frame_drawn = self.draw_frame(frame, frame_number, passes, interceptions)
            output_video_frames.append(frame_drawn)

        return output_video_frames
    
    def draw_frame(self, frame, frame_number, passes, interceptions):

        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2

        #O verlay position
        frame_height, frame_width = overlay.shape[:2]
        rectangle_x1 = int(frame_width * 0.16)
        rectangle_y1 = int(frame_height * 0.75)
        rectangle_x2 = int(frame_width * 0.55)
        rectangle_y2 = int(frame_height * 0.90)

        # Text position
        text_x = int(frame_width * 0.19)
        text_y1 = int(frame_height * 0.80)
        text_y2 = int(frame_height * 0.88)

        cv2.rectangle(overlay, 
                      (rectangle_x1, rectangle_y1), 
                      (rectangle_x2, rectangle_y2), 
                      (255, 255, 255), 
                      -1)
       
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        passes_till_frame = passes[:frame_number+1]
        interceptions_till_frame = interceptions[:frame_number+1]
        
        team1_passes, team2_passes, team1_interceptions, team2_interceptions = self.get_stats(
            passes_till_frame, 
            interceptions_till_frame
        )

        cv2.putText(
            frame, 
            f"Team 1 - Passes: {team1_passes} Interceptions: {team1_interceptions}",
            (text_x, text_y1), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (0,0,0), 
            font_thickness
        )
        
        cv2.putText(
            frame, 
            f"Team 2 - Passes: {team2_passes} Interceptions: {team2_interceptions}",
            (text_x, text_y2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (0,0,0), 
            font_thickness
        )
        return frame
