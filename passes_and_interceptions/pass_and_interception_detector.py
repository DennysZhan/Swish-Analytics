class PassAndInterceptionDetector:

    def __init__(self):
        pass

    def detect_passes(self, ball_acquisition_events, player_assignments):
        passes = [-1] * len(ball_acquisition_events)
        previous_holder = -1
        previous_frame = -1

        for frame in range (1, len(ball_acquisition_events)):

            if ball_acquisition_events[frame - 1] != -1:
                previous_holder = ball_acquisition_events[frame - 1]
                previous_frame = frame -1

            current_holder = ball_acquisition_events[frame]

            if previous_holder != -1 and current_holder != -1 and previous_holder != current_holder:
                previous_team = player_assignments[previous_frame].get(previous_holder, -1)
                current_team = player_assignments[frame].get(current_holder, -1)

                if previous_team == current_team and previous_team != -1:
                    passes[frame] = previous_team
        
        return passes
    
    def detect_interceptions(self, ball_acquisition_events, player_assignments):
        interceptions = [-1] * len(ball_acquisition_events)
        previous_holder = -1
        previous_frame = -1

        for frame in range (len(ball_acquisition_events)):

            if ball_acquisition_events[frame - 1] != -1:
                previous_holder = ball_acquisition_events[frame - 1]
                previous_frame = frame -1

            current_holder = ball_acquisition_events[frame]

            if previous_holder != -1 and current_holder != -1 and previous_holder != current_holder:
                previous_team = player_assignments[previous_frame].get(previous_holder, -1)
                current_team = player_assignments[frame].get(current_holder, -1)

                if previous_team != current_team and previous_team != -1 and current_team != -1:
                    interceptions[frame] = current_team
        
        return interceptions
    



        