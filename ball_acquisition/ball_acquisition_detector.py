import sys
sys.path.append("../")
from utils.bbox_utils import measure_distance, get_bbox_center


class BallAquisitionDetector:

    def __init__(self):
        self.possession_threshold = 50
        self.minimum_frames = 11
        self.containment_threshold = 0.8

    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):

        ball_center_x = ball_center[0]
        ball_center_y = ball_center[1]

        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1

        output_points = []

        if ball_center_y > y1 and ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))

        if ball_center_x > x1 and ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))
        
        output_points += [
            (x1, y1), # top left corner
            (x2, y1), # top right corner
            (x1, y2), # bottom left corner
            (x2, y2),  # bottom right corner
            (x1 + width //2, y1), # top center
            (x1 + width //2, y2),  # bottom center
            (x1, y1 + height //2), # left center
            (x2, y1 + height //2)  # right center
        ]

        return output_points
    

    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        """
        Finds the minimum distance from the player's bounding box to the ball center.
        
        Args:
            player_bbox: A tuple containing the player's bounding box coordinates (x1, y1, x2, y2).
            ball_center: A tuple containing the ball's center coordinates (x, y).
        
        Returns:
            The minimum distance from the player's bounding box to the ball center.
        """
        key_points = self.get_key_basketball_player_assignment_points(player_bbox, ball_center)
        
        #return mine(measure_distance(point, ball_center) for point in key_points)
        
        min_distance = float('inf')
        for point in key_points:
            distance = measure_distance(point, ball_center)
            if min_distance > distance:
                min_distance = distance
        
        return min_distance
    
    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):

        player_x1, player_y1, player_x2, player_y2 = player_bbox
        ball_x1, ball_y1, ball_x2, ball_y2 = ball_bbox

        ball_area = (ball_x2 - ball_x1) * (ball_y2 - ball_y1)
        
        intersection_x1 = max(player_x1, ball_x1)
        intersection_y1 = max(player_y1, ball_y1)
        intersection_x2 = min(player_x2, ball_x2)
        intersection_y2 = min(player_y2, ball_y2)

        if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
            return 0.0

        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)

        containment_ratio = intersection_area / ball_area

        return containment_ratio
    
    def find_best_candidate_for_possession(self, ball_center, player_tracks_frame, ball_bbox):

        high_containment_players = []
        regular_distance_players = []

        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get("bbox", [])
            if not player_bbox:
                continue

            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            minimum_distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)

            if containment > self.containment_threshold:
                high_containment_players.append((player_id, minimum_distance))
            else:
                regular_distance_players.append((player_id, minimum_distance))

        # high containment players are preferred
        if high_containment_players:
            best_candidate = max(high_containment_players, key=lambda x: x[1])
            return best_candidate[0]
        
        # regular distance players are considered
        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda x: x[1])
            if best_candidate[1] < self.possession_threshold:
                return best_candidate[0]
            
        return -1
    
    def detect_ball_possession(self, player_tracks, ball_tracks):

        number_of_frames = len(ball_tracks)
        possession_list = [-1] * number_of_frames

        consecutive_possession_count = {}

        for frame_number in range(number_of_frames):
            ball_info = ball_tracks[frame_number].get(1, {})
            if not ball_info:
                continue

            ball_bbox = ball_info.get("bbox", [])
            if not ball_bbox:
                continue

            ball_center = get_bbox_center(ball_bbox)

            best_player_id = self.find_best_candidate_for_possession(ball_center, player_tracks[frame_number], ball_bbox)

            if best_player_id != -1:
                number_of_consecutive_frames = consecutive_possession_count.get(best_player_id, 0) + 1
                consecutive_possession_count = {best_player_id: number_of_consecutive_frames}

                if consecutive_possession_count[best_player_id] >= self.minimum_frames:
                    possession_list[frame_number] = best_player_id

            else: 
                consecutive_possession_count = {}

        return possession_list