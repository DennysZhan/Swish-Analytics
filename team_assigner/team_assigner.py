from PIL import Image
import cv2
import sys
sys.path.append("../")
from utils.stubs_utils import read_stub, save_stub

from transformers import CLIPProcessor, CLIPModel

class TeamAssigner:
    def __init__(self, team_1_class_name = "white basketball jersey", team_2_class_name = "dark blue basketball jersey"):
        """
        Initializes the TeamAssigner with the class names for each team.
        
        Args:
            team_1_class_name: Class name for team 1.
            team_2_class_name: Class name for team 2.
        """

        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

        self.player_team_dict = {}

    def load_model(self):
        """
        Loads the CLIP model and processor.
        
        Returns:
            model: The loaded CLIP model.
            processor: The loaded CLIP processor.
        """
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


    def get_player_color(self, frame, bbox):
        """
        Determines the team color of a player based on their bounding box in the frame.
        
        Args:
            frame: The video frame as a PIL Image.
            bbox: The bounding box coordinates (x1, y1, x2, y2).
        
        Returns:
            team_color: The team color of the player ("team_1" or "team_2").
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        image = pil_image

        classes = [self.team_1_class_name, self.team_2_class_name]

        inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) 

        class_name =  classes[probs.argmax(dim=1)[0]]
        return class_name
    
    def get_player_team(self, frame, player_bbox, player_id):
        """
        Assigns a team to a player based on their bounding box in the frame.
        
        Args:
            frame: The video frame as a PIL Image.
            player_bbox: The bounding box coordinates (x1, y1, x2, y2) for the player.
            player_id: The ID of the player.
        
        Returns:
            team_color: The team color of the player ("team_1" or "team_2").
        """

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = 2
        if player_color == self.team_1_class_name:
            team_id = 1
        
        self.player_team_dict[player_id] = team_id

        return team_id
    
    def get_player_teams_across_frames(self, frames, player_tracks, read_from_stub=False, stub_path=None):
        """
        Assigns teams to players across multiple frames.
        
        Args:
            frames: List of video frames to process.
            player_tracks: List of tracks for each frame, where each track is a dictionary mapping track IDs to bounding boxes.
            read_from_stub: Boolean indicating whether to read from a stub file.
            stub_path: Path to the stub file.
        
        Returns:
            List of team assignments for each player in each frame.
        """

        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(frames):
                return player_assignment

        self.load_model()

        player_assignments = []

        for frame_number, player_track in enumerate(player_tracks):
            player_assignments.append({})

            if frame_number % 10 ==0:
                self.player_team_dict = {}

            for player_id, track in player_track.items():
                team = self.get_player_team(frames[frame_number], track['bbox'], player_id)
                player_assignments[frame_number][player_id] = team
        
        save_stub(stub_path, player_assignments)

        return player_assignments
    
            