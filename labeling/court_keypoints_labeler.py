import supervision as sv


class CourtKeypointLabeler:

    def __init__ (self): 
        self.keypoint_color = "#ff2c2c"

    def label(self, frames, court_keypoints):
        """
        Annotate court keypoints on the provided video frames.
        Args:
            frames: List of video frames to process.
            court_keypoints: List of court keypoints for each frame, where each keypoint is a list of coordinates.
        Returns:
            List of annotated video frames with court keypoints drawn.
        """
        
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=8,
        )

        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color= sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )

        output_frames = []
        for index, frame in enumerate(frames):
            annotate_frame = frame.copy()

            keypoints = court_keypoints[index]
            annotate_frame = vertex_annotator.annotate(
                scene=annotate_frame,
                key_points=keypoints
            )

            keypoints_numpy = keypoints.cpu().numpy()
            annotate_frame = vertex_label_annotator.annotate(
                scene=annotate_frame,
                key_points=keypoints_numpy
            )

            output_frames.append(annotate_frame)

        return output_frames

