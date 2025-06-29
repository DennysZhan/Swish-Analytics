import cv2
import sys
sys.path.append("../")
from utils import get_bbox_center, get_bbox_width

def draw_ellipse(frame, bbox, color=(0, 255, 0), track_id = None):
    """
    Draws an ellipse on the frame based on the bounding box coordinates.
    
    Parameters:
        frame: The video frame on which to draw.
        bbox: A tuple containing the bounding box coordinates (x1, y1, x1, y2).
        color): The color of the ellipse in BGR format.
        thickness: The thickness of the ellipse outline.
    
    Returns:
        The frame with the drawn ellipse.
    """
    
    x_center, _ = get_bbox_center(bbox)
    width = get_bbox_width(bbox)
    cv2.ellipse(frame, 
                center=(int(x_center), int(bbox[3])), 
                axes=(int(width), int(0.35 * width)), 
                angle=0, 
                startAngle=-45, 
                endAngle=235, 
                color=color, 
                thickness=2, 
                lineType=cv2.LINE_4)
    
    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (bbox[3] - (rectangle_height//2)) + 15
    y2_rect = (bbox[3] + (rectangle_height//2)) + 15

    if track_id is not None:
        cv2.rectangle(frame,
                     (int(x1_rect),int(y1_rect)),
                     (int(x2_rect), int(y2_rect)),
                     color,
                     cv2.FILLED)
        
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(frame,
                   str(track_id),
                   (int(x1_text), int(bbox[3] + 15)),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6,
                   (0, 0, 0), 
                   2)

    

    return frame