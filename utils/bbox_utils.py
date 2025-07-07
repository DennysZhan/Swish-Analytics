def get_bbox_center(bbox):
    """
    Calculates the center of a bounding box.

    Args:
        bbox: A tuple containing the bounding box coordinates (x1, y1, x2, y2).

    Returns:
        A tuple representing the center coordinates (center_x, center_y).
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def get_bbox_width(bbox):
    """
    Calculates the width of a bounding box.

    Args:
        bbox: A tuple containing the bounding box coordinates (x, y, width, height).

    Returns:
        The width of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    return x2 - x1

def measure_distance(p1, p2):
    """
    Measures the Euclidean distance between two points.
    Args:
        p1: A tuple representing the first point (x1, y1).
        p2: A tuple representing the second point (x2, y2).
    Returns:
        The Euclidean distance between the two points.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_foot_position(bbox):
    """
    Calculates the foot position from a bounding box.

    Args:
        bbox: A tuple containing the bounding box coordinates (x1, y1, x2, y2).

    Returns:
        A tuple representing the foot position (foot_x, foot_y).
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)