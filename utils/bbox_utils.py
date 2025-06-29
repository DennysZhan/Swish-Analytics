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