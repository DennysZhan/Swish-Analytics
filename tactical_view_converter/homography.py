import numpy as np
import cv2

class Homography:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        if source.shape != target.shape:
            raise ValueError("Source and target points must have the same shape.")
        
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D points.")
        
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.m, _ =cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography could not be computed. Check the input points.")
        
    def transform_points(self, points):
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D points.")
        
        points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(points, self.m)
        return transformed_points.reshape(-1, 2).astype(np.float32)
        
       