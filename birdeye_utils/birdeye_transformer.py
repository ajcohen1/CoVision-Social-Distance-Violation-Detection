import cv2
import numpy as np


def transform_center_coords_to_birdeye(center_coords, transformation_matrix):
    perspective_preprocessing_pts = [np.array([[[pt[0], pt[1]]]], dtype="float32") for pt in center_coords]
    warped_pts = [cv2.perspectiveTransform(pt, transformation_matrix)[0][0] for pt in perspective_preprocessing_pts]
    return warped_pts

