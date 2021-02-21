import cv2
import numpy as np


def compute_color_for_labels(risk_factor):
    if risk_factor > 1:
        return [0, 0, 255]
    return [0, 128, 0]

class birdeye_video_writer:
    def __init__(self, frame_h, frame_w, transformation_matrix, threshold_pixel_dist, ROI):
        self.background = cv2.imread('testpic.jpg', cv2.IMREAD_COLOR)
        self.transformation_matrix = transformation_matrix
        self.threshold_pixel_dist = threshold_pixel_dist
        self.ROI = ROI
        self.frame_h = abs(ROI[2][1] - ROI[1][1])
        self.frame_w = abs(ROI[2][0] - ROI[1][0])
        self.birdeye_background = cv2.warpPerspective(self.background, transformation_matrix, (frame_w, frame_h))

    def create_birdeye_frame(self, warped_center_coords, labels, risk_dict):
        node_radius = int(self.threshold_pixel_dist * 0.5)
        thickness_node = 4

        new_birdeye_frame = self.birdeye_background
        for index, warped_pt in enumerate(warped_center_coords):
            cluster_id = labels[index]
            pt_color = compute_color_for_labels(risk_dict[cluster_id])
            new_birdeye_frame = cv2.circle(
                new_birdeye_frame,
                (int(warped_pt[0]), int(warped_pt[1])),
                node_radius,
                pt_color,
                thickness_node,
            )
            new_birdeye_frame = cv2.circle(
                new_birdeye_frame,
                (int(warped_pt[0]), int(warped_pt[1])),
                1,
                pt_color,
                15,
            )
        return new_birdeye_frame
