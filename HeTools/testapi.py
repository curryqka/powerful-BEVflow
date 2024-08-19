from haplugin.utils.box_utils import boxes_to_corners_3d, corners_3d_to_boxes3d
import numpy as np
box = np.array([1,2,3,4,5,6,0]).reshape(1,7)
print(box.shape)
corner = boxes_to_corners_3d(box)
print(corner.shape)