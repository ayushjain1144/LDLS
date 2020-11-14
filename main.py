
import numpy as np
from pathlib import Path
import skimage

from lidar_segmentation.detections import MaskRCNNDetections
from lidar_segmentation.segmentation import LidarSegmentation
from lidar_segmentation.kitti_utils import load_kitti_lidar_data, load_kitti_object_calib
import lidar_segmentation.kitti_utils_custom
from lidar_segmentation.utils import load_image as my_utils
# from mask_rcnn.mask_rcnn import MaskRCNNDetector

calib_folder = Path("data/") / "kitti_demo" / "calib" 
image_folder = Path("data/") / "kitti_demo" / "image_2" 
lidar_folder = Path("data/") / "kitti_demo" / "velodyne"

calib_path = Path("testing/") / "calib" / "0.txt"
image_path = Path("testing/") / "image_2" / "0.png"
lidar_path = Path("testing/") / "velodyne" / "0.npy"
label_path = Path("testing/") / "label_2" / "0.txt"
object
# Load calibration data
projection = load_kitti_object_calib(calib_path)

# Load image
image = load_image(image_path)

skimage.io.imshow(image)

# Load lidar
lidar = load_kitti_lidar_data(lidar_path, load_reflectance=False)
print("Loaded LiDAR point cloud with %d points" % lidar.shape[0])

lidarseg = LidarSegmentation(projection)

# Load labels from txt instead of running their maskrcnn
label_file = open(label_path, 'r')
data_list = label_file.split(' ')
class_label = 






