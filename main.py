import numpy as np
from pathlib import Path
import skimage
import torch
from lidar_segmentation.detections import MaskRCNNDetections
from lidar_segmentation.segmentation import LidarSegmentation
from lidar_segmentation.kitti_utils import load_kitti_lidar_data, load_kitti_object_calib
# import lidar_segmentation.kitti_utils_custom
from lidar_segmentation.utils import load_image
from mask_rcnn.mask_rcnn import MaskRCNNDetector
import lidar_segmentation.utils_geom as utils_geom
import lidar_segmentation.utils_eval as utils_eval
import lidar_segmentation.utils_misc as utils_misc
import lidar_segmentation.utils_box as utils_box
import lidar_segmentation.utils_basic as utils_basic
import lidar_segmentation.utils_vox as utils_vox
import lidar_segmentation.utils_ap as utils_ap

# from lidar_segmentation.evaluation import evaluate_instance_segmentation, LidarSegmentationGroundTruth

import ipdb
import os
st = ipdb.set_trace




agg_mAP_ldls = 0.0
agg_mAP_pseudo = 0.0
total_imgs = 0

eval_ldls = True
eval_pseudo = False

if eval_ldls:
    image_folder = Path("testing/") / "image_2"
if eval_pseudo:
    image_folder =  Path("replica_pseudo_testing/") / "image_2"


for idx in range(len(os.listdir(image_folder))):

    calib_path = Path("testing/") / "calib" / f"{idx}.txt"
    image_path = Path("testing/") / "image_2" / f"{idx}.png"
    lidar_path = Path("testing/") / "velodyne" / f"{idx}.npy"
    label_path = Path("testing/") / "label_2" / f"{idx}.txt"
    pseudo_path = Path("pseudo") / "label_2" / f"{idx}.txt"

    print(f"processing images at path: {image_path}")

    # Load calibration data
    projection = load_kitti_object_calib(calib_path)

    # Load image
    image = load_image(image_path)

    # skimage.io.imshow(image)

    # Load lidar
    lidar = load_kitti_lidar_data(lidar_path, load_reflectance=False)
    print("Loaded LiDAR point cloud with %d points" % lidar.shape[0])

    lidarseg = LidarSegmentation(projection)

    # Load labels from txt instead of running their maskrcnn
    label_file = open(label_path, 'r')

    bbox_2d_list = []
    class_list = []
    box3d_list = []
    

    # loading ground truth
    for line in label_file.readlines():
        data_list = line.split(' ')
        class_label = data_list[0]
        # st()
        bbox_2d = [float(s) for s in data_list[4:8]]
        bbox_3d = np.array([float(s) for s in data_list[8:]]).reshape(8, 3)
        bbox_2d_list.append(bbox_2d)
        class_list.append(class_label)
        box3d_list.append(bbox_3d)

    # loading pseudo 3D just for calculating mAP for free
    box3d_list_pseudo = []
    label_file = open(pseudo_path, 'r')

    for line in label_file.readlines():
        data_list = line.split(' ')
        bbox_3d = np.array([float(s) for s in data_list[8:]]).reshape(8, 3)
        box3d_list_pseudo.append(bbox_3d)

    # create voxel scene 

    scene_centroid_x = 0.0
    scene_centroid_y = 1.0 #hyp.YMIN
    scene_centroid_z = 18.0
    # for cater the table is y=0, so we move upward a bit
    scene_centroid = np.array([scene_centroid_x,
                                scene_centroid_y,
                                scene_centroid_z]).reshape([1, 3])
    scene_centroid = torch.from_numpy(scene_centroid).float().cuda() 
    Z, Y, X  = 80, 80, 80             
                                                
    vox_util = utils_vox.Vox_util(
        Z, Y, X, 
        'train', scene_centroid=scene_centroid,
        assert_cube=True)
    # st()

    # mAP for ldls

    if eval_ldls:

        # detector
        detector = MaskRCNNDetector()
        detections = detector.detect(image)

        # results
        results = lidarseg.run(lidar, detections, max_iters=50, save_all=False)

        # st()
        #st()
        xyz_pc = results.points
        binary_pc = np.isclose(results.label_likelihoods[1], 0)[:, 1]
        #st()
        xyz_pc_obj = xyz_pc[~binary_pc]
        num_objs = results.class_ids.shape[0]
        scores = np.ones((num_objs, 1))
        mask_grid_s = vox_util.voxelize_xyz(torch.from_numpy(xyz_pc_obj).unsqueeze(0).cuda(), Z, Y, X, assert_cube=False)
        _, box3dlist, _, _, _ = utils_misc.get_boxes_from_flow_mag(mask_grid_s.squeeze(0), num_objs)
        pred_lrtlist = utils_geom.convert_boxlist_to_lrtlist(box3dlist)
        pred_lrtlist = vox_util.apply_ref_T_mem_to_lrtlist(pred_lrtlist, Z, Y, X)
        pred_xyzlist = utils_geom.get_xyzlist_from_lrtlist(pred_lrtlist)
        map3d,_ = utils_eval.get_mAP_from_xyzlist_py(pred_xyzlist.cpu().numpy(), scores, np.expand_dims(np.stack(box3d_list, axis=0), axis=0), iou_threshold=0.25)
        print(map3d)
        agg_mAP_ldls += map3d

    # map for pseudo
    if eval_pseudo:
        if not box3d_list_pseudo:
            continue
        scores = np.ones(len(box3d_list_pseudo))
        # st()
        map3d,_ = utils_eval.get_mAP_from_xyzlist_py(np.expand_dims(np.stack(box3d_list_pseudo, axis=0), axis=0), scores, np.expand_dims(np.stack(box3d_list, axis=0), axis=-0), iou_threshold=0.25)
        agg_mAP_pseudo += map3d


    total_imgs += 1


if eval_ldls:
    print("mAP results for LDLS:  ............................")
    print(f"mAP: {agg_mAP_ldls/total_imgs}")
    print(f"Total imgs: {total_imgs}")

if eval_pseudo:
    print("mAP results for Pseudo:  ............................")
    print(f"mAP: {agg_mAP_pseudo/total_imgs}")
    print(f"Total imgs: {total_imgs}")









