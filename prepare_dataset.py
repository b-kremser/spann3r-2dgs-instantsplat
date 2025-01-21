import os
import torch
import cv2
import argparse
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

# Example dataset
from spann3r.datasets import Scannetpp
from spann3r.loss import get_norm_factor
from dust3r.utils.geometry import inv, geotrf

# Project utilities (adjust import paths if needed)
from utils.sfm_utils import (
    save_intrinsics, 
    save_extrinsic, 
    save_valids,
)

def save_image(path, img):

    # Save images   
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, rgb_image)


def save_depth(path, depthmap):
    np.save(path, depthmap)



def get_args_parser():
    parser = argparse.ArgumentParser('Prepare Dataset', add_help=False)
    parser.add_argument('--root_path', type=str, default="assets_test")
    parser.add_argument('--num_train', type=int, default=3, help='Number of train views per scene')

    
    return parser

def main(args):
    # Example dataset definition
    dataset = Scannetpp(
        split='test',
        ROOT="/mnt/hdd/scannetpp/data",
        resolution=224,
        test_id=['a980334473', 'fb5a96b1a2', 'a24f64f7fb', '25f3b7a318', '3f15a9266d'],
        num_seq=5,
        num_frames=args.num_train,
        full_video=True,
        kf_every=6
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    root = args.root_path  # top-level folder where everything goes

    for i, batch in enumerate(dataloader):
        
        # Split train and test
        train_views = batch[:args.num_train]
        test_views  = batch[args.num_train:]

        scene_label = batch[0]['label'][0]  # e.g. "some_path/.../scene123"
        scene_short = scene_label.split("/")[0]  # last part
        scene_name = f"{scene_short}_{i:02d}"

        # The dataset name might be something from the first element as well
        dataset_name = batch[0]['dataset'][0]

        # Construct the main output folder:
        #   ./assets/dataset_name/scene_name/
        scene_root = os.path.join(root, dataset_name, scene_name)

        # All images go here:
        images_path = os.path.join(scene_root, "images")
        os.makedirs(images_path, exist_ok=True)

        # All images go here:
        depths_path = os.path.join(scene_root, "depths")
        os.makedirs(depths_path, exist_ok=True)

        
        train_path  = os.path.join(scene_root, f"sparse_{args.num_train}", "0")
        test_path  = os.path.join(scene_root, f"sparse_{args.num_train}", "1")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        first_train_pose = train_views[0]['camera_pose']  # (4,4) for the first train view
        T_cam0_world = inv(first_train_pose)  # invert(4x4)

        H, W = 224, 224  # example from your dataset

        test_extrinsics = []
        test_img_files  = []

        all_valids = []
        all_pts = []

        for j, view in enumerate(train_views):

            all_valids.append(view['valid_mask'])
            all_pts.append(geotrf(T_cam0_world, view['pts3d']))

            image_tensor = view['original_img'][0].permute(1, 2, 0).numpy() * 255.0 # (3,H,W)
            image_fname = f"train_{j:03d}.JPG"
            image_path_all = os.path.join(images_path, image_fname)
            save_image(image_path_all, image_tensor)  # implement accordingly

            depth_fname = f"train_{j:03d}_depth.npy"
            depth_path_all = os.path.join(depths_path, depth_fname)
            save_depth(depth_path_all, view['depthmap'][0].numpy())
        
        gt_factor = get_norm_factor(all_pts, "avg_dis", all_valids, False)
        
        all_valids = torch.stack(all_valids, dim=0)
        save_valids(Path(train_path), all_valids.numpy())

        for j, view in enumerate(test_views):
            pose_global = view['camera_pose'][0]  # (4,4)
            pose_rel = T_cam0_world[0] @ pose_global
            pose_rel[:3, 3] /= gt_factor[0].view(1)

            # Save test image to 'images/' as well
            image_tensor = view['original_img'][0].permute(1, 2, 0).numpy() * 255.0 # (3,H,W)
            image_fname  = f"test_{j:03d}.JPG"
            image_path_all   = os.path.join(images_path, image_fname)
            save_image(image_path_all, image_tensor)

            depth_fname = f"test_{j:03d}_depth.npy"
            depth_path_all = os.path.join(depths_path, depth_fname)
            save_depth(depth_path_all, view['depthmap'][0].numpy())

            test_extrinsics.append(pose_rel.numpy())
            test_img_files.append(image_fname)

        # Save extrinsics in the test path
        save_extrinsic(Path(test_path), test_extrinsics, test_img_files, image_suffix=[".JPG", ""])

        # And similarly, store intrinsics
        test_focals = []
        for j, view in enumerate(test_views):
            K = view['camera_intrinsics'][0].numpy()  # e.g. (3,3)
            fx = K[0, 0]
            test_focals.append(fx)

        save_intrinsics(Path(test_path), test_focals, [H, W], [H, W], save_focals=False)

        print(f"[{i}] Saved scene: {scene_name} -> train/test in {scene_root}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
