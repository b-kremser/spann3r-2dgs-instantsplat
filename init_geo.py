import os
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass
from torch.utils.data import DataLoader

# Spann3R + DUSt3R
from spann3r.model import Spann3R
from spann3r.datasets import Demo
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.geometry import inv

# Project utilities
from utils.sfm_utils import (
    save_intrinsics, 
    save_extrinsic, 
    save_points3D, 
    save_time,
    save_images, 
    init_filestructure, 
)
from utils.camera_utils import generate_interpolated_path

@dataclass
class ReconstructionConfig:
    source_path: str
    model_path: str
    ckpt_path: str
    device: str
    image_size: int
    n_views: int
    infer_video: bool = False


def process_batch(batch, device):
    """Move batch data to device."""
    for view in batch:
        for name in ['img', 'original_img', 'pts3d', 'valid_mask', 'camera_pose',
                     'camera_intrinsics', 'F_matrix', 'corres', 'depthmap']:
            if name in view:
                view[name] = view[name].to(device, non_blocking=True)
    return batch


class Reconstructor:
    def __init__(self, config: ReconstructionConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._setup_paths()
        self._load_model()

    def _setup_paths(self) -> None:
        """Initialize directory structure and paths."""
        source_path = Path(self.config.source_path)
        self.save_path, self.sparse_0_path, self.sparse_1_path = init_filestructure(
            source_path, self.config.n_views
        )

    def _load_model(self) -> None:
        """Load and initialize Spann3R model."""
        try:
            self.model = Spann3R(
                dus3r_name='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
                use_feat=False
            ).to(self.device)
            
            checkpoint = torch.load(self.config.ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _load_dataset(self) -> Tuple[List[dict], List[Path], Optional[List[Path]]]:
        """Load and prepare dataset."""
        dataset = Demo(
            ROOT=os.path.join(self.config.source_path, 'images'),
            resolution=self.config.image_size,
            full_video=True,
            kf_every=1
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        batch = process_batch(next(iter(dataloader)), self.config.device)
        
        if self.config.infer_video:
            return batch, [Path(view['file_name'][0]) for view in batch], None
                    
        train_batch = [b for b in batch if b['file_name'][0].startswith("train")]
        test_batch = [b for b in batch if b['file_name'][0].startswith("test")]
        
        return (
            train_batch,
            [Path(train_batch[i]['file_name'][0]) for i in range(len(train_batch))],
            [Path(test_batch[i]['file_name'][0]) for i in range(len(test_batch))]
        )

    def _estimate_focal(
        self, 
        pts3d: torch.Tensor,      # shape (B, H, W, 3)
        conf3d: Optional[torch.Tensor] = None  # shape (B, H, W)
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate focal length and create camera matrix, optionally using
        a confidence map to filter out low-confidence 3D points.
        """
        B, predH, predW, _ = pts3d.shape

        principal_point = torch.tensor((predW / 2.0, predH / 2.0), device=pts3d.device)

        # Here we pass pts3d and conf3d into the focal estimator
        # so that it can handle the high-confidence filtering.
        focal_est = estimate_focal_knowing_depth(
            pts3d,
            conf3d,  # can be None if you don't want filtering
            principal_point,
            focal_mode='weiszfeld'  # or 'median', etc.
        )

        if B == 1:
            focal_val = focal_est.item()
        else:
            focal_val = focal_est.mean().item()

        K = np.array([
            [focal_val, 0.0,        predW / 2.0],
            [0.0,       focal_val,  predH / 2.0],
            [0.0,       0.0,        1.0]
        ], dtype=np.float32)

        return focal_val, K

    def _solve_pnp(
        self,
        pts3d: np.ndarray,                # shape: (H, W, 3)
        K: np.ndarray,                    # shape: (3, 3), the camera intrinsics
        conf: np.ndarray = None,          # shape: (H, W), optional confidence map
        conf_threshold: float = 0.1,      # if percentile_mode=True, this is the quantile
        percentile_mode: bool = True
    ) -> np.ndarray:
        """
        Solve PnP to get camera extrinsics, optionally using a confidence map 
        to filter out low-confidence correspondences.
        
        Returns:
            extrinsic_matrix: np.ndarray of shape (4, 4), the pose in world->cam 
                            (or cam->world) form, depending on your convention.
        """

        # -- 1) Build the 2D grid (u,v) covering [0...W-1, 0...H-1]
        H, W, _ = pts3d.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))  # shape: (H, W) each
        uv_points_2d = np.stack((u, v), axis=-1)        # shape: (H, W, 2)

        # -- 2) Flatten everything to shape (N, ...)
        pts3d_flat = pts3d.reshape(-1, 3).astype(np.float32)       # (N, 3)
        uv_points_2d_flat = uv_points_2d.reshape(-1, 2).astype(np.float32)  # (N, 2)

        # -- 3) If we have confidence, filter by threshold
        if conf is not None:
            conf_flat = conf.reshape(-1)  # shape: (N,)

            if percentile_mode:
                thr_value = np.quantile(conf_flat, conf_threshold)
            else:
                thr_value = conf_threshold

            mask = (conf_flat >= thr_value)
            
            # Fallback if too few points remain
            if np.count_nonzero(mask) < 10:
                # e.g., keep all points if there's not enough high-confidence ones
                mask = np.ones_like(conf_flat, dtype=bool)

            # Filter out low-confidence points
            pts3d_flat = pts3d_flat[mask]
            uv_points_2d_flat = uv_points_2d_flat[mask]

        # -- 4) Call solvePnPRansac on the (filtered) correspondences
        dist_coeffs = np.zeros(4, dtype=np.float32)
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            pts3d_flat,
            uv_points_2d_flat,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            # Could not solve PnP; return identity
            return np.eye(4, dtype=np.float32)

        # -- 5) Convert rvec/tvec to a 4x4 extrinsic matrix
        R, _ = cv2.Rodrigues(rvec)
        extrinsic_matrix = np.vstack((
            np.hstack((R, tvec.reshape(3, 1))),
            [0, 0, 0, 1]
        ))

        # By default, solvePnP finds camera->world or world->camera
        # depending on your usage. If you need the inverse, you can invert it:
        return np.linalg.inv(extrinsic_matrix)

    def process(self) -> None:
        """Main processing pipeline."""
        # Load and prepare data
        train_batch, train_img_files, test_img_files = self._load_dataset()
        image_suffix = train_batch[0]['suffix']
        original_size = train_batch[0]['original_size']
        original_size = (original_size[1].item(), original_size[0].item())

        # Forward pass
        start_time = time()
        with torch.no_grad():
            preds, _= self.model(train_batch)

        tmp_pts3d = []
        all_conf = []

        for j, view in enumerate(train_batch):
            pts3d_pred = (preds[j]['pts3d'] if j == 0 
                        else preds[j]['pts3d_in_other_view'])
            pts3d_np = pts3d_pred.detach().cpu().numpy()[0]
            tmp_pts3d.append(pts3d_np.reshape(-1, 3))

            # Extract the confidence
            conf_np = preds[j]['conf'].detach().cpu().numpy()[0]
            all_conf.append(conf_np.reshape(-1))

        all_pts_concat = np.concatenate(tmp_pts3d, axis=0)  
        all_conf_concat = np.concatenate(all_conf, axis=0)  

        if self.config.infer_video:
            conf_threshold = np.quantile(all_conf_concat, 0.1)     
            valid_mask = all_conf_concat >= conf_threshold
        else:
            valid_mask = np.load(str(self.sparse_0_path / "valids.npy")).reshape(-1)

        scale_valid_pts = all_pts_concat.copy()
        scale_valid_pts[~valid_mask] = 0.0   

        scale = np.mean(np.linalg.norm(scale_valid_pts, axis=-1))  

        for j, view in enumerate(train_batch):
            # Scale the "main" pts3d if j==0, otherwise scale pts3d_in_other_view
            if j == 0:
                preds[j]['pts3d'] = (preds[j]['pts3d'] / scale) 
            else:
                preds[j]['pts3d_in_other_view'] = (preds[j]['pts3d_in_other_view'] / scale)

        focal_est, K = self._estimate_focal(preds[0]['pts3d'], preds[0]['conf'])

        all_pts3d, all_conf, all_extrinsics_w2c = [], [], []
        all_colors, all_imgs = [], []
        _, predH, predW, _ = preds[0]['pts3d'].shape 

        for j, view in enumerate(train_batch):
            # Extract scaled predictions
            pts3d_pred = (preds[j]['pts3d'] if j == 0 
                        else preds[j]['pts3d_in_other_view'])
            pts3d_pred = pts3d_pred.detach().cpu().numpy()[0]  
            conf_pred = preds[j]['conf'][0].detach().cpu().numpy()
            extrinsic_w2c = self._solve_pnp(pts3d_pred, K, conf_pred)

            # Store processed data
            img = view['original_img'][0].mul(255).cpu().numpy().transpose(1, 2, 0)
            all_pts3d.append(pts3d_pred.reshape(-1, 3))
            all_conf.append(conf_pred.reshape(-1))
            all_extrinsics_w2c.append(extrinsic_w2c)
            all_imgs.append(img)
            all_colors.append(img.reshape(-1, 3))

        # Stack and concatenate results
        all_extrinsics_w2c = np.stack(all_extrinsics_w2c)
        all_imgs = np.array(all_imgs)
        pts3d_concat = np.concatenate(all_pts3d) 
        conf_concat = np.concatenate(all_conf)
        colors_concat = np.concatenate(all_colors)

        # Save results
        # if not self.config.infer_video and test_img_files:
        #     self._save_test_data(
        #         all_extrinsics_w2c,
        #         test_img_files,
        #         focal_est,
        #         predH,
        #         predW,
        #         image_suffix
        #     )

        # Save training data
        train_focals = np.full(len(train_img_files), focal_est)
        save_intrinsics(
            self.sparse_0_path,
            train_focals,
            original_size,
            [predH, predW],
            save_focals=True
        )
        save_extrinsic(
            self.sparse_0_path,
            all_extrinsics_w2c,
            train_img_files,
            image_suffix
        )
        
        pts_num = save_points3D(
            self.sparse_0_path,
            colors_concat,
            pts3d_concat,
            conf_concat.reshape(-1, 1),
            masks=None,
            use_masks=False,
            save_all_pts=True,
            save_txt_path=self.config.model_path
        )
        
        save_images(
            self.sparse_0_path,
            len(train_img_files),
            all_imgs,
            train_img_files,
            image_suffix
        )

        # Save timing information
        total_time = time() - start_time
        save_time(self.config.model_path, '[1] coarse_init_TrainTime', total_time)
        save_time(self.config.model_path, '[1] init_geo', total_time)

        print(f"[INFO] Reconstruction saved to: {self.sparse_0_path}")
        print(f"[INFO] Processed points: {pts_num}")
        print(f"[INFO] Processing time: {total_time:.2f}s")

    def _save_test_data(
        self,
        all_extrinsics_w2c: np.ndarray,
        test_img_files: List[Path],
        focal_est: float,
        predH: int,
        predW: int,
        image_suffix: str
    ) -> None:
        """Save test data with pose interpolation if needed."""
        n_train = len(all_extrinsics_w2c)
        n_test = len(test_img_files)

        if n_train < n_test:
            # Interpolate poses
            n_interp = (n_test // (n_train - 1)) + 1
            all_inter_pose = []
            
            for i in range(n_train - 1):
                tmp_inter_pose = generate_interpolated_path(
                    poses=all_extrinsics_w2c[i:i+2],
                    n_interp=n_interp
                )
                all_inter_pose.append(tmp_inter_pose)
            
            all_inter_pose = np.concatenate([
                np.concatenate(all_inter_pose),
                all_extrinsics_w2c[-1][:3, :].reshape(1, 3, 4)
            ])
            
            indices = np.linspace(0, len(all_inter_pose) - 1, n_test, dtype=int)
            pose_test = np.array([
                np.vstack((pose, [0, 0, 0, 1]))
                for pose in all_inter_pose[indices]
            ])
        else:
            indices = np.linspace(0, n_train - 1, n_test, dtype=int)
            pose_test = all_extrinsics_w2c[indices]

        # Save test data
        save_extrinsic(self.sparse_1_path, pose_test, test_img_files, image_suffix)
        test_focals = np.full(n_test, focal_est)
        save_intrinsics(
            self.sparse_1_path,
            test_focals,
            [predH, predW],
            [predH, predW],
            save_focals=False
        )

def main():
    parser = argparse.ArgumentParser(description='Optimized 3D reconstruction using Spann3R')
    parser.add_argument('--source_path', '-s', required=True, help='Image directory path')
    parser.add_argument('--model_path', '-m', required=True, help='Results directory path')
    parser.add_argument('--ckpt_path', default='./checkpoints/spann3r.pth', help='Checkpoint path')
    parser.add_argument('--device', default='cuda', help='Device for inference')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--n_views', type=int, default=3, help='Number of training views')
    parser.add_argument('--infer_video', action="store_true", help='Process as video sequence')

    args = parser.parse_args()
    config = ReconstructionConfig(**vars(args))
    
    try:
        reconstructor = Reconstructor(config)
        reconstructor.process()
    except Exception as e:
        print(f"[ERROR] Reconstruction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()