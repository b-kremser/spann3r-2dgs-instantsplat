# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities for interpreting the DUST3R output
# --------------------------------------------------------
import numpy as np
import torch
from typing import Optional
from dust3r.utils.geometry import xy_grid


def estimate_focal_knowing_depth(
    pts3d: torch.Tensor,             # shape (B, H, W, 3)
    conf3d: Optional[torch.Tensor],  # shape (B, H, W) or None
    pp: torch.Tensor,                # principal_point, shape (2,) on same device
    focal_mode: str = 'median',
    min_focal: float = 0.0,
    max_focal: float = np.inf
) -> torch.Tensor:
    """
    Reprojection method for when absolute depth is known. Now adapted to 
    filter out low-confidence points (below 90th percentile) if conf3d is given.
    
    Returns:
        focal: shape (B,) if B>1, otherwise scalar if B=1
    """

    B, H, W, THREE = pts3d.shape
    assert THREE == 3, "pts3d must be shape (B, H, W, 3)"

    # Build a centered pixel grid of shape (B, HW, 2)
    # xy_grid(...) should produce shape (H, W, 2); we broadcast for batch
    pixel_grid = xy_grid(W, H, device=pts3d.device)      # shape (H, W, 2)
    pixel_grid = pixel_grid.view(1, -1, 2).expand(B, -1, -1)
    # Subtract principal point from each pixel ( broadcast over B, HW )
    pixels = pixel_grid - pp.view(1, 1, 2)  # shape (B, HW, 2)

    # Flatten pts3d from (B, H, W, 3) -> (B, HW, 3)
    pts3d = pts3d.view(B, -1, 3)

    if conf3d is not None:
        # Flatten conf3d to (B, HW)
        conf3d = conf3d.view(B, -1)  # shape (B, HW)

        conf_1d = conf3d.view(-1) 
        threshold = torch.quantile(conf_1d, 0.8)  

        # Build the mask of valid points
        mask = (conf3d >= threshold)  # shape (B, HW), True/False


        pts3d_flat   = pts3d.view(-1, 3)      # shape (B*HW, 3)
        pixels_flat  = pixels.view(-1, 2)     # shape (B*HW, 2)
        mask_flat    = mask.view(-1)          # shape (B*HW,)

        # (ii) Keep only the high-confidence points
        pts3d_highconf  = pts3d_flat[mask_flat]
        pixels_highconf = pixels_flat[mask_flat]

        if pts3d_highconf.shape[0] < 10:
            # Fallback to using all points or some default
            pts3d_highconf  = pts3d_flat
            pixels_highconf = pixels_flat

        x = pts3d_highconf[:, 0]
        y = pts3d_highconf[:, 1]
        z = pts3d_highconf[:, 2]
        u = pixels_highconf[:, 0]
        v = pixels_highconf[:, 1]

        # Depending on focal_mode, we handle them differently:
        if focal_mode == 'median':
            # direct estimation of focal
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y
            f_votes  = torch.cat([fx_votes, fy_votes], dim=0)
            focal    = torch.nanmedian(f_votes).unsqueeze(0)  # shape (1,)

        elif focal_mode == 'weiszfeld':
            # quick closed-form init
            xy_over_z = (pts3d_highconf[:, :2] / (z.unsqueeze(-1))).nan_to_num(0)
            dot_xy_px = (xy_over_z * pixels_highconf).sum(dim=-1)  # shape (M,)
            dot_xy_xy = xy_over_z.pow(2).sum(dim=-1)               # shape (M,)

            # initial guess
            focal_init = dot_xy_px.mean() / dot_xy_xy.mean()
            focal = focal_init.unsqueeze(0)  # shape (1,)

            # iterative re-weighting
            for _ in range(10):
                # distance between (u,v) and focal*(x/z, y/z)
                dis = (pixels_highconf - focal * xy_over_z).norm(dim=-1)  # shape (M,)
                w = dis.clamp_min(1e-8).reciprocal()
                focal = (w * dot_xy_px).sum() / (w * dot_xy_xy).sum()
                focal = focal.unsqueeze(0)  # keep shape (1,)

        else:
            raise ValueError(f'Unknown focal_mode={focal_mode}')

        # shape(1,) -> clamp and return shape(1,) or shape(B,) if desired
        # for simplicity, we'll do the same clamp logic
        focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # ~ size / 1.1547
        focal = focal.clamp(min=min_focal*focal_base, max=max_focal*focal_base)
        return focal  # shape (1,) if B=1, or your chosen aggregator

    u, v = pixels.unbind(dim=-1)  # shape (B, HW), (B, HW)
    x, y, z = pts3d.unbind(dim=-1)

    if focal_mode == 'median':
        fx_votes = (u * z) / x
        fy_votes = (v * z) / y
        f_votes  = torch.cat((fx_votes, fy_votes), dim=-1)  # shape (B, 2*HW)
        focal    = torch.nanmedian(f_votes, dim=-1).values  # shape (B,)

    elif focal_mode == 'weiszfeld':
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(0)  # shape (B, HW, 2)
        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)   # shape (B, HW)
        dot_xy_xy = xy_over_z.pow(2).sum(dim=-1)       # shape (B, HW)

        # init
        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)  # shape (B,)

        for _ in range(10):
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)  # shape (B, HW)
            w = dis.clamp_min(1e-8).reciprocal()  # shape (B, HW)
            focal = (w * dot_xy_px).sum(dim=1) / (w * dot_xy_xy).sum(dim=1)

    else:
        raise ValueError(f'Unknown focal_mode={focal_mode}')

    # Clip the focal range
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547
    focal = focal.clamp(min=min_focal*focal_base, max=max_focal*focal_base)

    return focal  # shape (B,)
