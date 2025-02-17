import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import multiview

def integrate_tensor_2d(heatmaps, softmax=True):
    """Applies softmax to heatmaps and integrates them to get their's "center of masses"

    Args:
        heatmaps torch tensor of shape (batch_size, n_heatmaps, h, w): input heatmaps [8,17,96,96]

    Returns:
        coordinates torch tensor of shape (batch_size, n_heatmaps, 2): coordinates of center of masses of all heatmaps

    """
    batch_size, n_heatmaps, h, w = heatmaps.shape

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, -1))
    if softmax:
        heatmaps = nn.functional.softmax(heatmaps, dim=2)
    else:
        heatmaps = nn.functional.relu(heatmaps)

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, h, w))

    mass_x = heatmaps.sum(dim=2)
    mass_y = heatmaps.sum(dim=3)

    mass_times_coord_x = mass_x * torch.arange(w).type(torch.float).to(mass_x.device)
    mass_times_coord_y = mass_y * torch.arange(h).type(torch.float).to(mass_y.device)

    x = mass_times_coord_x.sum(dim=2, keepdim=True)
    y = mass_times_coord_y.sum(dim=2, keepdim=True)

    if not softmax:
        x = x / mass_x.sum(dim=2, keepdim=True)
        y = y / mass_y.sum(dim=2, keepdim=True)

    coordinates = torch.cat((y, x), dim=2)
    coordinates = coordinates.reshape((batch_size, n_heatmaps, 2))

    return coordinates, heatmaps

def unproject_heatmaps(heatmaps, proj_matricies, coord_volumes, volume_aggregation_method='sum', vol_confidences=None):
    device = heatmaps.device
    batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])
    volume_shape = coord_volumes.shape[1:4]

    volume_batch = torch.zeros(batch_size, n_joints, *volume_shape, device=device)

    # TODO: speed up this this loop
    for batch_i in range(batch_size):
        coord_volume = coord_volumes[batch_i]
        grid_coord = coord_volume.reshape((-1, 3))

        volume_batch_to_aggregate = torch.zeros(n_views, n_joints, *volume_shape, device=device)

        for view_i in range(n_views):
            heatmap = heatmaps[batch_i, view_i]
            heatmap = heatmap.unsqueeze(0)

            grid_coord_proj = multiview.project_3d_points_to_image_plane(
                proj_matricies[batch_i, view_i], grid_coord, convert_back_to_euclidean=False
            )

            invalid_mask = grid_coord_proj[:, 2] <= 0.0  # depth must be larger than 0.0

            grid_coord_proj[grid_coord_proj[:, 2] == 0.0, 2] = 1.0  # not to divide by zero
            grid_coord_proj = multiview.homogeneous_to_euclidean(grid_coord_proj)

            # transform to [-1.0, 1.0] range
            grid_coord_proj_transformed = torch.zeros_like(grid_coord_proj)
            grid_coord_proj_transformed[:, 0] = 2 * (grid_coord_proj[:, 0] / heatmap_shape[1] - 0.5)
            grid_coord_proj_transformed[:, 1] = 2 * (grid_coord_proj[:, 1] / heatmap_shape[0] - 0.5)
            grid_coord_proj = grid_coord_proj_transformed

            # prepare to F.grid_sample
            grid_coord_proj = grid_coord_proj.unsqueeze(1).unsqueeze(0)#[1,64*64*64,1,2]
            try:
                current_volume = F.grid_sample(heatmap, grid_coord_proj, align_corners=True, padding_mode='zeros')
            except TypeError: # old PyTorch
                current_volume = F.grid_sample(heatmap, grid_coord_proj)

            # zero out non-valid points
            current_volume = current_volume.view(n_joints, -1)
            current_volume[:, invalid_mask] = 0.0

            # reshape back to volume
            current_volume = current_volume.view(n_joints, *volume_shape)

            # collect
            volume_batch_to_aggregate[view_i] = current_volume

        # agregate resulting volume
        if volume_aggregation_method.startswith('conf'):
            volume_batch[batch_i] = (volume_batch_to_aggregate * vol_confidences[batch_i].view(n_views, n_joints, 1, 1, 1)).sum(0)
        elif volume_aggregation_method == 'sum':
            volume_batch[batch_i] = volume_batch_to_aggregate.sum(0)
        elif volume_aggregation_method == 'max':
            volume_batch[batch_i] = volume_batch_to_aggregate.max(0)[0]
        elif volume_aggregation_method == 'softmax':
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate.clone()
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate_softmin.view(n_views, -1)
            volume_batch_to_aggregate_softmin = nn.functional.softmax(volume_batch_to_aggregate_softmin, dim=0)
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate_softmin.view(n_views, n_joints, *volume_shape)

            volume_batch[batch_i] = (volume_batch_to_aggregate * volume_batch_to_aggregate_softmin).sum(0)
        else:
            raise ValueError("Unknown volume_aggregation_method: {}".format(volume_aggregation_method))

    return volume_batch

def integrate_tensor_3d_with_coordinates(volumes, coord_volumes, softmax=True):
    batch_size, n_volumes, x_size, y_size, z_size = volumes.shape

    volumes = volumes.reshape((batch_size, n_volumes, -1))
    if softmax:
        volumes = nn.functional.softmax(volumes, dim=2)
    else:
        volumes = nn.functional.relu(volumes)

    volumes = volumes.reshape((batch_size, n_volumes, x_size, y_size, z_size))
    coordinates = torch.einsum("bnxyz, bxyzc -> bnc", volumes, coord_volumes)

    return coordinates, volumes