from .scn import Loc_SCN
from .v2v import V2VModel
from .v2v_res import V2VResModel
from .pose_resnet import get_pose_net
from .adafuse_network import AdafuseNet
from utils import op,multiview,volumetric

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class AlgAdafuseNet(nn.Module):
    def __init__(self, config,model_path,device):
        super().__init__()
        self.adafuse = AdafuseNet(config,device)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.adafuse.load_state_dict(checkpoint['state_dict'],strict=False)
    
    def forward(self, AP_img, AP_K, AP_T, LAT_img, LAT_K, LAT_T):
        """
        AP_img, LAT_img: [bs,768,512]
        AP_proj_matricies, LAT_projmatricies: [bs,3,4]
        """
        device = AP_img.device
        batch_size = AP_img.shape[0]

        keypoints_2d_pred, hmaps_pred = self.adafuse(AP_img, AP_K, AP_T, LAT_img, LAT_K, LAT_T)
        keypoints_2d_pred = keypoints_2d_pred.transpose(3,2)
        alg_confidences = torch.ones(batch_size, 2, keypoints_2d_pred.shape[2]).type(torch.float).to(device)
        
        AP_proj_matricies = torch.bmm(AP_K, AP_T)
        LAT_proj_matricies = torch.bmm(LAT_K, LAT_T)
        proj_matricies = torch.stack([AP_proj_matricies,LAT_proj_matricies],dim=1)

        # norm confidences
        alg_confidences = alg_confidences / alg_confidences.sum(dim=1, keepdim=True)
        alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # triangulate
        keypoints_3d = multiview.triangulate_batch_of_points(
            proj_matricies, keypoints_2d_pred,
            confidences_batch=alg_confidences
        )
        return keypoints_3d
    
class AlgResNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_confidences = config.use_confidences

        self.backbone = get_pose_net(config)
    
    def forward(self, AP_img, AP_K, AP_T, LAT_img, LAT_K, LAT_T):
        """
        AP_img, LAT_img: [bs,768,512]
        AP_proj_matricies, LAT_projmatricies: [bs,3,4]
        """
        device = AP_img.device
        batch_size = AP_img.shape[0]

        if self.use_confidences:
            AP_hmap, _, AP_alg_confidences, _ = self.backbone(AP_img.repeat(1,3,1,1))
            LAT_hmap, _, LAT_alg_confidences, _ = self.backbone(LAT_img.repeat(1,3,1,1))
            alg_confidences = torch.stack([AP_alg_confidences,LAT_alg_confidences],dim=1)
        else:
            AP_hmap, _, _, _ = self.backbone(AP_img.repeat(1,3,1,1))
            LAT_hmap, _, _, _ = self.backbone(LAT_img.repeat(1,3,1,1))
            alg_confidences = torch.ones(batch_size, 2, AP_hmap.shape[1]).type(torch.float).to(device)

        # test 2d resnet
        AP_hmap = F.interpolate(AP_hmap, size = (AP_img.shape[2],AP_img.shape[3]), mode="bilinear")
        LAT_hmap = F.interpolate(LAT_hmap, size = (AP_img.shape[2],AP_img.shape[3]), mode="bilinear")
        AP_keypoints_2d, _ = op.integrate_tensor_2d(torch.relu(AP_hmap)*50,True)
        LAT_keypoints_2d, _ = op.integrate_tensor_2d(torch.relu(LAT_hmap)*50,True)

        AP_keypoints_2d = torch.flip(AP_keypoints_2d, [2])
        LAT_keypoints_2d = torch.flip(LAT_keypoints_2d, [2])
        # stack
        hmaps = torch.stack([AP_hmap,LAT_hmap],dim=1)
        keypoints_2d = torch.stack([AP_keypoints_2d,LAT_keypoints_2d],dim=1)
        
        # image_shape, heatmap_shape = tuple([768,512]), tuple([192,128])
        # AP_K = multiview.update_after_resize(AP_K, image_shape, heatmap_shape)
        # LAT_K = multiview.update_after_resize(LAT_K, image_shape, heatmap_shape)
        AP_proj_matricies = torch.bmm(AP_K, AP_T)
        LAT_proj_matricies = torch.bmm(LAT_K, LAT_T)
        proj_matricies = torch.stack([AP_proj_matricies,LAT_proj_matricies],dim=1)

        # norm confidences
        alg_confidences = alg_confidences / alg_confidences.sum(dim=1, keepdim=True)
        alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # upscale keypoints_2d, because image shape != heatmap shape
        # keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        # keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (768 / 192)
        # keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (512 / 128)
        # keypoints_2d = keypoints_2d_transformed

        # triangulate
        keypoints_3d = multiview.triangulate_batch_of_points(
            proj_matricies, keypoints_2d,
            confidences_batch=alg_confidences
        )
        return keypoints_3d

class AlgebraicTriangulationNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_confidences = config.use_confidences

        self.backbone = Loc_SCN(config.num_classes,1,alg_confidences=self.use_confidences,vol_confidences=False)
        checkpoint = torch.load(config.weightFile, map_location='cpu')
        # model_dict = self.AP_backbone.state_dict()
        # state_dict = {k:v for k,v in checkpoint['state_dict'].items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        self.backbone.load_state_dict(checkpoint['state_dict'],strict=False)
    
    def forward(self, AP_img, AP_K, AP_T, LAT_img, LAT_K, LAT_T):
        """
        AP_img, LAT_img: [bs,768,512]
        AP_proj_matricies, LAT_projmatricies: [bs,3,4]
        """
        device = AP_img.device
        batch_size = AP_img.shape[0]

        if self.use_confidences:
            AP_hmap, _, AP_alg_confidences, _ = self.backbone(AP_img)
            LAT_hmap, _, LAT_alg_confidences, _ = self.backbone(LAT_img)
            alg_confidences = torch.stack([AP_alg_confidences,LAT_alg_confidences],dim=1)
        else:
            AP_hmap, _, _, _ = self.backbone(AP_img)
            LAT_hmap, _, _, _ = self.backbone(LAT_img)
            alg_confidences = torch.ones(batch_size, 2, AP_hmap.shape[1]).type(torch.float).to(device)

        AP_keypoints_2d, _ = op.integrate_tensor_2d(torch.relu(AP_hmap)*50,True)
        LAT_keypoints_2d, _ = op.integrate_tensor_2d(torch.relu(LAT_hmap)*50,True)

        AP_keypoints_2d = torch.flip(AP_keypoints_2d, [2])
        LAT_keypoints_2d = torch.flip(LAT_keypoints_2d, [2])
        # stack
        hmaps = torch.stack([AP_hmap,LAT_hmap],dim=1)
        keypoints_2d = torch.stack([AP_keypoints_2d,LAT_keypoints_2d],dim=1)
        # proj_matricies = torch.stack([AP_proj_matricies,LAT_proj_matricies],dim=1)

        # norm confidences
        alg_confidences = alg_confidences / alg_confidences.sum(dim=1, keepdim=True)
        alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # image_shape, heatmap_shape = tuple(AP_img.shape[2:]), tuple(AP_feat.shape[2:])
        # AP_K = multiview.update_after_resize(AP_K, image_shape, heatmap_shape)
        # LAT_K = multiview.update_after_resize(LAT_K, image_shape, heatmap_shape)
        AP_proj_matricies = torch.bmm(AP_K, AP_T)
        LAT_proj_matricies = torch.bmm(LAT_K, LAT_T)
        proj_matricies = torch.stack([AP_proj_matricies,LAT_proj_matricies],dim=1)


        # triangulate
        keypoints_3d = multiview.triangulate_batch_of_points(
            proj_matricies, keypoints_2d,
            confidences_batch=alg_confidences
        )
        return keypoints_3d


class VolResNet(nn.Module):
    def __init__(self, config,device):
        super().__init__()

        self.volume_aggregation_method = config.volume_aggregation_method

        # volume
        self.volume_multiplier = 50
        self.volume_size = 64
        self.cuboid_side = 250

        self.use_confidences = config.use_confidences

        self.backbone = get_pose_net(config,device)
        for p in self.backbone.final_layer.parameters():
            p.requires_grad = False
        if config.backbone_weightFile != "none":
            checkpoint = torch.load(config.backbone_weightFile, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['state_dict'],strict=False)
        
        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.volume_net = V2VResModel(32,config.num_classes)

    def forward(self, AP_img, AP_K, AP_T, LAT_img, LAT_K, LAT_T):
        device = AP_img.device
        batch_size = AP_img.shape[0]

        if self.use_confidences:
            AP_hmap, AP_feat, _, AP_vol_confidences = self.backbone(AP_img.repeat(1,3,1,1))
            LAT_hmap, LAT_feat, _, LAT_vol_confidences = self.backbone(LAT_img.repeat(1,3,1,1))
            vol_confidences = torch.stack([AP_vol_confidences,LAT_vol_confidences],dim=1)
        else:
            AP_hmap, AP_feat, _, _ = self.backbone(AP_img)
            LAT_hmap, LAT_feat, _, _ = self.backbone(LAT_img)
            vol_confidences = torch.ones(batch_size, 2, AP_hmap.shape[1]).type(torch.float).to(device)

        if self.use_confidences:
            vol_confidences = vol_confidences + torch.tensor([1e-32],device=device)
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # resize K
        image_shape, heatmap_shape = tuple([768,512]), tuple([192,128])
        AP_K = multiview.update_after_resize(AP_K, image_shape, heatmap_shape)
        LAT_K = multiview.update_after_resize(LAT_K, image_shape, heatmap_shape)

        AP_proj_matricies = torch.bmm(AP_K, AP_T)
        LAT_proj_matricies = torch.bmm(LAT_K, LAT_T)
        proj_matricies = torch.stack([AP_proj_matricies,LAT_proj_matricies],dim=1)

        features = torch.stack([AP_feat,LAT_feat],dim=1)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            center_2d_coord = torch.tensor([i / 2 for i in heatmap_shape],device=device)
            base_point = multiview.triangulate_point_from_multiple_views_linear_torch(proj_matricies[batch_i],center_2d_coord.unsqueeze(0).repeat(2,1))
            base_points[batch_i] = base_point

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point.detach().cpu().numpy() - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)
            
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            center = base_point

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, [0,0,1])
            coord_volume = coord_volume + center
            
            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, 2, *features.shape[1:])

        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=True)
        return vol_keypoints_3d, volumes, coord_volumes, AP_hmap, LAT_hmap
class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.volume_aggregation_method = config.volume_aggregation_method

        # volume
        self.volume_multiplier = 50
        self.volume_size = 64
        self.cuboid_side = 250

        self.use_confidences = config.use_confidences

        self.backbone = Loc_SCN(config.num_classes,1,alg_confidences=False,vol_confidences=self.use_confidences)

        if config.backbone_weightFile != "none":
            checkpoint = torch.load(config.backbone_weightFile, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['state_dict'],strict=False)

        self.volume_net = V2VModel(config.num_classes,16)

    def forward(self, AP_img, AP_K, AP_T, LAT_img, LAT_K, LAT_T):
        device = AP_img.device
        batch_size = AP_img.shape[0]

        if self.use_confidences:
            AP_hmap, AP_feat, _, AP_vol_confidences = self.backbone(AP_img)
            LAT_hmap, LAT_feat, _, LAT_vol_confidences = self.backbone(LAT_img)
            vol_confidences = torch.stack([AP_vol_confidences,LAT_vol_confidences],dim=1)
        else:
            AP_hmap, AP_feat, _, _ = self.backbone(AP_img)
            LAT_hmap, LAT_feat, _, _ = self.backbone(LAT_img)
            vol_confidences = torch.ones(batch_size, 2, AP_hmap.shape[1]).type(torch.float).to(device)

        if self.use_confidences:
            vol_confidences = vol_confidences + torch.tensor([1e-32],device=device)
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # resize K
        image_shape, heatmap_shape = tuple(AP_img.shape[2:]), tuple(AP_feat.shape[2:])
        AP_K = multiview.update_after_resize(AP_K, image_shape, heatmap_shape)
        LAT_K = multiview.update_after_resize(LAT_K, image_shape, heatmap_shape)

        AP_proj_matricies = torch.bmm(AP_K, AP_T)
        LAT_proj_matricies = torch.bmm(LAT_K, LAT_T)
        proj_matricies = torch.stack([AP_proj_matricies,LAT_proj_matricies],dim=1)

        features = torch.stack([AP_feat,LAT_feat],dim=1)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            center_2d_coord = torch.tensor([i / 2 for i in heatmap_shape],device=device)
            base_point = multiview.triangulate_point_from_multiple_views_linear_torch(proj_matricies[batch_i],center_2d_coord.unsqueeze(0).repeat(2,1))
            base_points[batch_i] = base_point

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point.detach().cpu().numpy() - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)
            coord_volumes[batch_i] = coord_volume

        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=True)

        return vol_keypoints_3d, volumes, coord_volumes, AP_hmap, LAT_hmap

class Vol3dResNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.volume_aggregation_method = config.volume_aggregation_method

        # volume
        self.volume_multiplier = 50
        self.volume_size = 64
        self.cuboid_side = 250

        self.use_confidences = config.use_confidences

        self.backbone = Loc_SCN(config.num_classes,1,alg_confidences=False,vol_confidences=self.use_confidences)
        
        if config.backbone_weightFile != "none":
            checkpoint = torch.load(config.backbone_weightFile, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['state_dict'],strict=False)
        
        self.process_features = nn.Sequential(
            nn.Conv2d(16, 16, 1)
        )

        self.volume_net = V2VResModel(16,config.num_classes)
        self.local_downsampled = nn.AdaptiveAvgPool3d((32,32,32))

    def forward(self, AP_img, AP_K, AP_T, LAT_img, LAT_K, LAT_T):
        device = AP_img.device
        batch_size = AP_img.shape[0]

        if self.use_confidences:
            AP_hmap, AP_feat, _, AP_vol_confidences = self.backbone(AP_img)
            LAT_hmap, LAT_feat, _, LAT_vol_confidences = self.backbone(LAT_img)
            vol_confidences = torch.stack([AP_vol_confidences,LAT_vol_confidences],dim=1)
        else:
            AP_hmap, AP_feat, _, _ = self.backbone(AP_img)
            LAT_hmap, LAT_feat, _, _ = self.backbone(LAT_img)
            vol_confidences = torch.ones(batch_size, 2, AP_hmap.shape[1]).type(torch.float).to(device)

        if self.use_confidences:
            vol_confidences = vol_confidences + torch.tensor([1e-32],device=device)
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # resize K
        image_shape, heatmap_shape = tuple(AP_img.shape[2:]), tuple(AP_feat.shape[2:])
        AP_K = multiview.update_after_resize(AP_K, image_shape, heatmap_shape)
        LAT_K = multiview.update_after_resize(LAT_K, image_shape, heatmap_shape)

        AP_proj_matricies = torch.bmm(AP_K, AP_T)
        LAT_proj_matricies = torch.bmm(LAT_K, LAT_T)
        proj_matricies = torch.stack([AP_proj_matricies,LAT_proj_matricies],dim=1)

        features = torch.stack([AP_feat,LAT_feat],dim=1)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            center_2d_coord = torch.tensor([i / 2 for i in heatmap_shape],device=device)
            base_point = multiview.triangulate_point_from_multiple_views_linear_torch(proj_matricies[batch_i],center_2d_coord.unsqueeze(0).repeat(2,1))
            base_points[batch_i] = base_point

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point.detach().cpu().numpy() - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)
            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, 2, *features.shape[1:])

        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        volumes = self.local_downsampled(volumes)
        volumes = self.volume_net(volumes)
        volumes = F.interpolate(volumes, [64,64,64], mode='trilinear')

        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=True)

        return vol_keypoints_3d, volumes, coord_volumes, AP_hmap, LAT_hmap