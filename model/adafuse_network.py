import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from .pose_resnet import get_pose_net
from .epipolar_fusion_layer import CamFusionModule, get_inv_cam, get_inv_affine_transform
from .ada_weight_layer import ViewWeightNet
from .soft_argmax import SoftArgmax2D

class AdafuseNet(nn.Module):
    def __init__(self, config,device):
        super().__init__()
        self.resnet = get_pose_net(config,device)

        self.h = int(128)#need to change with dataset #192
        self.w = int(128)
        self.num_classes = config.num_classes

        self.cam_fusion_net = CamFusionModule(2, config.num_classes, self.h, self.h)
        self.smax = SoftArgmax2D(window_fn='Uniform', window_width=5*3, softmax_temp=0.05)
        self.view_weight_net = ViewWeightNet(config.num_classes, self.h, self.w)
    def forward(self, AP_img, AP_K, AP_T, LAT_img, LAT_K, LAT_T):
        device = AP_img.device

        inputs = torch.stack([AP_img,LAT_img],dim=1)
        batch = inputs.shape[0]
        nview = inputs.shape[1]
        inputs = inputs.view(batch*nview, *inputs.shape[2:])
        njoints = self.num_classes
        origin_hms, feature_before_final,_,_ = self.resnet(inputs.repeat(1,3,1,1))#[bs*n_view,5,192,128],[bs*n,256,192,128]

        scale_trans = np.eye(3,3)  # cropped img -> heatmap
        scale_trans[0,0] = 0.25
        scale_trans[1, 1] = 0.25
        aug_trans = torch.from_numpy(scale_trans).float().to(device)
        aug_trans = aug_trans.repeat(batch, nview, 1, 1)
        inv_affine_trans = torch.inverse(aug_trans)  # heatmap -> origin image

        # obtain camera in (batch, nview, ...)
        cam_Intri = torch.stack([AP_K,LAT_K],dim=1)[:,:,:,:3]
        cam_R = torch.stack([AP_T,LAT_T],dim=1)[:,:,:3,:3]
        standard_cam_T = torch.stack([AP_T,LAT_T],dim=1)[:,:,:3,3].unsqueeze(3)
        cam_T = torch.bmm(self.collate_first_two_dims(torch.inverse(cam_R)),
                        self.collate_first_two_dims(-standard_cam_T)).view(batch, nview, standard_cam_T.size(-2), standard_cam_T.size(-1))
        pmat = torch.bmm(self.collate_first_two_dims(cam_Intri),
                         self.collate_first_two_dims(torch.cat((cam_R, cam_T), dim=3)))
        # # Notice: T is not h36m t, should be standard t
        pmat = pmat.view(batch, nview, 3, 4)
        fund_mats2 = self.get_fund_mat_pairs(cam_R, standard_cam_T, cam_Intri)  # (batch, nview, nview, 3, 3)

        # camera in (batch*nview, ...)
        cam_R_collate = self.collate_first_two_dims(cam_R)
        cam_T_collate = self.collate_first_two_dims(cam_T)
        standard_cam_T_collate = self.collate_first_two_dims(standard_cam_T)
        cam_Intri_collate = self.collate_first_two_dims(cam_Intri)
        aug_trans_collate = self.collate_first_two_dims(aug_trans)
        inv_affine_trans_collate = self.collate_first_two_dims(inv_affine_trans)

        # --- --- view weight network forward
        hms_nofusion = origin_hms
        j2d_nofusion, j2d_nofusion_maxv = self.smax(hms_nofusion)
        j2d_nofusion_img = heatmap_coords_to_image(j2d_nofusion, inv_affine_trans)
        j2d_nofusion_img = j2d_nofusion_img.view(batch, nview, 3, njoints)
        confi = j2d_nofusion_maxv.view(batch, nview, njoints)
        distances = torch.zeros(batch, nview, nview-1, njoints).to(device)
        confidences = torch.zeros(batch, nview, nview-1, njoints).to(device)
        for b in range(batch):
            for i in range(nview):
                cv_joints = j2d_nofusion_img[b, i]  # cv-current view, col vector
                other_views = set(range(nview))
                other_views.remove(i)

                for idx_j, j in enumerate(other_views):
                    ov_joints = j2d_nofusion_img[b, j]  # ov-other view
                    # fund_mat = fund_mats[(b, j, i)]
                    fund_mat = fund_mats2[b,i,j]  # F_ij
                    l_i = torch.matmul(fund_mat, ov_joints)
                    distance_d = torch.sum(cv_joints * l_i, dim=0)**2
                    tmp_l_i = l_i**2
                    lp_i = torch.matmul(fund_mat.t(), cv_joints)
                    tmp_lp_i = lp_i**2
                    distance_div = tmp_l_i[0, :] + tmp_l_i[1, :] + tmp_lp_i[0, :] + tmp_lp_i[1, :]
                    distance = distance_d / distance_div  # Sampson first order here
                    distances[b, i, idx_j] = distance
                    confidences[b, i, idx_j] = confi[b, i]
        distances = torch.sqrt(distances)

        view_weight = self.view_weight_net(feat_map=feature_before_final.detach(), maxv=confi, heatmap=hms_nofusion.detach(),
                                           distances=distances, confidences=confidences)
        # view_weight (batch, njoint, nview)
        # --- End --- view weight network forward

        # --- fuse heatmaps with learned weight
        hms = hms_nofusion  # (batch*nview, n_used_joint, h, w)
        maxv = j2d_nofusion_maxv
        large_num = torch.ones_like(maxv) * 1e6
        maxv = torch.where(maxv>0.01, maxv, large_num)
        maxv = maxv.view(batch*nview, njoints, 1, 1)
        hms_norm = hms/maxv
        xview_self_hm_norm = self.cam_fusion_net(hms_norm, aug_trans_collate,
                                                                 cam_Intri_collate, cam_R_collate, cam_T_collate,
                                                                 inv_affine_trans_collate, standard_cam_T_collate)
        warp_weight = self.get_warp_weight(view_weight)
        cat_hm = torch.cat((hms_norm.unsqueeze(dim=1), xview_self_hm_norm), dim=1)\
            .view(batch, nview, nview, njoints, self.h, self.w)
        fused_hm = warp_weight * cat_hm
        fused_hm = torch.sum(fused_hm, dim=2)
        fused_hm = fused_hm.view(batch * nview, *fused_hm.shape[2:])

        j2d_fused, j2d_fused_maxv, j2d_fused_smax = self.smax(fused_hm, out_smax=True)  # shape of (batch, 3, njoint) and (batch, njoint)
        j2d_fused_image = heatmap_coords_to_image(j2d_fused, inv_affine_trans)
        j2d_fused_image = j2d_fused_image.view(batch, nview, 3, njoints)

        # ---End- fuse heatmaps with learned weight
        return j2d_fused_image[:,:,:2,:],fused_hm#.view(batch, nview, -1, fused_hm.size(-2), fused_hm.size(-1))
    
    def get_warp_weight(self, view_weight):
        """

        :param view_weight: (batch, njoints, nview)
        :return: weights for merging warpped heatmap of shape (batch, nview, nview, njoints, 1, 1)
        """
        batch, njoints, nview = view_weight.shape
        dev = view_weight.device
        warp_weight = torch.zeros(batch, nview, nview, njoints).to(dev)
        warp_vw = view_weight.view(batch, njoints, nview).permute(0, 2, 1).contiguous()  # (batch, nview, njoint)
        for ci in range(nview):
            warp_weight[:, ci, 0] = warp_vw[:, ci]  # cur view weight at first
            # warp_weight[:, ci, 0] = 0
            all_views = list(range(nview))
            all_views.remove(ci)
            for idx, oi in enumerate(all_views):  # other views
                warp_weight[:, ci, idx + 1] = warp_vw[:, oi]
        warp_weight = warp_weight.view(*warp_weight.shape, 1, 1)
        return warp_weight

    def collate_first_two_dims(self, tensor):
        dim0 = tensor.shape[0]
        dim1 = tensor.shape[1]
        left = tensor.shape[2:]
        return tensor.view(dim0 * dim1, *left)
    
    def get_fund_mat_pairs(self, cam_R, cam_T, cam_Intri):
        """

        :param cam_R: (batch, nview, 3, 3)
        :param cam_T:
        :param cam_Intri:
        :return:
        """
        assert len(cam_R.shape) == 4, 'wrong shape of camera parameters'
        dev = cam_R.device
        batch, nview = cam_R.shape[0:2]
        # to get fundamental mat every two views, we need cam R,T,K in (batch, nview, nview-1)
        batch_camR_1 = torch.zeros(batch, nview, nview, 3, 3, device=dev)
        batch_camT_1 = torch.zeros(batch, nview, nview, 3, 1, device=dev)
        batch_camK_1 = torch.zeros(batch, nview, nview, 3, 3, device=dev)
        batch_camR_2 = torch.zeros(batch, nview, nview, 3, 3, device=dev)
        batch_camT_2 = torch.zeros(batch, nview, nview, 3, 1, device=dev)
        batch_camK_2 = torch.zeros(batch, nview, nview, 3, 3, device=dev)
        for b in range(batch):
            for i in range(nview):
                for j in range(nview):
                    batch_camR_1[b, i, j] = cam_R[b, j]
                    batch_camR_2[b, i, j] = cam_R[b, i]
                    batch_camT_1[b, i, j] = cam_T[b, j]
                    batch_camT_2[b, i, j] = cam_T[b, i]
                    batch_camK_1[b, i, j] = cam_Intri[b, j]
                    batch_camK_2[b, i, j] = cam_Intri[b, i]

        batch_camR_1 = batch_camR_1.view(-1, 3, 3)
        batch_camT_1 = batch_camT_1.view(-1, 3, 1)
        batch_camK_1 = batch_camK_1.view(-1, 3, 3)
        batch_camR_2 = batch_camR_2.view(-1, 3, 3)
        batch_camT_2 = batch_camT_2.view(-1, 3, 1)
        batch_camK_2 = batch_camK_2.view(-1, 3, 3)
        fund_mats2 = get_batch_fundamental_mat(batch_camR_1, batch_camT_1, batch_camK_1,
                                               batch_camR_2, batch_camT_2, batch_camK_2)
        fund_mats2 = fund_mats2.view(batch, nview, nview, 3, 3)
        return fund_mats2

def get_batch_fundamental_mat(r1, t1, k1, r2, t2, k2):
    """

    :param r1:
    :param t1: in h36m t style
    :param k1:
    :param r2:
    :param t2:
    :param k2:
    :return:
    """
    nbatch = r1.shape[0]
    r = torch.bmm(r2, r1.permute(0,2,1))
    # t = torch.bmm(r2, (t1 - t2))  # t is h36m t.
    t = t2 - torch.bmm(r,t1)
    t = t.view(nbatch,3)
    t_mat = torch.zeros(nbatch, 3, 3).type_as(r1)  # cross product matrix
    t_mat[:, 0, 1] = -t[:, 2]
    t_mat[:, 0, 2] = t[:, 1]
    t_mat[:, 1, 2] = -t[:, 0]
    t_mat = -t_mat.permute(0,2,1) + t_mat
    inv_k1 = torch.inverse(k1)
    inv_k2 = torch.inverse(k2)
    inv_k2_t = inv_k2.permute(0,2,1)
    fundmat = torch.bmm(inv_k2_t, torch.bmm(t_mat, torch.bmm(r, inv_k1)))

    return fundmat

def heatmap_coords_to_image(coords, inv_affine_trans):
    """

    :param coords: (batch*nview, 3, njoints)
    :param inv_affine_trans: (batch, nview, 3, 3)
    :return:
    """
    if len(inv_affine_trans.shape) == 4:
        inv_affine_trans = inv_affine_trans.view(-1, 3, 3)
    coords_img = torch.bmm(inv_affine_trans, coords)
    return coords_img