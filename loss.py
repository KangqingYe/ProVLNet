import torch
import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = torch.Tensor([1e-3]).cuda()
    
    def forward(self, input, label):
        input = input.flatten()
        label = label.flatten()
        intersect = torch.dot(input, label)
        input_sum = torch.sum(torch.dot(input, input))
        label_sum = torch.sum(torch.dot(label, label))
        union = input_sum + label_sum
        loss = (2 * intersect + self.epsilon) / (union + self.epsilon)
        return loss

class KeypointsMSESmoothLoss(nn.Module):
    def __init__(self, threshold=100):
        super().__init__()

        self.threshold = threshold

    def forward(self, keypoints_pred, keypoints_gt):
        dimension = keypoints_pred.shape[-1]
        diff = (keypoints_gt - keypoints_pred) ** 2
        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
        loss = torch.sum(diff) / (dimension * keypoints_gt.shape[0] * keypoints_gt.shape[1])
        return loss

class KeypointsMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum(torch.abs(keypoints_gt - keypoints_pred))
        loss = loss / (dimension * keypoints_gt.shape[0] * keypoints_gt.shape[1])
        return loss

class KeypointsL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt):
        loss = torch.sum(torch.sqrt(torch.sum((keypoints_gt - keypoints_pred) ** 2, dim=2)))
        loss = loss / (keypoints_gt.shape[0] * keypoints_gt.shape[1])
        return loss

class VolumetricCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coord_volumes_batch, volumes_batch_pred, keypoints_gt, keypoints_binary_validity):
        loss = 0.0
        n_losses = 0

        batch_size = volumes_batch_pred.shape[0]
        for batch_i in range(batch_size):
            coord_volume = coord_volumes_batch[batch_i]
            keypoints_gt_i = keypoints_gt[batch_i]

            coord_volume_unsq = coord_volume.unsqueeze(0)
            keypoints_gt_i_unsq = keypoints_gt_i.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            dists = torch.sqrt(((coord_volume_unsq - keypoints_gt_i_unsq) ** 2).sum(-1))
            dists = dists.view(dists.shape[0], -1)

            min_indexes = torch.argmin(dists, dim=-1).detach().cpu().numpy()
            min_indexes = np.stack(np.unravel_index(min_indexes, volumes_batch_pred.shape[-3:]), axis=1)

            for joint_i, index in enumerate(min_indexes):
                validity = keypoints_binary_validity[batch_i, joint_i]
                loss += validity[0] * (-torch.log(volumes_batch_pred[batch_i, joint_i, index[0], index[1], index[2]] + 1e-6))
                n_losses += 1


        return loss / n_losses