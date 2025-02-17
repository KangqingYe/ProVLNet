import torch

def get_mae(pred_pts, gt_pts):
    # 计算预测点和真实点之间的绝对差值
    absolute_difference = torch.abs(pred_pts - gt_pts)
    # 求和每个维度的差异，得到每个点的绝对误差
    per_point_error = torch.sum(absolute_difference, dim=-1)
    # 计算所有点的平均绝对误差
    # mae = torch.mean(per_point_error)
    return per_point_error
def get_pe(pred_pts,gt_pts):
    return torch.sqrt(torch.sum((pred_pts - gt_pts) ** 2, dim=-1))
def get_min_distance_mask(pred_pts, gt_pts):
    """
    Given predicted points and ground truth points, 
    returns a mask indicating which ground truth are closest to their corresponding prediction.
    
    Args:
        pred_pts (torch.Tensor): Predicted points of shape [1, N, 2]
        gt_pts (torch.Tensor): Ground truth points of shape [1, N, 2]
        
    Returns:
        torch.Tensor: Binary mask of shape [1, N] where 1 indicates correct prediction and 0 indicates incorrect prediction.
    """
    distances = get_pe(pred_pts.unsqueeze(2), gt_pts.unsqueeze(1))  # shape: [1, N, N]
    
    # Find the index of the predicted point that is closest to each ground truth point.
    min_distance_indices = distances.argmin(dim=2)  # shape: [1, N]
    
    correct_mask = (torch.arange(pred_pts.size(1), device=pred_pts.device) == min_distance_indices).float()
    
    return correct_mask