import torch
import numpy as np
from box_intersection_2d import oriented_box_intersection_2d

def box2corners_th(box:torch.Tensor)-> torch.Tensor:
    """convert box coordinate to corners

    Args:
        box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha

    Returns:
        torch.Tensor: (B, N, 4, 2) corners
    """
    B = box.size()[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5] # (B, N, 1)
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).cuda() # (1,1,4)
    x4 = x4 * w     # (B, N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).cuda()
    y4 = y4 * h     # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
    rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated

def cal_iou(box1:torch.Tensor, box2:torch.Tensor):
    """calculate iou

    Args:
        box1 (torch.Tensor): (B, N, 5)
        box2 (torch.Tensor): (B, N, 5)
    
    Returns:
        iou (torch.Tensor): (B, N)
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners1 (torch.Tensor): (B, N, 4, 2)
        U (torch.Tensor): (B, N) area1 + area2 - inter_area
    """
    corners1 = box2corners_th(box1)
    corners2 = box2corners_th(box2)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)        #(B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    u = area1 + area2 - inter_area
    iou = inter_area / u
    return iou, corners1, corners2, u

def cal_diou(box1:torch.Tensor, box2:torch.Tensor):
    """calculate diou loss

    Args:
        box1 (torch.Tensor): [description]
        box2 (torch.Tensor): [description]
    """
    iou, corners1, corners2, u = cal_iou(box1, box2)
    w, h = enclosing_box(corners1, corners2)
    c2 = w*w + h*h      # (B, N)
    x_offset = box1[...,0] - box2[..., 0]
    y_offset = box1[...,1] - box2[..., 1]
    d2 = x_offset*x_offset + y_offset*y_offset
    diou_loss = 1. - iou + d2/c2
    return diou_loss, iou

def cal_giou(box1:torch.Tensor, box2:torch.Tensor):
    iou, corners1, corners2, u = cal_iou(box1, box2)
    w, h = enclosing_box(corners1, corners2)
    area_c =  w*h
    giou_loss = 1. - iou + ( area_c - u )/area_c
    return giou_loss, iou

def enclosing_box(corners1:torch.Tensor, corners2:torch.Tensor):
    """calculate the smallest enclosing box. axis-aligned.

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
    
    Returns:
        w (torch.Tensor): (B, N)
        h (torch.Tensor): (B, N)
    """
    x1_max = torch.max(corners1[..., 0], dim=2)[0]     # (B, N)
    x1_min = torch.min(corners1[..., 0], dim=2)[0]     # (B, N)
    y1_max = torch.max(corners1[..., 1], dim=2)[0]
    y1_min = torch.min(corners1[..., 1], dim=2)[0]
    
    x2_max = torch.max(corners2[..., 0], dim=2)[0]     # (B, N)
    x2_min = torch.min(corners2[..., 0], dim=2)[0]    # (B, N)
    y2_max = torch.max(corners2[..., 1], dim=2)[0]
    y2_min = torch.min(corners2[..., 1], dim=2)[0]

    x_max = torch.max(x1_max, x2_max)
    x_min = torch.min(x1_min, x2_min)
    y_max = torch.max(y1_max, y2_max)
    y_min = torch.min(y1_min, y2_min)

    w = x_max - x_min       # (B, N)
    h = y_max - y_min
    return w, h

if __name__ == "__main__":
    from utiles import box2corners
    box = np.array([1, 1, 2, 3, np.pi/3])
    corners = box2corners(*box)

    box_b = torch.FloatTensor(box).unsqueeze(0).unsqueeze(0)
    corners_b = box2corners_th(box_b)
    print(corners)
    print(corners_b)