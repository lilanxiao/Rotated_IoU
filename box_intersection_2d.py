'''
torch implementation of 2d oriented box intersection

author: lanxiao li
2020.8
'''
import torch
from torch import Tensor
EPSILON = 1e-8


def box_intersection_th(corners1:Tensor, corners2:Tensor):
    """find intersection points of rectangles
    Convention: if two edges are collinear, there is no intersection point

    Args:
        corners1 (Tensor): B, N, 4, 2
        corners2 (Tensor): B, N, 4, 2

    Returns:
        intersectons (Tensor): B, N, 4, 4, 2
        mask (Tensor) : B, N, 4, 4; bool
    """
    # build edges from corners
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3) # B, N, 4, 4: Batch, Box, edge, point
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    # duplicate data to pair each edges from the boxes
    # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(3).repeat([1,1,1,4,1])
    line2_ext = line2.unsqueeze(2).repeat([1,1,4,1,1])
    x1 = line1_ext[..., 0]
    y1 = line1_ext[..., 1]
    x2 = line1_ext[..., 2]
    y2 = line1_ext[..., 3]
    x3 = line2_ext[..., 0]
    y3 = line2_ext[..., 1]
    x4 = line2_ext[..., 2]
    y4 = line2_ext[..., 3]
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    num = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)     
    den_t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
    t = den_t / num
    t[num == .0] = -1.
    mask_t = (t > 0) * (t < 1)                # intersection on line segment 1
    den_u = (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)
    u = -den_u / num
    u[num == .0] = -1.
    mask_u = (u > 0) * (u < 1)                # intersection on line segment 2
    mask = mask_t * mask_u 
    t = den_t / (num + EPSILON)                 # overwrite with EPSILON. otherwise numerically unstable
    intersections = torch.stack([x1 + t*(x2-x1), y1 + t*(y2-y1)], dim=-1)
    intersections = intersections * mask.type(corners1.dtype).unsqueeze(-1)
    return intersections, mask


def box1_in_box2(corners1:Tensor, corners2:Tensor):
    """check if corners of box1 lie in box2
    Convention: if a corner is exactly on the edge of the other box, it's also a valid point

    Args:
        corners1 (Tensor): (B, N, 4, 2)
        corners2 (Tensor): (B, N, 4, 2)

    Returns:
        c1_in_2: (B, N, 4) Bool
    """
    a = corners2[:, :, 0:1, :]  # (B, N, 1, 2)
    b = corners2[:, :, 1:2, :]  # (B, N, 1, 2)
    d = corners2[:, :, 3:4, :]  # (B, N, 1, 2)
    ab = b - a                  # (B, N, 1, 2)
    am = corners1 - a           # (B, N, 4, 2)
    ad = d - a                  # (B, N, 1, 2)
    p_ab = torch.sum(ab * am, dim=-1)       # (B, N, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)    # (B, N, 1)
    p_ad = torch.sum(ad * am, dim=-1)       # (B, N, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)    # (B, N, 1)
    # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
    # also stable with different scale of bboxes
    cond1 = (p_ab >= 0) & (p_ab <= norm_ab )   # (B, N, 4)
    cond2 = (p_ad >= 0) & (p_ad <= norm_ad )   # (B, N, 4)
    return cond1*cond2


@torch.no_grad()
def box_in_box_th(corners1:Tensor, corners2:Tensor):
    """check if corners of two boxes lie in each other

    Args:
        corners1 (Tensor): (B, N, 4, 2)
        corners2 (Tensor): (B, N, 4, 2)

    Returns:
        c1_in_2: (B, N, 4) Bool. i-th corner of box1 in box2
        c2_in_1: (B, N, 4) Bool. i-th corner of box2 in box1
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1


def build_vertices(corners1:Tensor, corners2:Tensor, 
                c1_in_2:Tensor, c2_in_1:Tensor, 
                inters:Tensor, mask_inter:Tensor):
    """find vertices of intersection area

    Args:
        corners1 (Tensor): (B, N, 4, 2)
        corners2 (Tensor): (B, N, 4, 2)
        c1_in_2 (Tensor): Bool, (B, N, 4)
        c2_in_1 (Tensor): Bool, (B, N, 4)
        inters (Tensor): (B, N, 4, 4, 2)
        mask_inter (Tensor): (B, N, 4, 4)
    
    Returns:
        vertices (Tensor): (B, N, 24, 2) vertices of intersection area. only some elements are valid
        mask (Tensor): (B, N, 24) indicates valid elements in vertices
    """
    # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0). 
    # can be used as trick
    B = corners1.size()[0]
    N = corners1.size()[1]
    vertices = torch.cat([corners1, corners2, inters.view([B, N, -1, 2])], dim=2) # (B, N, 4+4+16, 2)
    mask = torch.cat([c1_in_2, c2_in_1, mask_inter.view([B, N, -1])], dim=2) # Bool (B, N, 4+4+16)
    return vertices, mask


def sort_vertices(vertices:Tensor, mask:Tensor):
    """

    Args:
        vertices (Tensor): float (B, N, 24, 2)
        mask (Tensor): bool (B, N, 24)

    Returns:
        sorted vertices: (B, N, 9, 2)
    
    Note:
        why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X) 
        and X indicates the index of arbitary elements in the last 16 (intersections not corners) with 
        value 0 and mask False. (cause they have zero value and zero gradient)
    """
    num_valid = torch.sum(mask.int(), dim=2).int()      # (B, N)
    mean = torch.sum(vertices * mask.type(vertices.dtype).unsqueeze(-1), dim=2, keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
    vertices_normalized = vertices - mean       # normalization makes sorting easier
    idx_sorted = sort_vertice_th(vertices_normalized, mask, num_valid.long()).long()
    
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1,1,1,2])
    selected = torch.gather(vertices, 2, idx_ext)
    
    # zero padding for invalid vertices
    m = _generate_mask(selected.size(-2), num_valid+1, dtype=vertices.dtype, device=vertices.device)
    selected = selected * m.unsqueeze(-1)
    
    # set zero of too few vertices
    m = num_valid >= 3
    selected = selected * m.type(selected.dtype).unsqueeze(-1).unsqueeze(-1)
    
    return selected
    

def calculate_area(selected):
    """calculate area of intersection

    Args:
        idx_sorted (Tensor): (B, N, 9)
        vertices (Tensor): (B, N, 24, 2)
    
    return:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding 
    """
    total = selected[:, :, 0:-1, 0]*selected[:, :, 1:, 1] - selected[:, :, 0:-1, 1]*selected[:, :, 1:, 0]
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected


def oriented_box_intersection_2d(corners1:Tensor, corners2:Tensor):
    """calculate intersection area of 2d rectangles 

    Args:
        corners1 (Tensor): (B, N, 4, 2)
        corners2 (Tensor): (B, N, 4, 2)

    Returns:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding 
    """
    inters, mask_inter = box_intersection_th(corners1, corners2)
    c12, c21 = box_in_box_th(corners1, corners2)
    c12, c21 = check_overlap(corners1, corners2, c12, c21)
    
    vertices, mask = build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
    vertices_gathered = sort_vertices(vertices, mask)
    return calculate_area(vertices_gathered)


@torch.no_grad()
def check_overlap(corners1:Tensor, corners2:Tensor, cond12:Tensor, cond21:Tensor):
    """check if corners are overlapped and update the conditions. 
    useful to avoid incorrect intersection calculation. 
    Without this check, the intersection would have duplicated vertices, which makes the 
    shoelace-formula broken. 

    Args:
        corners1 (Tensor): (B, N, 4, 2)
        corners2 (Tensor): (B, N, 4, 2)
        cond12 (Tensor): bool, (B, N, 4)
        cond21 (Tensor): bool, (B, N, 4)

    Returns:
        Tensor: bool, (B, N, 4)
        Tensor: bool, (B, N, 4)
    """
    c_roll = corners2
    cd_roll = cond21
    for _ in range(4):
        c_roll = torch.roll(c_roll, shifts=1, dims=2)
        cd_roll = torch.roll(cd_roll, shifts=1, dims=2)
        crit = torch.all(corners1 == c_roll, dim=-1)
        cond12[crit] = True
        cd_roll[crit] = False
    return cond12, cd_roll


@torch.no_grad()
def sort_vertice_th(vertices_normalized:Tensor, mask:Tensor, num_valid:Tensor):
    """_summary_

    Args:
        vertices_normalized (Tensor): (B, N, 24, 2)
        mask (Tensor): (B, N, 24)
        num_valid (Tensor): (B, N)

    Returns:
        Tensor: (B, N, 9)
    """
    x = vertices_normalized[..., 0]
    y = vertices_normalized[..., 1]
    
    # sorting
    x[~mask] = -1e6
    y[~mask] = 1e-6
    ang = torch.atan2(y, x)
    index = torch.argsort(ang, dim=-1)  # (B, N, 24)
    
    # duplicate the first
    temp = index[..., :1].clone()   # (B, N, 1)
    index.scatter_(dim=-1, index=num_valid.unsqueeze(-1), src=temp.expand(-1, -1, index.size(-1)))
    return index[..., :9]


@torch.no_grad()
def _generate_mask(num: int, valid_num: Tensor, dtype, device):
    B, N = valid_num.size()
    ar = torch.arange(num, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(B, N, 1).to(device)
    mask = ar < valid_num.unsqueeze(-1)
    # NOTE: this expression doesn't work some earlier PyTorch version:
    # arr = torch.where(mask, 1., 0.)
    ar = torch.where(mask, torch.ones((1,)).expand_as(mask).to(device), torch.zeros(1,).expand_as(mask).to(device))
    return ar
