import torch
from oriented_iou_loss import cal_iou

device = torch.device('cuda:0')
box_0 = torch.tensor([.0, .0, 2., 2., .0], device=device)

def test_same_box():
    expect = 0.
    box_1 = torch.tensor([.0, .0, 2., 2., .0], device=device)
    result = cal_iou(box_0[None, None, ...], box_1[None, None, ...])[0].cpu().numpy()
    print("expect:", expect, "get:", result[0,0])

def test_same_edge():
    expect = 0.
    box_1 = torch.tensor([.0, 2, 2., 2., .0], device=device)
    result = cal_iou(box_0[None, None, ...], box_1[None, None, ...])[0].cpu().numpy()
    print("expect:", expect, "get:", result[0,0])

def test_same_edge_offset():
    expect = 0.3333
    box_1 = torch.tensor([.0, 1., 2., 2., .0], device=device)
    result = cal_iou(box_0[None, None, ...], box_1[None, None, ...])[0].cpu().numpy()
    print("expect:", expect, "get:", result[0,0])

if __name__ == "__main__":
    test_same_box()
    test_same_edge()
    test_same_edge_offset()