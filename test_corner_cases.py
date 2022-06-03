'''
test cases for issue #8
check the calculation if:
    1. edges are collinear
    2. boxes are exactly the same
    3. input in different scale level
'''
import torch
from oriented_iou_loss import cal_iou

device = torch.device('cpu')
dtpye = torch.float32
box_0 = torch.tensor([.0, .0, 2., 2., .0], device=device).type(dtpye)


def test_same_box():
    expect = 1.
    box_1 = torch.tensor([.0, .0, 2., 2., .0], device=device).type(dtpye)
    result = cal_iou(box_0[None, None, ...], box_1[None, None, ...])[0].cpu().numpy()
    print("expect:", expect, "get:", result[0,0])


def test_same_edge():
    expect = 0.
    box_1 = torch.tensor([.0, 2, 2., 2., .0], device=device).type(dtpye)
    result = cal_iou(box_0[None, None, ...], box_1[None, None, ...])[0].cpu().numpy()
    print("expect:", expect, "get:", result[0,0])


def test_same_edge_offset():
    expect = 0.3333
    box_1 = torch.tensor([.0, 1., 2., 2., .0], device=device).type(dtpye)
    result = cal_iou(box_0[None, None, ...], box_1[None, None, ...])[0].cpu().numpy()
    print("expect:", expect, "get:", result[0,0])


def test_same_box2():
    expect = 1
    box_1 = torch.tensor([38, 120, 1.3, 20, 50], device=device).type(dtpye)
    result = cal_iou(box_1[None, None, ...], box_1[None, None, ...])[0].cpu().numpy()
    print("expect:", expect, "get:", result[0,0])


def test_partial_same():
    expect = 48
    box1 = [4,5,8,10,0]
    box2 = [3,4,6,8,0]
    box1 = torch.tensor(box1).to(device).type(dtpye)
    box2 = torch.tensor(box2).to(device).type(dtpye)
    result = cal_iou(box1[None, None, ...], box2[None, None, ...])
    result = result[0] * result[-1]
    print("expect:", expect, "get:", result[0,0])

if __name__ == "__main__":
    test_same_box()
    test_same_edge()
    test_same_edge_offset()
    test_same_box2()
    test_partial_same()
    