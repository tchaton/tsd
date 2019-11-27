from tsd.hag import graph_to_hag
from tsd import scatter_max
import torch

print("Simple graph")
# 1 -> droite à gauche : 1 == source, 0 == target
input_gnn = torch.tensor([ [0, 0, 0, 1, 1, 1, 2, 2, 3, 3], [1, 2, 3, 0, 2, 3, 0, 1, 0, 1] ])
expected_hag = torch.tensor([ [1, 5, 0, 5, 4, 4, 0, 1, 2, 3], [0, 0, 1, 1, 2, 3, 4, 4, 5, 5] ])

# ARGUMENTS
direction = 1
maxDepth = 10
maxWidth = 10

actual_hag = graph_to_hag(input_gnn, direction, maxDepth, maxWidth)

#print("direction", direction)
#print("input_gnn", input_gnn)

#print("expected_hag", expected_hag)
#print("actual_hag", torch.t(actual_hag))

assert torch.equal(torch.t(actual_hag), expected_hag)

print("Complex graph")
# 1 -> droite à gauche : 1 == source, 0 == target
input_gnn = torch.tensor([ [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6],
                           [1, 2, 3, 4, 0, 5, 6, 0, 5, 6, 0, 6, 0, 6, 1, 2, 1, 2, 3, 4] ])
expected_hag = torch.tensor([ [11, 10, 10,  7,  7,  8, 11,  0,  6,  1,  2,  3,  4,  5,  7,  8,  9],
                              [0,  1,  2,  3,  4,  5,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11, 11] ])
direction = 1

actual_hag = graph_to_hag(input_gnn, direction, maxDepth, maxWidth)

print("direction", direction)
print("input_gnn", input_gnn)

print("expected_hag", expected_hag)
print("actual_hag", torch.t(actual_hag))

assert torch.equal(torch.t(actual_hag), expected_hag)

def test_max_fill_value():
    device = torch.device('cuda:0')
    print(torch.cuda.get_device_name())
    src = torch.Tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6]]).cuda()
    index = torch.tensor([[1, 2, 3, 4, 0, 5, 6, 0, 5, 6, 0, 6, 0, 6, 1, 2, 1, 2, 3, 4]]).cuda()

    actual_hag = graph_to_hag(input_gnn, direction, maxDepth, maxWidth)
    
    out, _ = scatter_max(src, index)

    assert out.tolist() == [[4., 6., 6., 6., 6., 2., 4.]]

test_max_fill_value()