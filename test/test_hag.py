from tsd.hag import graph_to_hag
from tsd.hag_scatter import scatter_max
import torch

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
