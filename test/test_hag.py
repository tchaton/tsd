from tsd.hag_utils import graph_to_hag
import torch


edge_indexes = torch.randint(0, 100, (2, 100))
direction = 1
print(edge_indexes)

print(graph_to_hag(edge_indexes, direction))