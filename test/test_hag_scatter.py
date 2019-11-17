from tsd.hag_scatter import scatter_max
from time import time
from tsd import scatter_max as sc_max
import torch

num_samples = 100
edge_samples = 1000
input = torch.randn((num_samples, 5))

edge_i = torch.randint(0, num_samples, (edge_samples,)).unsqueeze(0)
edge_j = torch.randint(0, num_samples, (edge_samples,)).unsqueeze(0)

edge_indexes = torch.cat([edge_i, edge_j])

input_expanded = torch.index_select(input, 0, edge_indexes[0])

t0 = time()
output = scatter_max(input_expanded, edge_indexes, torch.zeros_like(input), 0)
print(time() - t0, output[0])

t0 = time()
out, argmax = sc_max(input_expanded, edge_indexes[0], 0, fill_value=0)
print(time() - t0, out[0])
