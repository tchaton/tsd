from tsd.hag_scatter import scatter_max
import torch

num_samples = 100
edge_samples = 1000
input = torch.randn((num_samples, 5))

edge_i = torch.randint(0, num_samples, (edge_samples,)).unsqueeze(0)
edge_j = torch.randint(0, num_samples, (edge_samples,)).unsqueeze(0)

edge_indexes = torch.cat([edge_i, edge_j])

output = scatter_max(input, edge_indexes, torch.zeros_like(input), 0)