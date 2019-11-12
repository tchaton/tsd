#include <torch/extension.h>
#include<torch/torch.h>
#include<iostream>
#include <vector> 

std::vector<std::tuple<at::Tensor, at::Tensor>> find_covers(at::Tensor t1, at::Tensor t2){
    std::tuple<at::Tensor, at::Tensor> unique_t2 = at::_unique(t2);
    at::Tensor uniques_t2 = std::get<0>(unique_t2);
    std::vector<std::tuple<at::Tensor, at::Tensor>> covers;
    for (int i = 0; i < uniques_t2.size(0); i++){
      at::Tensor u = uniques_t2[i];
      at::Tensor mask = at::eq(t2, u);
      at::Tensor non_zero_mask = at::nonzero(mask);
      at::Tensor non_zero_mask_filtered = at::narrow(non_zero_mask, 1, 1, 1);
      at::Tensor t1_filtered = at::index_select(t1, 1, at::squeeze(non_zero_mask_filtered));
      covers.push_back(std::make_tuple(u, t1_filtered));
    }
    return covers;
}


at::Tensor graph_to_hag(at::Tensor edge_indexes, int64_t direction) {
  int64_t dim = 0;
  int64_t start = 0;
  int64_t length = 1;

  at::Tensor source;
  at::Tensor target;
  std::vector<std::tuple<at::Tensor, at::Tensor>> covers;

  source = at::narrow(edge_indexes, dim, direction, length);
  target = at::narrow(edge_indexes, dim, (direction + 1) % 2, length);
  covers = find_covers(target, source);
  return source;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("graph_to_hag", &graph_to_hag, "Graph to HAG (CPU)");
}