#include <torch/extension.h>
#include<torch/torch.h>
#include<iostream>

void find_covers(at::Tensor t1, at::Tensor t2){
    std::tuple<at::Tensor, at::Tensor> unique_t2 = at::_unique(t2);
    at::Tensor uniques_t2 = std::get<0>(unique_t2);
    std::list<at::Tensor> covers;
    float trace = 0;
    for (int i = 0; i < uniques_t2.size(0); i++){
      at::Tensor mask = at::eq(t2, uniques_t2[i]);
      at::Tensor non_zero_mask = at::nonzero(mask);
      at::Tensor non_zero_mask_filtered = at::narrow(non_zero_mask, 1, 1, 1);
      at::Tensor t1_filtered = at::index_select(t1, 1, at::squeeze(non_zero_mask_filtered));
      std::cout << t1_filtered << std::endl;
    }
}

at::Tensor graph_to_hag(at::Tensor edge_indexes, int64_t direction) {
  int64_t dim = 0;
  int64_t start = 0;
  int64_t length = 1;

  if (direction == 1){
    at::Tensor source = at::narrow(edge_indexes, dim, start + 1, length);
    at::Tensor target = at::narrow(edge_indexes, dim, start, length);
    find_covers(target, source);
    return source;
  }else{
    at::Tensor source = at::narrow(edge_indexes, dim, start, length);
    at::Tensor target = at::narrow(edge_indexes, dim, start + 1, length);
    find_covers(source, target);
    return source;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("graph_to_hag", &graph_to_hag, "Graph to HAG (CPU)");
}