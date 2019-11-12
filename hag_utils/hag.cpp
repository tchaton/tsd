#include <torch/extension.h>
#include<torch/torch.h>
#include<iostream>
#include <vector> 

struct covers_t {
  at::Tensor value;
  at::Tensor matches;
};

std::vector<covers_t> find_covers(at::Tensor t1, at::Tensor t2){
    std::tuple<at::Tensor, at::Tensor> unique_t2 = at::_unique(t2);
    at::Tensor uniques_t2 = std::get<0>(unique_t2);
    std::vector<covers_t> covers;
    for (int i = 0; i < uniques_t2.size(0); i++){
      at::Tensor u = uniques_t2[i];
      at::Tensor mask = at::eq(t2, u);
      at::Tensor non_zero_mask = at::nonzero(mask);
      at::Tensor non_zero_mask_filtered = at::narrow(non_zero_mask, 1, 1, 1);
      at::Tensor t1_filtered = at::index_select(t1, 1, at::squeeze(non_zero_mask_filtered));
      covers_t cover;
      cover.value = u;
      cover.matches = t1_filtered;
      covers.push_back(cover);
    }
    return covers;
}

int redundancy(covers_t cover_1, covers_t cover_2){
  at::Tensor maches_1 = cover_1.matches;
  at::Tensor maches_2 = cover_2.matches;
  at::Tensor maches_1_extended = at::unsqueeze(maches_1, -1);
  at::Tensor redundancy_count = at::sum(maches_1_extended == maches_2);
  return redundancy_count.item<float>();
}

std::tuple<covers_t, covers_t> find_argmax_from_redundancy(std::vector<covers_t> covers, at::Tensor source, at::Tensor target){
  float highest_capacity = -1;
  covers_t cover_i; 
  covers_t cover_j; 
  for (int i = 0; i < covers.size(); i++){
    for (int j = i + 1; j < covers.size(); j++){
      float capacity = redundancy(covers.at(i), covers.at(j));
      if (highest_capacity < capacity){
        highest_capacity = capacity;
        cover_i = covers.at(i);
        cover_j = covers.at(j);
      }
    }
  }
  return {cover_i, cover_j};
}

at::Tensor graph_to_hag(at::Tensor edge_indexes, int64_t direction, int64_t capacity = 100) {
  int64_t dim = 0;
  int64_t start = 0;
  int64_t length = 1;

  at::Tensor source;
  at::Tensor target;
  std::vector<covers_t> covers;

  source = at::narrow(edge_indexes, dim, direction, length);
  target = at::narrow(edge_indexes, dim, (direction + 1) % 2, length);
  covers = find_covers(target, source);

  std::tuple<covers_t, covers_t> cover_pair = find_argmax_from_redundancy(covers, source, target);

  return source;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("graph_to_hag", &graph_to_hag, "Graph to HAG (CPU)");
}