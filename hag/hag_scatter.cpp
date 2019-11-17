#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "hag.hpp"
#include "gnn_to_hag.hpp"

void scatter_max(at::Tensor src, at::Tensor edge_indexes, at::Tensor out, int64_t dim) {
  std::map<V_ID, std::set<V_ID>* > inEdges;
  V_ID maxNodeIndex = 0;
  E_ID numEdges = 0;
  V_ID maxDepth = 10;
  V_ID maxWidth = 10;
  torch_to_graph(edge_indexes, inEdges, maxNodeIndex, numEdges);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_max", &scatter_max, "Scatter Max (CPU)");
}
