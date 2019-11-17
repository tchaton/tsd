#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "hag.hpp"
#include "gnn_to_hag.hpp"

at::Tensor graph_to_hag(at::Tensor edge_indexes, int64_t direction, int64_t maxDepth = 10, int64_t maxWidth = 10) {
  std::map<V_ID, std::set<V_ID>* > inEdges;
  V_ID maxNodeIndex = 0;
  E_ID numEdges = 0;
  torch_to_graph(edge_indexes, inEdges, maxNodeIndex, numEdges);

  std::map<V_ID, std::set<V_ID>*> optInEdges;
  std::vector<std::pair<V_ID, V_ID> > optRanges;
  V_ID new_max_node_index;
  transfer_graph(inEdges, optInEdges, optRanges, maxNodeIndex, numEdges, maxDepth, maxWidth, new_max_node_index);

  return graph_to_torch(new_max_node_index, optInEdges);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("graph_to_hag", &graph_to_hag, "Graph to HAG (CPU)");
}
