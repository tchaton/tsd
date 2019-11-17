#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "hag.hpp"
#include "gnn_to_hag.hpp"

void scatter_max_(std::map<V_ID, std::set<V_ID>*>& orgList,
                    std::map<V_ID, std::set<V_ID>*>& optList,
                    std::vector<std::pair<V_ID, V_ID> >& ranges,
                    V_ID nv, at::Tensor src, at::Tensor out, int64_t dim){

}

void scatter_max(at::Tensor src, at::Tensor edge_indexes, at::Tensor out, int64_t dim) {
  std::map<V_ID, std::set<V_ID>*> inEdges;
  V_ID maxNodeIndex = 0;
  E_ID numEdges = 0;
  V_ID maxDepth = 10;
  V_ID maxWidth = 10;
  torch_to_graph(edge_indexes, inEdges, maxNodeIndex, numEdges);

  std::map<V_ID, std::set<V_ID>*> optInEdges;
  std::vector<std::pair<V_ID, V_ID> > optRanges;
  V_ID new_max_node_index;
  transfer_graph(inEdges, optInEdges, optRanges, maxNodeIndex, numEdges, maxDepth, maxWidth, new_max_node_index);
  scatter_max_(inEdges, optInEdges, optRanges, new_max_node_index, src, out, dim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_max", &scatter_max, "Scatter Max (CPU)");
}
