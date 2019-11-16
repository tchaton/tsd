#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "gnn_to_hag.hpp"

at::Tensor graph_to_torch(V_ID nvNewSrc,
                    std::map<V_ID, std::set<V_ID>* >& inEdges)
{
  int numEdges = 0;
  for (V_ID v = 0; v < nvNewSrc; v++)
    if (inEdges.find(v) != inEdges.end())
      numEdges += inEdges[v]->size();

  at::Tensor edge_indexes = torch::zeros({numEdges, 2}, torch::TensorOptions().dtype(torch::kInt64));

  E_ID count = 0;
  for (V_ID v = 0; v < nvNewSrc; v++) {
    if (inEdges.find(v) != inEdges.end()) {
      std::set<V_ID>::const_iterator first = inEdges[v]->begin();
      std::set<V_ID>::const_iterator last = inEdges[v]->end();
      std::set<V_ID>::const_iterator it = first;
      for (it = first; it != last; it++) {
        edge_indexes[count][0] = *it;
        edge_indexes[count][1] = v;
        count ++;
      }

    }
  }
  return edge_indexes;
}

void torch_to_graph(const at::Tensor edge_indexes,
                    std::map<V_ID, std::set<V_ID>* >& inEdges,
                    V_ID& maxNodeIndex,
                    E_ID& numEdges)
{
  maxNodeIndex = 0;
  numEdges = 0;
  auto indexes_accessor = edge_indexes.accessor<long,2>();
  for (V_ID i = 0; i < indexes_accessor.size(1); i++)
  {
    V_ID source = indexes_accessor[0][i];
    V_ID target = indexes_accessor[1][i];

    if (std::max(source, target) >= maxNodeIndex)
      maxNodeIndex = std::max(source, target) + 1;

    // Populate the map of edges.
    if (inEdges.find(target) == inEdges.end())
      inEdges[target] = new std::set<V_ID>();
    inEdges[target]->insert(source);
  }
}

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
