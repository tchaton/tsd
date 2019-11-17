#include <torch/extension.h>
#include <iostream>
#include "gnn_to_hag.hpp"

//====================================================================
// GNN Header Definitions
//====================================================================

at::Tensor graph_to_torch(V_ID nvNewSrc,
                    std::map<V_ID, std::set<V_ID>* >& inEdges);

void torch_to_graph(const at::Tensor edge_indexes,
                    std::map<V_ID, std::set<V_ID>* >& inEdges,
                    V_ID& maxNodeIndex,
                    E_ID& numEdges);

