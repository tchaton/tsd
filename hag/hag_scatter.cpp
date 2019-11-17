#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "hag.hpp"
#include "gnn_to_hag.hpp"

void scatter_max_(std::map<V_ID, std::set<V_ID>*>& orgList,
                    std::map<V_ID, std::set<V_ID>*>& optList,
                    std::vector<std::pair<V_ID, V_ID> >& ranges,
                    V_ID nv, at::Tensor src, at::Tensor out, int64_t dim, int64_t maxWidth){
  at::Tensor tmp_ = torch::zeros({maxWidth, src.size(-1)}, torch::TensorOptions().dtype(torch::kFloat));
  for (int i = 0; i < ranges.size(); i++){
    V_ID rowLeft = ranges[i].first;
    V_ID rowRight = ranges[i].second;
    //std::cout << rowLeft << " " << rowRight << std::endl;
    for (V_ID j = rowLeft; j <= rowRight; j++){
      std::set<V_ID>::const_iterator it, first = orgList[i]->begin(),
                                    last = orgList[i]->end();
      at::Tensor tmp_element;
      for (it = first; it != last; it++){
        if (it == first){
          tmp_element = src[*it];
        }else{
          tmp_element = at::cat({tmp_element, src[*it]});
        }
      }
      std::tuple<at::Tensor, at::Tensor> max_ = tmp_element.max(0, false);
      tmp_[j - nv] = std::get<0>(max_);
    }
  }
  for (int i = 0; i < nv; i++){
    std::set<V_ID>::const_iterator it, first = orgList[i]->begin(),
                                   last = orgList[i]->end();  
    at::Tensor tmp_element;
    for (it = first; it != last; it++){
      if (it == first){
        if (*it > nv){
          tmp_element = tmp_[*it - nv];
        }else{
          tmp_element =  src[*it];
        }        
      }else{
        if (*it > nv){
          tmp_element = at::cat({tmp_element, tmp_[*it - nv]});
        }else{
          tmp_element = at::cat({tmp_element, src[*it]});
        }
      }
    }
    std::tuple<at::Tensor, at::Tensor> max_ = tmp_element.max(0, false);
    out[i] = std::get<0>(max_);
  }
}

at::Tensor scatter_max(at::Tensor src, at::Tensor edge_indexes, at::Tensor out, int64_t dim) {
  std::map<V_ID, std::set<V_ID>*> inEdges;
  V_ID nv = out.size(0);
  //std::cout << nv << std::endl;
  V_ID maxNodeIndex = 0;
  E_ID numEdges = 0;
  V_ID maxDepth = 10;
  V_ID maxWidth = 10;
  torch_to_graph(edge_indexes, inEdges, maxNodeIndex, numEdges);

  std::map<V_ID, std::set<V_ID>*> optInEdges;
  std::vector<std::pair<V_ID, V_ID> > optRanges;
  V_ID new_max_node_index;
  transfer_graph(inEdges, optInEdges, optRanges, maxNodeIndex, numEdges, maxDepth, maxWidth, new_max_node_index);
  scatter_max_(inEdges, optInEdges, optRanges, nv, src, out, dim, maxWidth);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_max", &scatter_max, "Scatter Max (CPU)");
}
