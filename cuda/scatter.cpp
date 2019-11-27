#include <torch/extension.h>
#include "gnn.h"
#include "gnn_kernel.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

void scatter_mul_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      int64_t dim);
void scatter_mul_cuda_opti(at::Tensor src, at::Tensor index, at::Tensor out,
                      int64_t dim);
void scatter_div_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      int64_t dim);
void scatter_max_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      at::Tensor arg, int64_t dim);
void scatter_min_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      at::Tensor arg, int64_t dim);
void index_backward_cuda(at::Tensor grad, at::Tensor index, at::Tensor arg,
                         at::Tensor out, int64_t dim);

void scatter_mul(at::Tensor src, at::Tensor index, at::Tensor out,
                 int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  scatter_mul_cuda(src, index, out, dim);
}

void scatter_mul_opti(at::Tensor src, at::Tensor index, at::Tensor out,
                 int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  // scatter_mul_cuda_opti(src, index, out, dim);

  std::string graphFile, hyGraphFile, nodeLabelFile = "", graphLabelFile = "";
  std::string graphIndFile;
  double learningRate = 0.001f;
  int epochs = 100;
  V_ID maxDepth = 10;
  V_ID maxWidth = 0;
  // parse_input_args(argv, argc, graphFile, hyGraphFile,
  //                  nodeLabelFile, graphLabelFile, graphIndFile,
  //                  maxDepth, maxWidth, learningRate, epochs);
  printf("maxDepth = %d maxWidth = %d\n", maxDepth, maxWidth);
  Handler handle;
  //FILE* file = fopen("BZR_MD/BZR_MD_A.txt", "r");
  FILE* file = fopen(graphFile.c_str(), "r");
  V_ID u, v;
  V_ID nv = 0;
  std::map<V_ID, std::set<V_ID>* > inEdges;
  E_ID ne = 0;
  while (fscanf(file, "%d, %d", &u, &v) != EOF) {
    // shift node indices by 1 to make them 0-indexed
    u --;
    // shift node indices by 1 to make them 0-indexed
    v --;
    ne ++;
    if (std::max(u, v) >= nv)
      nv = std::max(u, v) + 1;
    // add inEdge
    if (inEdges.find(v) == inEdges.end())
      inEdges[v] = new std::set<V_ID>();
    inEdges[v]->insert(u);
  }

  //int cnt = 0;
  //for (v = 0; v < nv; v++)
  //  if (inEdges.find(v) != inEdges.end()) {
  //    printf("v = %d inEdges[v] = %zu\n", v, inEdges[v]->size());
  //    cnt += inEdges[v]->size() * inEdges[v]->size();
  //  }
  //printf("cnt = %d\n", cnt);
  fclose(file);
  if (graphIndFile.length() > 0) {
    file = fopen(graphIndFile.c_str(), "r");
    std::vector<V_ID> graphIdx;
    while (fscanf(file, "%d", &u) != EOF) {
      graphIdx.push_back(u);
    }
    fclose(file);
  }
 
  float* inputZC = (float*) malloc(nv * HIDDEN_SIZE * sizeof(float));
  memset(inputZC, 0, nv * HIDDEN_SIZE * sizeof(float));
  for (v = 0; v < nv; v++)
    if (inEdges.find(v) != inEdges.end())
      inputZC[v * HIDDEN_SIZE + inEdges[v]->size() % HIDDEN_SIZE] = 1.0f;
    else
      inputZC[v * HIDDEN_SIZE] = 1.0f;
    
  float* inputFB;
  checkCUDA(cudaMalloc(&inputFB, nv * HIDDEN_SIZE * sizeof(float)));
  checkCUDA(cudaMemcpy(inputFB, inputZC, nv * HIDDEN_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice));

  // Optimize Computation Graph
  std::map<V_ID, std::set<V_ID>*> optInEdges;
  std::vector<std::pair<V_ID, V_ID> > optRanges;
  V_ID newNv;

  GNNModel model(GNNModel::GCN, handle);
  model.set_dep_graph(nv, newNv, nv, optInEdges, optRanges);
  //std::vector<std::pair<V_ID, V_ID> > ranges;
  //model.set_dep_graph(nv, nv, nv, inEdges, ranges);
  model.load_node_label(nv, nodeLabelFile);

  // Init adam optimizer
  AdamOpt adam;
  adam.alpha = learningRate;

  std::vector<Layer*> layers;
  for (int i = 0; i < NUM_LAYERS; i++) {
    float* inputPtr = (i == 0) ? inputFB : layers[i-1]->outputPtr;
    float* inputGradPtr = (i == 0) ? NULL : layers[i-1]->outputGradPtr;
    layers.push_back(new GNNLayer(&model, inputPtr, inputGradPtr,
                             HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE,
                             ACT_MODE_RELU, AGG_MODE_MEAN_POOLING));
  }
  //layers.push_back(new NCLayer(&model, inputFB, NULL, HIDDEN_SIZE, model.numClass));
  layers.push_back(new NCLayer(&model, layers[layers.size() - 1]->outputPtr,
                               layers[layers.size() - 1]->outputGradPtr,
                               HIDDEN_SIZE, model.numClass));
  layers.push_back(new SMLayer(&model, layers[layers.size() - 1]->outputPtr,
                               layers[layers.size() - 1]->outputGradPtr,
                               nv, model.numClass));

  cudaEvent_t startEvent, endEvent;
  checkCUDA(cudaEventCreate(&startEvent));
  checkCUDA(cudaEventCreate(&endEvent));
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int iter = 0; iter < epochs; iter ++) {
    adam.next_epoch();
    for (int i = 0; i < layers.size(); i++) {
      layers[i]->forward();
    }
    for (int i = layers.size() - 1; i >= 0; i--) {
      layers[i]->backward();
      layers[i]->update(adam);
    }
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  printf("EXECUTION TIME = %.4lfms\n", milliseconds);
}

void scatter_div(at::Tensor src, at::Tensor index, at::Tensor out,
                 int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  scatter_div_cuda(src, index, out, dim);
}

void scatter_max(at::Tensor src, at::Tensor index, at::Tensor out,
                 at::Tensor arg, int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  CHECK_CUDA(arg);
  scatter_max_cuda(src, index, out, arg, dim);
}

void scatter_min(at::Tensor src, at::Tensor index, at::Tensor out,
                 at::Tensor arg, int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  CHECK_CUDA(arg);
  scatter_min_cuda(src, index, out, arg, dim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_mul_opti", &scatter_mul_opti, "Scatter Mul Opti (CUDA)");
  m.def("scatter_mul", &scatter_mul, "Scatter Mul (CUDA)");
  m.def("scatter_div", &scatter_div, "Scatter Div (CUDA)");
  m.def("scatter_max", &scatter_max, "Scatter Max (CUDA)");
  m.def("scatter_min", &scatter_min, "Scatter Min (CUDA)");
}

GNNModel::GNNModel(Name _name, Handler _handle)
: name(_name), handle(_handle) {}

void GNNModel::set_graph(Graph& graph, V_ID nvSrc, V_ID nvNewSrc, V_ID nvDst,
                         std::map<V_ID, std::set<V_ID>* >& inEdges,
                         std::vector<std::pair<V_ID, V_ID> >& ranges)
{
  graph.nvSrc = nvSrc;
  graph.nvNewSrc = nvNewSrc;
  graph.nvDst = nvDst;
  graph.ne = 0;
  graph.ranges = ranges;
  std::map<V_ID, std::set<V_ID>* > outEdges;
  for (V_ID v = 0; v < nvNewSrc; v++)
    if (inEdges.find(v) != inEdges.end())
      graph.ne += inEdges[v]->size();
  NodeStruct *rowPtrZC, *inRowPtrFB, *outRowPtrFB;
  EdgeStruct *colIdxZC, *inColIdxFB, *outColIdxFB;
  V_ID *inDegZC, *inDegFB;
  rowPtrZC = (NodeStruct*) malloc(graph.nvNewSrc * sizeof(NodeStruct));
  colIdxZC = (EdgeStruct*) malloc(graph.ne * sizeof(EdgeStruct));
  inDegZC = (V_ID*) malloc(graph.nvNewSrc * sizeof(V_ID));
  // Step 1: compute in-degree
  for (V_ID v = nvSrc; v < nvNewSrc; v++) {
    inDegZC[v] = 0;
    assert(inEdges.find(v) != inEdges.end());
    std::set<V_ID>::const_iterator it;
    for (it = inEdges[v]->begin(); it != inEdges[v]->end(); it++) {
      inDegZC[v] += *it < nvSrc ? 1 : inDegZC[*it];
    }
  }
  for (V_ID v = 0; v < nvSrc; v++) {
    inDegZC[v] = 0;
    if (inEdges.find(v) != inEdges.end()) {
      std::set<V_ID>::const_iterator first = inEdges[v]->begin();
      std::set<V_ID>::const_iterator last = inEdges[v]->end();
      std::set<V_ID>::const_iterator it = first;
      for (it = first; it != last; it++)
        inDegZC[v] += *it < nvSrc ? 1 : inDegZC[*it];
    }
  }
  // Step 2: construct in edges;
  E_ID count = 0;
  for (V_ID v = 0; v < nvNewSrc; v++) {
    if (inEdges.find(v) != inEdges.end()) {
      std::set<V_ID>::const_iterator first = inEdges[v]->begin();
      std::set<V_ID>::const_iterator last = inEdges[v]->end();
      std::set<V_ID>::const_iterator it = first;
      for (it = first; it != last; it++) {
        colIdxZC[count].src = *it;
        colIdxZC[count].dst = v;
        count ++;
        if (outEdges.find(*it) == outEdges.end())
          outEdges[*it] = new std::set<V_ID>();
        outEdges[*it]->insert(v);
      }
    }
    rowPtrZC[v].index = count;
  }
  checkCUDA(cudaMalloc(&inRowPtrFB, graph.nvNewSrc * sizeof(NodeStruct)));
  checkCUDA(cudaMalloc(&inColIdxFB, graph.ne * sizeof(EdgeStruct)));
  checkCUDA(cudaMalloc(&inDegFB, graph.nvNewSrc * sizeof(V_ID)));
  checkCUDA(cudaMemcpy(inRowPtrFB, rowPtrZC, graph.nvNewSrc * sizeof(NodeStruct),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(inColIdxFB, colIdxZC, graph.ne * sizeof(EdgeStruct),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(inDegFB, inDegZC, graph.nvNewSrc * sizeof(V_ID),
                       cudaMemcpyHostToDevice));
  graph.inRowPtr = inRowPtrFB;
  graph.inColIdx = inColIdxFB;
  graph.inDeg = inDegFB;
  // Step 3: construct out edges
  count = 0;
  for (V_ID v = 0; v < nvNewSrc; v++) {
    if (outEdges.find(v) != outEdges.end()) {
      std::set<V_ID>::const_iterator first = outEdges[v]->begin();
      std::set<V_ID>::const_iterator last = outEdges[v]->end();
      std::set<V_ID>::const_iterator it = first;
      for (it = first; it != last; it++) {
        colIdxZC[count].src = *it;
        colIdxZC[count].dst = v;
        count ++;
      }
    }
    rowPtrZC[v].index = count;
  }
  checkCUDA(cudaMalloc(&outRowPtrFB, graph.nvNewSrc * sizeof(NodeStruct)));
  checkCUDA(cudaMalloc(&outColIdxFB, graph.ne * sizeof(EdgeStruct)));
  checkCUDA(cudaMemcpy(outRowPtrFB, rowPtrZC, graph.nvNewSrc * sizeof(NodeStruct),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(outColIdxFB, colIdxZC, graph.ne * sizeof(EdgeStruct),
                       cudaMemcpyHostToDevice));
  graph.outRowPtr = outRowPtrFB;
  graph.outColIdx = outColIdxFB;
  // Step 3: free resources
  free(rowPtrZC);
  free(colIdxZC);
  free(inDegZC);
}

void GNNModel::set_dep_graph(V_ID nvSrc, V_ID nvNewSrc, V_ID nvDst,
         std::map<V_ID, std::set<V_ID>* >& edgeList,
         std::vector<std::pair<V_ID, V_ID> >& ranges)
{
  set_graph(depGraph, nvSrc, nvNewSrc, nvDst, edgeList, ranges);
  printf("Add normal in-edge graph: nvSrc(%d) nvDst(%d) ne(%d)\n",
         depGraph.nvSrc, depGraph.nvDst, depGraph.ne);
}

void GNNModel::set_hyper_graph(int nvSrc, int nvDst,
         std::map<V_ID, std::set<V_ID>* >& edgeList)
{
  assert(nvSrc >= nvDst);
  std::vector<std::pair<V_ID, V_ID> > ranges;
  set_graph(hyGraph, nvSrc, nvSrc, nvDst, edgeList, ranges);
  printf("Add hyper in-edge graph: nvSrc(%d) nvDst(%d) ne(%d)\n",
         hyGraph.nvSrc, hyGraph.nvDst, hyGraph.ne);
}

void GNNModel::load_node_label(int nv, std::string filename)
{
  FILE* file = fopen(filename.c_str(), "r");
  int* labelsZC = (int*) malloc(nv * sizeof(int));
  V_ID cnt = 0;
  int u;
  numClass = 0;
  while (fscanf(file, "%d", &u) != EOF) {
    labelsZC[cnt ++] = u;
    if (u >= numClass) numClass = u + 1;
  }
  printf("numClass = %d\n", numClass);
  assert(cnt == nv);
  checkCUDA(cudaMalloc(&labels, nv * sizeof(int)));
  checkCUDA(cudaMemcpy(labels, labelsZC, nv * sizeof(int),
                       cudaMemcpyHostToDevice));
  free(labelsZC);
  fclose(file);
}

Layer::Layer(GNNModel* _model, float* _inputPtr, float* _inputGradPtr)
: model(_model), inputPtr(_inputPtr), inputGradPtr(_inputGradPtr)
{}

void AdamOpt::next_epoch(void)
{
  beta1_t *= beta1;
  beta2_t *= beta2;
  alpha_t = alpha * sqrt(1 - beta2_t) / (1 - beta1_t);
}
