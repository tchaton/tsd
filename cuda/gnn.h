/* Copyright 2019 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _GNN_H_
#define _GNN_H_
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <string.h>
#include <map>
#include <set>
#include <vector>

//=====================================================================
// CUDA Helper Functions
//=====================================================================
#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCUDA(status) do {                                         \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int BLOCK_SIZE_LIMIT = 32768;

// CUDA: number of blocks for threads
inline int GET_BLOCKS(const int N)
{
  int ret = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  return (ret > BLOCK_SIZE_LIMIT) ? BLOCK_SIZE_LIMIT : ret;
}

//====================================================================
// GNN Header Definitions
//====================================================================

typedef int V_ID;
typedef int E_ID;
#define HIDDEN_SIZE 256
#define NUM_LAYERS 2

struct NodeStruct {
  E_ID index;
};

struct EdgeStruct {
  V_ID src, dst;
};

enum ActMode {
  ACT_MODE_RELU,
};

enum AggMode {
  AGG_MODE_MEAN_POOLING,
};

struct Handler {
  Handler(void);
  cudnnHandle_t dnn;
  cublasHandle_t blas;
  curandGenerator_t gen;
};

struct Graph {
  Graph(void): nvSrc(0), nvNewSrc(0), nvDst(0), ne(0),
               inRowPtr(NULL), outRowPtr(NULL),
               inColIdx(NULL), outColIdx(NULL),
               inDeg(NULL) {};
  V_ID nvSrc, nvNewSrc, nvDst;
  E_ID ne;
  NodeStruct *inRowPtr, *outRowPtr;
  EdgeStruct *inColIdx, *outColIdx;
  V_ID *inDeg;
  std::vector<std::pair<V_ID, V_ID> > ranges;
};

struct AdamOpt {
  AdamOpt(void): alpha(0.001), beta1(0.9), beta2(0.999), epsilon(1e-5), beta1_t(1), beta2_t(1) {}
  void next_epoch(void);
  double alpha, beta1, beta2, epsilon;
  double alpha_t, beta1_t, beta2_t;
};

struct AdamParameter {
  AdamParameter(Handler handle, int inputDim, int outputDim);
  void update(AdamOpt);
public:
  int count;
  float *W, *WGrad, *M, *V;
};

class GNNModel {
public:
  enum Name {
    GCN,
    GCN_P,
    GraphSAGE_LSTM
  };
  GNNModel(Name name, Handler handle);
  void set_graph(Graph& graph, V_ID nvSrc, V_ID nvNewSrc, V_ID nvDst,
                 std::map<V_ID, std::set<V_ID>* >& inEdges,
                 std::vector<std::pair<V_ID, V_ID> >& ranges);
  void set_dep_graph(V_ID nvSrc, V_ID nvNewSrc, V_ID nvDst,
                    std::map<V_ID, std::set<V_ID>* >& edgeList,
                    std::vector<std::pair<V_ID, V_ID> >& ranges);
  void set_hyper_graph(int nvSrc, int nvDst,
                          std::map<V_ID, std::set<V_ID>* >& edgeList);
  void load_node_label(int nv, std::string filename);
public:
  Name name;
  Handler handle;
  Graph depGraph, hyGraph;
  int* labels;
  int numClass;
};

class Layer {
public:
  Layer(GNNModel* model, float* inputPtr, float *inputGradPtr);
  virtual void forward(void) = 0;
  virtual void backward(void) = 0;
  virtual void update(AdamOpt adam) = 0;
  float *outputPtr, *outputGradPtr;
protected:
  GNNModel* model;
  float *inputPtr, *inputGradPtr;
};

// hiddenPtr [graph->nv x hiddenDim]
// aggrePtr [graph->nv x hiddenDim]
// outputPtr [graph->nv x outputDim]
class GNNLayer : public Layer {
public:
  GNNLayer(GNNModel* model,
           float* inputPtr, float* inputGradPtr,
           int inputDim, int hiddenDim, int outputDim,
           ActMode act, AggMode agg);
  void forward(void);
  void backward(void);
  void update(AdamOpt adam);
private:
  int inputDim, hiddenDim, outputDim;
  AdamParameter *dense, *neigh, *self;
  ActMode actMode;
  AggMode aggMode;
  cudnnActivationDescriptor_t actiDesc;
  cudnnTensorDescriptor_t hiddenTensor, outputTensor;
  float *hiddenPtr, *hiddenGradPtr;
  float *aggrePtr, *aggreGradPtr;
  bool edgeMLP, edgeNorm, selfWeights;
};

// aggrePtr [graph->ng x inputDim]
// outputPtr [graph->ng x numClass]
class GCLayer : public Layer {
public:
  GCLayer(GNNModel* model,
          float *inputPtr, float* inputGradPtr,
          int inputDim, int numClass);
  void forward(void);
  void backward(void);
  void update(AdamOpt adam);
private:
  int inputDim, numClass;
  AdamParameter *dense;
  float *aggrePtr, *aggreGradPtr;
};

// outputPtr [graph->nv x numClass]
class NCLayer : public Layer {
public:
  NCLayer(GNNModel* model,
          float *inputPtr, float *inputGradPtr,
          int inputDim, int numClass);
  void forward(void);
  void backward(void);
  void update(AdamOpt adam);
private:
  int inputDim, numClass;
  AdamParameter *dense;
  float *denseWPtr, *denseWGradPtr;
};

class SMLayer : public Layer {
public:
  SMLayer(GNNModel* model,
          float *inputPtr, float *inputGradPtr,
          int numSamples, int numClass);
  void forward(void);
  void backward(void);
  void update(AdamOpt adam);
private:
  int numSamples, numClass;
  cudnnTensorDescriptor_t outputTensor; 
};

void transfer_graph(std::map<V_ID, std::set<V_ID>*>& orgList,
                    std::map<V_ID, std::set<V_ID>*>& optList,
                    std::vector<std::pair<V_ID, V_ID> >& ranges,
                    V_ID nv, E_ID ne, V_ID maxDepth, V_ID width, V_ID& newNv);

void transfer_graph_reddit(std::map<V_ID, std::set<V_ID>*>& orgList,
                           std::map<V_ID, std::set<V_ID>*>& optList,
                           std::vector<std::pair<V_ID, V_ID> >& ranges,
                           V_ID nv, E_ID ne, V_ID maxDepth, V_ID width, V_ID& newNv);

#endif
