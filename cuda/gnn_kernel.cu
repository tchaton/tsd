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

#include <cub/cub.cuh>
#include "gnn_kernel.h"

__global__
void reluBackward(float *gradPtr, const float *input, int n)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    gradPtr[i] = (input[i] > 0.001f) ? gradPtr[i] : 0;
  }
}

__global__
void assign_weights(float *w, int count, float initValue)
{
  CUDA_KERNEL_LOOP(i, count)
  {
    w[i] = initValue;
  }
}

__global__
void norm_coop_kernel(V_ID rowLeft,
                      V_ID rowRight,
                      int hiddenDim,
                      V_ID *inDeg,
                      float *h)
{
  assert(blockDim.x % hiddenDim == 0);
  int vtxPerBlock = blockDim.x / hiddenDim;
  int tidDiv = threadIdx.x / hiddenDim;
  for (V_ID blkRowStart = blockIdx.x * vtxPerBlock + rowLeft;
       blkRowStart <= rowRight;
       blkRowStart += vtxPerBlock * gridDim.x)
    if (blkRowStart + tidDiv <= rowRight)
    {
      h[blkRowStart * hiddenDim + threadIdx.x] /= inDeg[blkRowStart + tidDiv];
    }
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__
void aggre_coop_kernel(V_ID rowLeft,
                       V_ID rowRight,
                       int hiddenDim,
                       const NodeStruct* row_ptrs,
                       const EdgeStruct* col_idxs,
                       const float* old_h,
                       float* new_h)
{
  assert(blockDim.x % hiddenDim == 0);
  int vtxPerBlock = blockDim.x / hiddenDim;
  typedef cub::BlockScan<E_ID, CUDA_NUM_THREADS> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ E_ID blkColStart;
  __shared__ float acc_h[CUDA_NUM_THREADS];
  int tidDiv = threadIdx.x / hiddenDim;
  int tidMod = threadIdx.x % hiddenDim;
  for (V_ID blkRowStart = blockIdx.x * vtxPerBlock + rowLeft;
       blkRowStart <= rowRight;
       blkRowStart += vtxPerBlock * gridDim.x)
  {
    E_ID myNumEdges = 0, scratchOffset, totalNumEdges = 0;
    if (threadIdx.x + blkRowStart <= rowRight && threadIdx.x < vtxPerBlock) {
      V_ID curVtx = threadIdx.x + blkRowStart;
      E_ID startColIdx, endColIdx = row_ptrs[curVtx].index;
      if (curVtx == 0)
        startColIdx = 0;
      else
        startColIdx = row_ptrs[curVtx-1].index;
      myNumEdges = endColIdx - startColIdx;
      if (threadIdx.x == 0)
        blkColStart = startColIdx;
    }
    //if (myNumEdges > 0) printf("tid(%d) myNumEdges(%d)\n", threadIdx.x, myNumEdges);
    acc_h[threadIdx.x] = 0.0f;
    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(myNumEdges, scratchOffset, totalNumEdges);
    E_ID done = 0;
    while (totalNumEdges > 0) {
      if (tidDiv < totalNumEdges) {
        EdgeStruct es = col_idxs[blkColStart + done + tidDiv];
        float val = old_h[es.src * hiddenDim + tidMod];
        int offset = (es.dst - blkRowStart) * hiddenDim + tidMod;
        atomicAdd(&acc_h[offset], val);
      }
      done += vtxPerBlock;
      totalNumEdges -= (totalNumEdges > vtxPerBlock) ? vtxPerBlock : totalNumEdges;
    }
    __syncthreads();
    if (tidDiv + blkRowStart <= rowRight)
      new_h[blkRowStart * hiddenDim + threadIdx.x] = acc_h[threadIdx.x];
  }
}

GNNLayer::GNNLayer(GNNModel* _model,
                   float* _inputPtr, float* _inputGradPtr,
                   int _inputDim, int _hiddenDim,
                   int _outputDim,
                   ActMode _actMode, AggMode _aggMode)
: Layer(_model, _inputPtr, _inputGradPtr),
  inputDim(_inputDim), hiddenDim(_hiddenDim), outputDim(_outputDim),
  actMode(_actMode), aggMode(_aggMode)
{
  switch (model->name) {
    case GNNModel::GCN:
    {
      edgeMLP = false;
      edgeNorm = true;
      selfWeights = false;
      break;
    }
    case GNNModel::GCN_P:
    {
      edgeMLP = true;
      edgeNorm = true;
      selfWeights = false;
      break;
    }
    default:
      assert(false);
  } 
  Handler handle = model->handle;
  int nvSrc = model->depGraph.nvSrc;
  int nvNewSrc = model->depGraph.nvNewSrc;
  int nvDst = model->depGraph.nvDst;
  // Create and init weights
  // denseW [_inputDim x _hiddenDim]
  dense = new AdamParameter(handle, inputDim, hiddenDim);
  // neighW [_hiddenDIm x _outputDim]
  neigh = new AdamParameter(handle, hiddenDim, outputDim);
  // selfW [_inputDim x _outputDim]
  self = new AdamParameter(handle, inputDim, outputDim);
  // initialize tensors
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU,
                                          CUDNN_PROPAGATE_NAN, 0.0));
  checkCUDNN(cudnnCreateTensorDescriptor(&hiddenTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(hiddenTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        nvSrc, hiddenDim, 1, 1));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        nvDst, outputDim, 1, 1));
  // allocate hiddenPtr, aggrePtr, outputPtr
  checkCUDA(cudaMalloc(&hiddenPtr, nvNewSrc * hiddenDim * sizeof(float)));
  checkCUDA(cudaMalloc(&hiddenGradPtr, nvNewSrc * outputDim * sizeof(float)));
  checkCUDA(cudaMalloc(&aggrePtr, nvNewSrc * hiddenDim * sizeof(float)));
  checkCUDA(cudaMalloc(&aggreGradPtr, nvNewSrc * hiddenDim * sizeof(float)));
  checkCUDA(cudaMalloc(&outputPtr, nvDst * outputDim * sizeof(float)));
  checkCUDA(cudaMalloc(&outputGradPtr, nvDst * outputDim * sizeof(float)));
}

void print(std::string prefix, const float* ptr, int num, int stride)
{
  printf("%s:\n", prefix.c_str());
  if (ptr == NULL) {
    printf("NULL\n");
    return;
  }
  float* ptrZC = (float*) malloc(num * sizeof(float));
  checkCUDA(cudaMemcpy(ptrZC, ptr, num * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < num; i++) {
    printf("%.4lf ", ptrZC[i]);
    if ((i + 1) % stride == 0)
      printf("\n");
  }
  free(ptrZC);
}

void GNNLayer::forward(void)
{
  Handler handle = model->handle;
  float alpha = 1.0f, beta = 0.0f;
  Graph graph = model->depGraph;
  assert(graph.nvSrc == graph.nvDst);
  V_ID nv = graph.nvSrc;
  if (edgeMLP) {
    // Compute hiddenPtr
    checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                          hiddenDim, nv, inputDim,
                          &alpha, dense->W, inputDim,
                          inputPtr, inputDim,
                          &beta, hiddenPtr, hiddenDim));
    // relu over hiddenPtr
    checkCUDNN(cudnnActivationForward(handle.dnn, actiDesc,
                                      &alpha, hiddenTensor, hiddenPtr,
                                      &beta, hiddenTensor, hiddenPtr));
  }
  // Compute aggregated vectors for vertices [nvSrc, nvNewSrc)
  // save the results in hiddenPtr
  int blkSize = CUDA_NUM_THREADS / hiddenDim * hiddenDim;
  //printf("graph.ranges.size() = %zu\n", graph.ranges.size());
  for (int i = 0; i < graph.ranges.size(); i++) {
    V_ID myNv = graph.ranges[i].second - graph.ranges[i].first + 1;
    int numBlks = (myNv * hiddenDim + blkSize - 1) / blkSize;
    if (numBlks > BLOCK_SIZE_LIMIT)
      numBlks = BLOCK_SIZE_LIMIT;
    aggre_coop_kernel<<<numBlks, blkSize>>>(
        graph.ranges[i].first, graph.ranges[i].second, hiddenDim,
        graph.inRowPtr, graph.inColIdx, hiddenPtr, hiddenPtr);
  }
  // Compute aggrePtr
  int numBlks = (graph.nvDst * hiddenDim + blkSize - 1) / blkSize;
  if (numBlks > BLOCK_SIZE_LIMIT)
    numBlks = BLOCK_SIZE_LIMIT;
  aggre_coop_kernel<<<numBlks, blkSize>>>(0, graph.nvDst - 1, hiddenDim,
    graph.inRowPtr, graph.inColIdx, hiddenPtr, aggrePtr);
  if (edgeNorm) {
    // Normalize aggreVector by in-degree of vertices
    norm_coop_kernel<<<numBlks, blkSize>>>(0, graph.nvDst - 1, hiddenDim,
      graph.inDeg, aggrePtr);
  }

#ifdef DEADCODE
  // Compute outputPtr
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        outputDim, nv, hiddenDim,
                        &alpha, neigh->W, hiddenDim,
                        aggrePtr, hiddenDim,
                        &beta, outputPtr, outputDim));
  if (selfWeights) {
    // We should add activations on top of the previous Sgemm
    checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                          outputDim, nv, inputDim,
                          &alpha, self->W, inputDim,
                          inputPtr, inputDim,
                          &alpha, outputPtr, outputDim));
  }
  if (actMode == ACT_MODE_RELU) {
    checkCUDNN(cudnnActivationForward(handle.dnn, actiDesc,
                                      &alpha, outputTensor, outputPtr,
                                      &beta, outputTensor, outputPtr));
  } else {
    assert(false);
  }
#endif
  //print("GNNLayer:input", inputPtr, std::min(20, nv) * inputDim , inputDim);
  //print("GNNLayer:selfW", selfWPtr, inputDim * outputDim, inputDim);
  //print("GNNLayer:denseW", denseWPtr, inputDim * hiddenDim, inputDim);
  //print("GNNLayer:hidden", hiddenPtr, std::min(20, graph.nvNewSrc) * hiddenDim, hiddenDim);
  //print("GNNLayer:aggre", aggrePtr, std::min(20, nv) * hiddenDim, hiddenDim);
  //print("GNNLayer:neighW", neighWPtr, hiddenDim * outputDim, hiddenDim);
  //print("GNNLayer:output", outputPtr, std::min(20, nv) * outputDim, outputDim);
}

void GNNLayer::backward(void)
{
  Handler handle = model->handle;
  float alpha = 1.0f, beta = 0.0f;
  Graph graph = model->depGraph;
  assert(graph.nvSrc == graph.nvDst);
  V_ID nv = graph.nvSrc;
#ifdef DEADCODE
  if (actMode == ACT_MODE_RELU) {
    int n = nv * outputDim;
    reluBackward<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
        outputGradPtr, outputPtr, n);
  } else {
    assert(false);
  }
  if (selfWeights) {
    // Compute selfWGrad
    checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                          inputDim, outputDim, nv,
                          &alpha, inputPtr, inputDim,
                          outputGradPtr, outputDim,
                          &beta, self->WGrad, inputDim));
    // Compute inputGradPtr
    if (inputGradPtr != NULL) {
      checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                            inputDim, nv, outputDim,
                            &alpha, self->W, inputDim,
                            outputGradPtr, outputDim,
                            &beta, inputGradPtr, inputDim));
    }
  }
  // Compute neighWGrad
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        hiddenDim, outputDim, nv,
                        &alpha, aggrePtr, hiddenDim,
                        outputGradPtr, outputDim,
                        &beta, neigh->WGrad, hiddenDim));
  // Compute aggreGrad
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        hiddenDim, nv, outputDim,
                        &alpha, neigh->W, hiddenDim,
                        outputGradPtr, outputDim,
                        &beta, aggreGradPtr, hiddenDim));
#endif
  // normalize aggreVector by in-degree of vertices
  int blkSize = CUDA_NUM_THREADS / hiddenDim * hiddenDim;
  int numBlks = (nv * hiddenDim + blkSize - 1) / blkSize;
  if (numBlks > BLOCK_SIZE_LIMIT)
    numBlks = BLOCK_SIZE_LIMIT;
  if (edgeNorm) {
    norm_coop_kernel<<<numBlks, blkSize>>>(0, nv - 1, hiddenDim,
      graph.inDeg, aggrePtr);
  }
  // Compute aggreGrad for vertices [depGraph.nv, depGraph.nvNewSrc)
  // save the results in aggreGradPrt
  for (int i = graph.ranges.size() - 1; i >= 0; i--) {
    V_ID myNv = graph.ranges[i].second - graph.ranges[i].first + 1;
    int myBlks = (myNv * hiddenDim + blkSize - 1) / blkSize;
    if (myBlks > BLOCK_SIZE_LIMIT)
      myBlks = BLOCK_SIZE_LIMIT;
    aggre_coop_kernel<<<myBlks, blkSize>>>(
        graph.ranges[i].first, graph.ranges[i].second, hiddenDim,
        graph.outRowPtr, graph.outColIdx, aggreGradPtr, aggreGradPtr);
  }
  // Compute hiddenGrad
  aggre_coop_kernel<<<numBlks, blkSize>>>(0, nv - 1, hiddenDim,
      graph.outRowPtr, graph.outColIdx, aggreGradPtr, hiddenGradPtr);

  if (edgeMLP) {
    // Backprop relu
    int n = nv * hiddenDim;
    reluBackward<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
        hiddenGradPtr, hiddenPtr, n);
    // Compute denseWGrad
    checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                          inputDim, hiddenDim, nv,
                          &alpha, inputPtr, inputDim,
                          hiddenGradPtr, hiddenDim,
                          &beta, dense->WGrad, inputDim));
    // Compute inputGrad
    if (inputGradPtr != NULL) {
      // Note: this is the second time we compute inputGrad,
      // so we replace beta with alpha
      checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                            inputDim, nv, hiddenDim,
                            &alpha, dense->W, inputDim,
                            hiddenGradPtr, hiddenDim,
                            &alpha/**1.0**/,  inputGradPtr, inputDim));
    }
  }
  //print("GNNLayer:aggreGrad", aggreGradPtr, std::min(20, graph.nvNewSrc) * inputDim, inputDim);
  //print("GNNLayer:hiddenGrad", hiddenGradPtr, std::min(20, nv) * inputDim, inputDim);
  //print("GNNLayer:denseWGrad", dense->WGrad, hiddenDim * inputDim, inputDim);
}

void GNNLayer::update(AdamOpt adam)
{
  if (edgeMLP) {
    // Update dense
    dense->update(adam);
  }
  // Update neighW
  neigh->update(adam);
  if (selfWeights) {
    // Update selfW
    self->update(adam);
  }
}

GCLayer::GCLayer(GNNModel* _model,
                 float* _inputPtr, float* _inputGradPtr,
                 int _inputDim, int _numClass)
: Layer(_model, _inputPtr, _inputGradPtr),
  inputDim(_inputDim), numClass(_numClass)
{
  Handler handle = model->handle;
  int ng = model->hyGraph.nvDst;
  // Create and init weights
  // denseW [_inputDim x _numClass]
  dense = new AdamParameter(handle, inputDim, numClass);
  // Allocate aggregate and output tensors
  checkCUDA(cudaMalloc(&aggrePtr, ng * inputDim * sizeof(float)));
  checkCUDA(cudaMalloc(&outputPtr, ng * numClass * sizeof(float)));
  checkCUDA(cudaMalloc(&aggreGradPtr, ng * inputDim * sizeof(float)));
  checkCUDA(cudaMalloc(&outputGradPtr, ng * numClass * sizeof(float)));
}

void GCLayer::forward(void)
{
  Handler handle = model->handle;
  float alpha = 1.0f, beta = 0.0f;
  Graph graph = model->hyGraph;
  // Compute aggrePtr
  int blkSize = CUDA_NUM_THREADS / inputDim * inputDim;
  int numBlks = (graph.nvDst * inputDim + blkSize - 1) / blkSize;
  if (numBlks > BLOCK_SIZE_LIMIT)
    numBlks = BLOCK_SIZE_LIMIT;
  aggre_coop_kernel<<<numBlks, blkSize>>>(0, graph.nvDst-1, inputDim,
      graph.inRowPtr, graph.inColIdx, inputPtr, aggrePtr);
  
  // TODO: normalize graph vector by degrees

  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        numClass, graph.nvDst, inputDim,
                        &alpha, dense->W, inputDim,
                        aggrePtr, inputDim,
                        &beta, outputPtr, numClass)); 
}

void GCLayer::backward(void)
{
  Handler handle = model->handle;
  float alpha = 1.0f, beta = 0.0f;
  Graph graph = model->hyGraph;
  // Compute denseW grad
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        inputDim, numClass, graph.nvDst,
                        &alpha, aggrePtr, inputDim,
                        outputGradPtr, numClass,
                        &beta, dense->WGrad, inputDim));
  // Compute aggreGrad
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        inputDim, graph.nvDst, numClass,
                        &alpha, dense->W, inputDim,
                        outputGradPtr, numClass,
                        &beta, aggreGradPtr, inputDim));
  // TODO: normalize graph vector by degrees

  int blkSize = CUDA_NUM_THREADS / inputDim * inputDim;
  int numBlks = (graph.nvSrc * inputDim + blkSize - 1) / blkSize;
  if (numBlks > BLOCK_SIZE_LIMIT)
    numBlks = BLOCK_SIZE_LIMIT;
  aggre_coop_kernel<<<numBlks, blkSize>>>(0, graph.nvSrc, inputDim,
      graph.outRowPtr, graph.outColIdx, aggreGradPtr, inputGradPtr);
}

void GCLayer::update(AdamOpt adam)
{
  // Update denseW
  dense->update(adam);
}

NCLayer::NCLayer(GNNModel* _model,
                 float* _inputPtr, float* _inputGradPtr,
                 int _inputDim, int _numClass)
: Layer(_model, _inputPtr, _inputGradPtr),
  inputDim(_inputDim), numClass(_numClass)
{
  Handler handle = model->handle;
  int nvDst = model->depGraph.nvDst;
  // Create and init weights
  // denseW [_inputDim x _numClass]
  dense = new AdamParameter(handle, inputDim, numClass);
  // Allocate output tensors
  checkCUDA(cudaMalloc(&outputPtr, nvDst * numClass * sizeof(float)));
  checkCUDA(cudaMalloc(&outputGradPtr, nvDst * numClass * sizeof(float)));
}

void NCLayer::forward(void)
{
  Handler handle = model->handle;
  float alpha = 1.0f, beta = 0.0f;
  int nv = model->depGraph.nvDst;
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        numClass, nv, inputDim,
                        &alpha, dense->W, inputDim,
                        inputPtr, inputDim,
                        &beta, outputPtr, numClass));
  //print("NCLayer::outputPtr", outputPtr, std::min(20, nv) * numClass, numClass);
}

void NCLayer::backward(void)
{
  Handler handle = model->handle;
  float alpha = 1.0f, beta = 0.0f;
  int nv = model->depGraph.nvDst;
  // Compute denseW grad
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        inputDim, numClass, nv,
                        &alpha, inputPtr, inputDim,
                        outputGradPtr, numClass,
                        &beta, dense->WGrad, inputDim));
  // Compute inputGrad
  if (inputGradPtr != NULL) {
    checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                          inputDim, nv, numClass,
                          &alpha, dense->W, inputDim,
                          outputGradPtr, numClass,
                          &beta, inputGradPtr, inputDim));
  }
  //print("NCLayer::denseWGrad", denseWGradPtr, inputDim * numClass, inputDim);
  //print("NCLayer::inputGrad", inputGradPtr, nv * inputDim, inputDim);
}

void NCLayer::update(AdamOpt adam)
{
  // Update denseW
  dense->update(adam);
}

SMLayer::SMLayer(GNNModel* _model,
                 float* _inputPtr, float* _inputGradPtr,
                 int _numSamples, int _numClass)
: Layer(_model, _inputPtr, _inputGradPtr),
  numSamples(_numSamples), numClass(_numClass)
{
  // Create output tensors
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        numSamples, numClass, 1, 1));
  // Allocate output tensors
  checkCUDA(cudaMalloc(&outputPtr, numSamples * numClass * sizeof(float)));
}

void SMLayer::forward(void)
{
  Handler handle = model->handle;
  float alpha = 1.0f, beta = 0.0f;
  
  checkCUDNN(cudnnSoftmaxForward(handle.dnn, CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha, outputTensor, inputPtr,
                                 &beta, outputTensor, outputPtr));
  //print("SMLayer::outputPtr", outputPtr, 4 * numClass, numClass);
}

__global__
void softmax_backward(float* inputGrad, const int *label,
                      int numClass, int numSamples)
{
  CUDA_KERNEL_LOOP(i, numSamples)
  {
    int labelIdx = label[i];
    inputGrad[i * numClass + labelIdx] -= 1.0f;
  }
}

__global__
void calc_loss(const float* output, const int *label,
               int numClass, int numSamples, float* loss)
{
  CUDA_KERNEL_LOOP(i, numSamples)
  {
    int labelIdx = label[i];
    atomicAdd(loss, 1 - output[i * numClass + labelIdx]);
  }
}

void SMLayer::backward(void)
{
  checkCUDA(cudaMemcpyAsync(inputGradPtr, outputPtr,
                            numSamples * numClass * sizeof(float),
                            cudaMemcpyDeviceToDevice));
  softmax_backward<<<GET_BLOCKS(numSamples), CUDA_NUM_THREADS>>>(
      inputGradPtr, model->labels, numClass, numSamples);
  return;
  //print("SMLayer::inputGradPtr", inputGradPtr, numSamples * numClass, numClass);
  // TODO: remote following code
  float* loss;
  float lossZC;
  checkCUDA(cudaMalloc(&loss, sizeof(float)));
  checkCUDA(cudaMemset(loss, 0, sizeof(float)));
  calc_loss<<<GET_BLOCKS(numSamples), CUDA_NUM_THREADS>>>(
      outputPtr, model->labels, numClass, numSamples, loss);
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaMemcpy(&lossZC, loss, sizeof(float),
                       cudaMemcpyDeviceToHost));
  printf("Loss = %.4lf\n", lossZC);
  cudaFree(loss);
}

void SMLayer::update(AdamOpt adam)
{
  // Do nothing...
}

Handler::Handler(void)
{
  checkCUDA(cublasCreate(&blas));
  checkCUDNN(cudnnCreate(&dnn));
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
}

AdamParameter::AdamParameter(Handler handle, int inputDim, int outputDim)
: count(inputDim * outputDim)
{
  checkCUDA(cudaMalloc(&W, count * sizeof(float)));
  checkCUDA(cudaMalloc(&WGrad, count * sizeof(float)));
  checkCUDA(cudaMalloc(&M, count * sizeof(float)));
  checkCUDA(cudaMalloc(&V, count * sizeof(float)));
  float scale = sqrt(6.0 / (inputDim + outputDim));
  init_weights(W, count, scale, handle.gen);
  assign_weights<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(M, count, 0.0f);
  assign_weights<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(V, count, 0.0f);
}

__global__
void adam_update(int count, float alpha_t,
                 float beta1, float beta2, float epsilon,
                 const float *WGrad, float *M, float *V, float *W)
{
  CUDA_KERNEL_LOOP(i, count)
  {
    float gt = WGrad[i];
    float mt = beta1 * M[i] + (1 - beta1) * gt;
    float vt = beta2 * V[i] + (1 - beta2) * gt * gt;
    M[i] = mt;
    V[i] = vt;
    W[i] -= alpha_t * mt / (sqrt(vt) + epsilon);
  }
}

void AdamParameter::update(AdamOpt adam)
{
  //printf("alpha_t(%.4lf) beta1(%.4lf) beta2(%.4lf) beta1_t(%.4lf) beta2_t(%.4lf)\n",
  //       adam.alpha_t, adam.beta1, adam.beta2, adam.beta1_t, adam.beta2_t);
  adam_update<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
      count, adam.alpha_t, adam.beta1, adam.beta2, adam.epsilon,
      WGrad, M, V, W);
}

__global__
void scale_kernel(float* ptr, int size, float a, float b)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = (b - a) * ptr[i] + a;
  }
}

__global__
void seq_kernel(float* ptr, int size)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = 1;
  }
}

void init_weights(float* ptr, int num, float scale, curandGenerator_t genGPU)
{
  curandGenerateUniform(genGPU, ptr, num);
  scale_kernel<<<GET_BLOCKS(num), CUDA_NUM_THREADS>>>(
      ptr, num, -scale, scale);
}

void seq_weights(float* ptr, int num)
{
  seq_kernel<<<GET_BLOCKS(num), CUDA_NUM_THREADS>>>(ptr, num);
}
