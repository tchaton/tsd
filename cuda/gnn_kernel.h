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

#ifndef __GNN_KERNEL_H__
#define __GNN_KERNEL_H__
#include "gnn.h"

void init_weights(float* ptr, int num, float scale, curandGenerator_t gen);

void seq_weights(float* ptr, int num);

__global__
void block_coop_kernel(V_ID rowLeft, V_ID rowRight,
                       const NodeStruct* row_ptrs,
                       const EdgeStruct* col_idxs,
                       float* old_h, float* new_h);
#endif
