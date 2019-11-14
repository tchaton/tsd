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
#include <assert.h>
#include <string.h>
#include <map>
#include <set>
#include <vector>

//====================================================================
// GNN Header Definitions
//====================================================================

typedef int V_ID;
typedef int E_ID;

struct EdgeStruct {
  V_ID src, dst;
};

void transfer_graph(std::map<V_ID, std::set<V_ID>*>& orgList,
                    std::map<V_ID, std::set<V_ID>*>& optList,
                    std::vector<std::pair<V_ID, V_ID> >& ranges,
                    V_ID nv, E_ID ne, V_ID maxDepth, V_ID width, V_ID& newNv);

#endif
