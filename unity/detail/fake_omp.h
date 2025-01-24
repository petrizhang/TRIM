/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/** Faiss refers to some OpenMP functions, let it happy. */

#pragma once

#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif

void omp_set_num_threads(int) {}
int omp_get_num_threads(void) { return 1; }
int omp_get_max_threads(void) { return 1; }
int omp_get_thread_num(void) { return 0; }

int omp_in_parallel(void) { return 0; }

void omp_set_nested(int) {}
int omp_get_nested(void) { return 0; }

void omp_init_lock(omp_lock_t*) {}
void omp_destroy_lock(omp_lock_t*) {}
void omp_set_lock(omp_lock_t*) {}
void omp_unset_lock(omp_lock_t*) {}

#ifdef __cplusplus
}
#endif