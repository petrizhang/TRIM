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
#pragma once

#include <string>

#include "top/detail/core/common.h"
#include "top/detail/io/read_hnswlib.h"
#include "top/detail/searcher/hnsw_searcher.h"

namespace top {

namespace detail {

#define TOP_GET_REQUIRED_OPTION_TO(type, options, name, variable)          \
  do {                                                                 \
    std::optional<type> name##_opt = options.checked_get<type>(#name); \
    if (!name##_opt.has_value()) {                                     \
      TOP_THROW_FMT("option `%s` is required", #name);                 \
    }                                                                  \
    variable = name##_opt.value();                                     \
  } while (false)

std::unique_ptr<Searcher> build_hnsw_searcher(const Dict& options) {
  std::string path;
  int dim;
  TOP_GET_REQUIRED_OPTION_TO(std::string, options, hnswlib_index_path, path);
  TOP_GET_REQUIRED_OPTION_TO(int, options, dim, dim);
  return nullptr;
}

}  // namespace detail

struct SearcherBuilder {
  virtual ~SearcherBuilder() = default;
  Dict options;
  SearcherBuilder& set(const std::string& name, const Object& value) {
    options.put(name, value);
    return *this;
  }
  std::unique_ptr<Searcher> build() { return nullptr; }
};

}  // namespace top