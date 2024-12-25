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

#include <unordered_map>

#include "top/core/object.h"

namespace top {

struct Dict {
  std::unordered_map<std::string, Object> mapping;

  void put(const std::string& key, const Object& value) { mapping[key] = value; }

  Object get(const std::string& key) {
    auto it = mapping.find(key);
    if (it == mapping.end()) {
      return Object();
    }
    return it->second;
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "{";
    bool first = true;
    for (const auto& pair : mapping) {
      if (!first) {
        ss << ", ";
      }
      if (pair.second.type == ObjectType::STRING_TYPE) {
        ss << "\"" << pair.first << "\":\"" << pair.second.to_string() << "\"";
      } else {
        ss << "\"" << pair.first << "\":" << pair.second.to_string();
      }
      first = false;
    }
    ss << "}";
    return ss.str();
  }
};

}  // namespace top
