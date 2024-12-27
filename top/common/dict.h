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

#include <optional>
#include <unordered_map>

#include "top/common/object.h"
#include "top/common/top_assert.h"

namespace top {

struct Dict {
  std::unordered_map<std::string, Object> mapping;

  void put(const std::string& key, const Object& value) { mapping[key] = value; }

  std::optional<Object> get(const std::string& key) const {
    auto it = mapping.find(key);
    if (it == mapping.end()) {
      return {};
    }
    return std::make_optional(it->second);
  }

  template <typename T>
  std::optional<T> checked_get(const std::string& key) const {
    auto it = mapping.find(key);
    if (it == mapping.end()) {
      return {};
    }
    const Object& obj = it->second;
    if constexpr (std::is_same_v<T, bool>) {
      TOP_THROW_IF_NOT_FMT(obj.type == ObjectType::BOOL_TYPE, "%s must be of boolean type",
                           key.c_str());
      return obj.get_bool();
    }

    if constexpr (std::is_integral_v<T>) {
      TOP_THROW_IF_NOT_FMT(obj.type == ObjectType::INTEGER_TYPE, "%s must be of integer type",
                           key.c_str());
      return obj.get_integer();
    }

    if constexpr (std::is_floating_point_v<T>) {
      TOP_THROW_IF_NOT_FMT(obj.type == ObjectType::DOUBLE_TYPE, "%s must be of double type",
                           key.c_str());
      return obj.get_double();
    }

    if constexpr (std::is_same_v<T, std::string>) {
      TOP_THROW_IF_NOT_FMT(obj.type == ObjectType::STRING_TYPE, "%s must be of string type",
                           key.c_str());
      return obj.get_string();
    }

    TOP_THROW_MSG("reading unsupported type from dict");
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
