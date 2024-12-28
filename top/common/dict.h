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

  Dict& put(const std::string& key, const Object& value) {
    mapping[key] = value;
    return *this;
  }

  std::optional<Object> get(const std::string& key) const {
    auto it = mapping.find(key);
    if (it == mapping.end()) {
      return {};
    }
    return std::make_optional(it->second);
  }

  template <typename T>
  T require(const std::string& key) const {
    return _checked_get<T, true>(key).value();
  }

  template <typename T>
  std::optional<T> optional(const std::string& key) const {
    return _checked_get<T, false>(key);
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

  template <typename T, bool is_required = false>
  std::optional<T> _checked_get(const std::string& key) const {
    static_assert(!std::is_reference_v<T> && !std::is_pointer_v<T>);

    auto it = mapping.find(key);

    if (it == mapping.end()) {
      if constexpr (is_required) {
        TOP_THROW_FMT("missing required key `%s`", key.c_str());
      }
      return {};
    }

    const Object& obj = it->second;
    if constexpr (std::is_same_v<T, bool>) {
      TOP_THROW_IF_NOT_FMT(obj.type == ObjectType::BOOL_TYPE, "`%s` must be a boolean value",
                           key.c_str());
      return obj.get_bool();
    }

    if constexpr (std::is_integral_v<T>) {
      TOP_THROW_IF_NOT_FMT(obj.type == ObjectType::INTEGER_TYPE, "`%s` must be an integer type",
                           key.c_str());
      return obj.get_integer();
    }

    if constexpr (std::is_floating_point_v<T>) {
      TOP_THROW_IF_NOT_FMT(obj.type == ObjectType::DOUBLE_TYPE, "`%s` must be a double value",
                           key.c_str());
      return obj.get_double();
    }

    if constexpr (std::is_same_v<T, std::string>) {
      TOP_THROW_IF_NOT_FMT(obj.type == ObjectType::STRING_TYPE, "`%s` must be a string",
                           key.c_str());
      return obj.get_string();
    }

    TOP_THROW_MSG("reading unsupported type from Dict");
  }
};

}  // namespace top
