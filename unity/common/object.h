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

#include <any>
#include <cstdint>
#include <unordered_map>

#include "unity/common/u_assert.h"

namespace unity {

enum ObjectType { NULL_TYPE, BOOL_TYPE, INTEGER_TYPE, DOUBLE_TYPE, STRING_TYPE };

struct Object {
  ObjectType type;
  std::any value;

  Object(std::nullptr_t) { this->type = NULL_TYPE; }

  Object() { this->type = NULL_TYPE; }

  Object(bool value) {
    this->type = BOOL_TYPE;
    this->value = value;
  }

  Object(int64_t value) {
    this->type = INTEGER_TYPE;
    this->value = value;
  }

  Object(int32_t value) {
    this->type = INTEGER_TYPE;
    this->value = static_cast<int64_t>(value);
  }

  Object(float value) {
    this->type = DOUBLE_TYPE;
    this->value = static_cast<double>(value);
  }

  Object(double value) {
    this->type = DOUBLE_TYPE;
    this->value = value;
  }

  Object(std::string value) {
    this->type = STRING_TYPE;
    this->value = std::move(value);
  }

  Object(const char* value) {
    this->type = STRING_TYPE;
    this->value = std::string(value);
  }

  [[nodiscard]] bool is_null() const { return type == NULL_TYPE; }

  [[nodiscard]] bool get_bool() const {
    U_ASSERT(type == BOOL_TYPE);
    return std::any_cast<bool>(value);
  }

  [[nodiscard]] int64_t get_int64() const {
    U_ASSERT(type == INTEGER_TYPE);
    return std::any_cast<int64_t>(value);
  }

  [[nodiscard]] double get_double() const {
    U_ASSERT(type == DOUBLE_TYPE);
    return std::any_cast<double>(value);
  }

  [[nodiscard]] std::string get_string() const {
    U_ASSERT(type == STRING_TYPE);
    return std::any_cast<std::string>(value);
  }

  [[nodiscard]] std::string to_string() const {
    switch (this->type) {
      case NULL_TYPE:
        return "null";
      case BOOL_TYPE:
        return get_bool() ? "true" : "false";
      case INTEGER_TYPE:
        return std::to_string(get_int64());
      case DOUBLE_TYPE:
        return std::to_string(get_double());
      case STRING_TYPE:
        return get_string();
      default:
        throw unity::UnityException("unexpected code path");
    }
  }
};

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
        U_THROW_FMT("missing required key `%s`", key.c_str());
      }
      return {};
    }

    const Object& obj = it->second;
    if constexpr (std::is_same_v<T, bool>) {
      U_THROW_IF_NOT_FMT(obj.type == ObjectType::BOOL_TYPE, "`%s` must be a boolean value",
                         key.c_str());
      return obj.get_bool();
    }

    if constexpr (std::is_integral_v<T>) {
      U_THROW_IF_NOT_FMT(obj.type == ObjectType::INTEGER_TYPE, "`%s` must be an integer value",
                         key.c_str());
      return obj.get_int64();
    }

    if constexpr (std::is_floating_point_v<T>) {
      U_THROW_IF_NOT_FMT(obj.type == ObjectType::DOUBLE_TYPE, "`%s` must be a double value",
                         key.c_str());
      return obj.get_double();
    }

    if constexpr (std::is_same_v<T, std::string>) {
      U_THROW_IF_NOT_FMT(obj.type == ObjectType::STRING_TYPE, "`%s` must be a string", key.c_str());
      return obj.get_string();
    }

    U_THROW_MSG("reading unsupported type from Dict");
  }
};

}  // namespace unity
