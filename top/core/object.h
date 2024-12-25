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

#include "top/faiss/FaissAssert.h"

namespace top {

enum ObjectType { NULL_TYPE, BOOL_TYPE, INTEGER_TYPE, DOUBLE_TYPE, STRING_TYPE };

struct Object {
  ObjectType type;
  std::any value;

  explicit Object() { this->type = NULL_TYPE; }

  explicit Object(bool value) {
    this->type = BOOL_TYPE;
    this->value = value;
  }

  explicit Object(int64_t value) {
    this->type = INTEGER_TYPE;
    this->value = value;
  }

  explicit Object(int32_t value) {
    this->type = INTEGER_TYPE;
    this->value = static_cast<int64_t>(value);
  }

  explicit Object(double value) {
    this->type = DOUBLE_TYPE;
    this->value = value;
  }

  explicit Object(std::string value) {
    this->type = STRING_TYPE;
    this->value = std::move(value);
  }

  explicit Object(const char* value) {
    this->type = STRING_TYPE;
    this->value = std::string(value);
  }

  [[nodiscard]] bool is_null() const { return type == NULL_TYPE; }

  [[nodiscard]] bool get_bool() const {
    FAISS_ASSERT(type == BOOL_TYPE);
    return std::any_cast<bool>(value);
  }

  [[nodiscard]] int64_t get_integer() const {
    FAISS_ASSERT(type == INTEGER_TYPE);
    return std::any_cast<int64_t>(value);
  }

  [[nodiscard]] double get_double() const {
    FAISS_ASSERT(type == DOUBLE_TYPE);
    return std::any_cast<double>(value);
  }

  [[nodiscard]] std::string get_string() const {
    FAISS_ASSERT(type == STRING_TYPE);
    return std::any_cast<std::string>(value);
  }

  [[nodiscard]] std::string to_string() const {
    switch (this->type) {
      case NULL_TYPE:
        return "null";
      case BOOL_TYPE:
        return get_bool() ? "true" : "false";
      case INTEGER_TYPE:
        return std::to_string(get_integer());
      case DOUBLE_TYPE:
        return std::to_string(get_double());
      case STRING_TYPE:
        return get_string();
      default:
        throw faiss::FaissException("unexpected code path");
    }
  }
};

}  // namespace top
