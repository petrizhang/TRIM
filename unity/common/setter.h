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

#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "unity/common/object.h"

namespace unity {

struct SetterProxy {
  using Setter = std::function<void(const Object&)>;
  using SetterPair = std::pair<ObjectType, Setter>;

  std::string _prompt{"error"};
  std::unordered_map<std::string, SetterPair> _setter_map;

  SetterProxy() = default;

  explicit SetterProxy(const std::string& prefix) : _prompt(prefix) {}

  virtual ~SetterProxy() = default;

  virtual void set(const std::string& key, const Object& value) {
    auto it = _setter_map.find(key);
    if (it != _setter_map.end()) {
      auto& setter = it->second.second;
      auto expect_type = it->second.first;
      if (expect_type != value.type) {
        U_THROW_FMT("%s error: parameter `%s` must be a %s value, but got %s [%s]", _prompt.c_str(),
                    key.c_str(), get_type_desc(expect_type), get_type_desc(value.type),
                    value.to_string().c_str());
      }
      setter(value);
      return;
    }

    U_THROW_FMT("%s error: got unknown parameter: `%s`", _prompt.c_str(), key.c_str());
  }

  virtual void try_set(const std::string& key, const Object& value) {
    auto it = _setter_map.find(key);
    if (it != _setter_map.end()) {
      auto& setter = it->second.second;
      auto expect_type = it->second.first;
      if (expect_type == value.type) {
        setter(value);
      }
    }
  }

  template <ObjectType obj_type, typename T>
  SetterProxy& bind(const std::string& name, T& ref) {
    static_assert(!std::is_reference_v<T>);
    Setter setter = [&ref](const Object& value) -> void {
      ref = static_cast<T>(value.get<obj_type>());
    };
    _setter_map[name] = std::make_pair(obj_type, std::move(setter));
    return *this;
  }

  template <ObjectType obj_type>
  SetterProxy& bind(const std::string& name, const Setter& setter) {
    _setter_map[name] = std::make_pair(obj_type, setter);
    return *this;
  }
};

}  // namespace unity