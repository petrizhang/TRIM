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

#include "unity/common/common.h"
#include "unity/common/object.h"

namespace unity {

template <typename ChildClass>
struct SetterProxy {
  template <typename T>
  using Setter = void (ChildClass::*)(T);
  using ProxyFunc = std::function<void(ChildClass*, const Object&)>;
  using ProxyPair = std::pair<ObjectType, ProxyFunc>;

  std::string _prompt{"error"};
  std::unordered_map<std::string, ProxyPair> _proxy_map;

  SetterProxy() = default;

  explicit SetterProxy(const std::string& prefix) : _prompt(prefix) {}

  virtual ~SetterProxy() = default;

  void proxied_set(const std::string& key, const Object& value) {
    auto it = _proxy_map.find(key);
    if (it != _proxy_map.end()) {
      auto& setter = it->second.second;
      auto expect_type = it->second.first;
      if (expect_type != value.type) {
        U_THROW_FMT("%s error: parameter `%s` must be a %s value, but got %s value [%s]",
                    _prompt.c_str(), key.c_str(), get_type_desc(expect_type),
                    get_type_desc(value.type), value.to_string().c_str());
      }
      setter((ChildClass*)this, value);
      return;
    }

    U_THROW_FMT("%s error: got unknown parameter: `%s`", _prompt.c_str(), key.c_str());
  }

  void try_proxied_set(const std::string& key, const Object& value) {
    auto it = _proxy_map.find(key);
    if (it != _proxy_map.end()) {
      auto& setter = it->second.second;
      auto expect_type = it->second.first;
      if (expect_type == value.type) {
        setter((ChildClass*)this, value);
      }
    }
  }

  template <ObjectType obj_type, typename T>
  SetterProxy& bind(const std::string& name, Setter<T> setter) {
    ProxyFunc proxy = [setter](ChildClass* that, const Object& value) -> void {
      auto typed_value = static_cast<T>(value.get<obj_type>());
      (that->*setter)(typed_value);
    };
    _proxy_map[name] = std::make_pair(obj_type, std::move(proxy));
    return *this;
  }
};

}  // namespace unity