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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

#include "unity/common/setter_proxy.h"

struct A : unity::SetterProxy<A> {
  A() : SetterProxy("A") {
    bind<unity::INTEGER_TYPE>("a", &A::set_a);
    bind<unity::STRING_TYPE>("s", &A::set_s);
  }

  int _a = 0;
  std::string _s;

  void set_a(int a) { _a = a; }

  void set_s(const std::string& s) { _s = s; }
};

int main() {
  A a;
  a.proxied_set("a", 100);
  a.try_proxied_set("a", true);
  a.proxied_set("s", "string");

  std::cout << a._a << "\n";
  std::cout << a._s << "\n";

  return 0;
}