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

#include <exception>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef __GNUG__
#include <cxxabi.h>
#endif

namespace unity {

/// Base class for UNITY exceptions
class UnityException : public std::exception {
 public:
  explicit UnityException(const std::string& msg);

  UnityException(const std::string& msg, const char* funcName, const char* file, int line);

  /// from std::exception
  const char* what() const noexcept override;

  std::string msg;
};

/// Handle multiple exceptions from worker threads, throwing an appropriate
/// exception that aggregates the information
/// The pair int is the thread that generated the exception
void unity_handle_exceptions(std::vector<std::pair<int, std::exception_ptr>>& exceptions);

/// make typeids more readable
std::string unity_demangle_cpp_symbol(const char* name);

}  // namespace unity

namespace unity {

UnityException::UnityException(const std::string& m) : msg(m) {}

UnityException::UnityException(const std::string& m, const char* funcName, const char* file,
                               int line) {
  int size = snprintf(nullptr, 0, "Error in %s at %s:%d: %s", funcName, file, line, m.c_str());
  msg.resize(size + 1);
  snprintf(&msg[0], msg.size(), "Error in %s at %s:%d: %s", funcName, file, line, m.c_str());
}

const char* UnityException::what() const noexcept { return msg.c_str(); }

void unity_handle_exceptions(std::vector<std::pair<int, std::exception_ptr>>& exceptions) {
  if (exceptions.size() == 1) {
    // throw the single received exception directly
    std::rethrow_exception(exceptions.front().second);

  } else if (exceptions.size() > 1) {
    // multiple exceptions; aggregate them and return a single exception
    std::stringstream ss;

    for (auto& p : exceptions) {
      try {
        std::rethrow_exception(p.second);
      } catch (std::exception& ex) {
        if (ex.what()) {
          // exception message available
          ss << "Exception thrown from index " << p.first << ": " << ex.what() << "\n";
        } else {
          // No message available
          ss << "Unknown exception thrown from index " << p.first << "\n";
        }
      } catch (...) {
        ss << "Unknown exception thrown from index " << p.first << "\n";
      }
    }

    throw UnityException(ss.str());
  }
}

// From
// https://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname

std::string unity_demangle_cpp_symbol(const char* name) {
#ifdef __GNUG__
  int status = -1;
  const char* res = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  std::string sres;
  if (status == 0) {
    sres = res;
  }
  free((void*)res);
  return sres;
#else
  // don't know how to do this on other platforms
  return std::string(name);
#endif
}

}  // namespace unity
