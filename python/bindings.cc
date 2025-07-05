#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thread>

#include "trim/trim.h"

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

namespace trim {

Object create_obejct(const py::object& obj) {
  if (obj.is_none()) {
    return Object();
  } else if (py::isinstance<py::bool_>(obj)) {
    return Object(obj.cast<bool>());
  } else if (py::isinstance<py::int_>(obj)) {
    return Object(obj.cast<int64_t>());
  } else if (py::isinstance<py::float_>(obj)) {
    return Object(obj.cast<double>());
  } else if (py::isinstance<py::str>(obj)) {
    return Object(obj.cast<std::string>());
  } else {
    throw std::runtime_error("Unsupported type for conversion to TRIM Object");
  }
}

}  // namespace trim

inline void get_input_array_shapes(const py::buffer_info& buffer, size_t* rows, size_t* features) {
  if (buffer.ndim != 2 && buffer.ndim != 1) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "Input vector data wrong shape. Number of dimensions %ld. Data "
             "must be a 1D or 2D array.",
             buffer.ndim);
  }
  if (buffer.ndim == 2) {
    *rows = buffer.shape[0];
    *features = buffer.shape[1];
  } else {
    *rows = 1;
    *features = buffer.shape[0];
  }
}

struct Searcher {
  std::unique_ptr<trim::ISearcher> searcher;

  Searcher() : searcher(nullptr) {}
  
  Searcher(std::unique_ptr<trim::ISearcher> searcher) : searcher(std::move(searcher)) {};

  void set_data(py::object input) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
    auto buffer = items.request();
    size_t rows, features;
    get_input_array_shapes(buffer, &rows, &features);
    float* vector_data = (float*)items.data(0);
    searcher->set_data(vector_data);
    // T_THROW_IF_NOT_MSG(py::isinstance<py::str>(input), "name must be a string");
    // std::string data_path = input.cast<std::string>();
    // searcher->set_data(data_path.c_str());
  }

  py::object ann_search(py::object query, int k) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(query);
    int* ids = new int[k];
    searcher->ann_search(items.data(0), k, ids);
    py::capsule free_when_done(ids, [](void* f) { delete[] (int*)f; });
    return py::array_t<int>({k}, {sizeof(int)}, ids, free_when_done);
  }

  py::object range_search(py::object query, float radius) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(query);
    std::vector<int> result;
    searcher->range_search(items.data(0), radius, result);
    py::array_t<int> result_array(result.size(), result.data());
    return result_array;
  }

  void set(py::object py_name, py::object py_value) {
    T_THROW_IF_NOT_MSG(py::isinstance<py::str>(py_name), "name must be a string");
    std::string name = py_name.cast<std::string>();
    trim::Object value = trim::create_obejct(py_value);
    searcher->set(name, value);
  }

  void optimize(int num_threads = 0) { searcher->optimize(num_threads); }

  void clear_pruning_ratio() { searcher->clear_pruning_ratio(); }

  void clear_num_distance_computation() { searcher->clear_num_distance_computation(); }

  float get_pruning_ratio() const {
    if (!searcher) {
        throw std::runtime_error("Searcher is not initialized.");
    }
    return searcher->get_pruning_ratio();
  }

  float get_actual_distance_computation() const {
    if (!searcher) {
        throw std::runtime_error("Searcher is not initialized.");
    }
    return searcher->get_actual_distance_computation();
  }

  float get_total_distance_computation() const {
    if (!searcher) {
        throw std::runtime_error("Searcher is not initialized.");
    }
    return searcher->get_total_distance_computation();
  }

};

struct SearcherCreator {
  std::unique_ptr<trim::SearcherCreator> builder;

  explicit SearcherCreator(const std::string& index_type) {
    builder = std::make_unique<trim::SearcherCreator>(index_type);
  }

  SearcherCreator& set(py::object py_name, py::object py_value) {
    T_THROW_IF_NOT_MSG(py::isinstance<py::str>(py_name), "name must be a string");
    std::string name = py_name.cast<std::string>();
    trim::Object value = trim::create_obejct(py_value);
    builder->set(name, value);
    return *this;
  }

  Searcher create() {
    Searcher searcher(builder->create());
    return searcher;
  }
};

PYBIND11_MODULE(trimlib, m) {
  py::class_<Searcher>(m, "Searcher")
      .def(py::init<>())
      .def("set_data", &Searcher::set_data, py::arg("data"))
      .def("ann_search", &Searcher::ann_search, py::arg("query"), py::arg("k"))
      .def("range_search", &Searcher::range_search, py::arg("query"), py::arg("radius"))
      .def("set", &Searcher::set, py::arg("name"), py::arg("value"))
      .def("optimize", &Searcher::optimize, py::arg("num_threads") = 0)
      .def("clear_pruning_ratio", &Searcher::clear_pruning_ratio)
      .def("clear_num_distance_computation", &Searcher::clear_num_distance_computation)
      .def("get_pruning_ratio", &Searcher::get_pruning_ratio)
      .def("get_actual_distance_computation", &Searcher::get_actual_distance_computation)
      .def("get_total_distance_computation", &Searcher::get_total_distance_computation); 

  py::class_<SearcherCreator>(m, "SearcherCreator")
      .def(py::init<std::string>(), py::arg("index_type"))
      .def("set", &SearcherCreator::set, py::arg("key"), py::arg("value"))
      .def("create", &SearcherCreator::create);
}