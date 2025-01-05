#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thread>

#include "top/common/object.h"
#include "top/common/searcher.h"
#include "top/factory.h"

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

namespace top {

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
    throw std::runtime_error("Unsupported type for conversion to TOP Object");
  }
}

}  // namespace top

inline void get_input_array_shapes(const py::buffer_info& buffer, size_t* rows, size_t* features) {
  if (buffer.ndim != 2 && buffer.ndim != 1) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "Input vector data wrong shape. Number of dimensions %d. Data "
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
  std::unique_ptr<top::Searcher> searcher;

  Searcher() : searcher(nullptr) {}
  Searcher(std::unique_ptr<top::Searcher> searcher) : searcher(std::move(searcher)) {};

  // TODO: delete this method, read data from hnswlib directly
  void set_data(py::object input) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
    auto buffer = items.request();
    size_t rows, features;
    get_input_array_shapes(buffer, &rows, &features);
    float* vector_data = (float*)items.data(0);
    searcher->set_data(vector_data, rows, features);
  }

  py::object ann_search(py::object query, int k) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(query);
    int* ids;
    ids = new int[k];
    searcher->ann_search(items.data(0), k, ids);
    py::capsule free_when_done(ids, [](void* f) { delete[] f; });
    return py::array_t<int>({k}, {sizeof(int)}, ids, free_when_done);
  }

  void set(py::object py_name, py::object py_value) {
    TOP_THROW_IF_NOT_MSG(py::isinstance<py::str>(py_name), "name must be a string");
    std::string name = py_name.cast<std::string>();
    top::Object value = top::create_obejct(py_value);
    searcher->set(name, value);
  }

  void optimize(int num_threads = 0) { searcher->optimize(num_threads); }
};

struct SearcherBuilder {
  std::unique_ptr<top::SearcherBuilder> builder;

  explicit SearcherBuilder(const std::string& index_type) {
    builder = std::make_unique<top::SearcherBuilder>(index_type);
  }

  SearcherBuilder& set(py::object py_name, py::object py_value) {
    TOP_THROW_IF_NOT_MSG(py::isinstance<py::str>(py_name), "name must be a string");
    std::string name = py_name.cast<std::string>();
    top::Object value = top::create_obejct(py_value);
    builder->set(name, value);
    return *this;
  }

  Searcher build() {
    Searcher searcher(builder->build());
    return searcher;
  }
};

PYBIND11_MODULE(topnn, m) {
  py::class_<Searcher>(m, "Searcher")
      .def(py::init<>())
      .def("set_data", &Searcher::set_data, py::arg("data"))
      .def("ann_search", &Searcher::ann_search, py::arg("query"), py::arg("k"))
      .def("set", &Searcher::set, py::arg("name"), py::arg("value"))
      .def("optimize", &Searcher::optimize, py::arg("num_threads") = 0);

  py::class_<SearcherBuilder>(m, "SearcherBuilder")
      .def(py::init<std::string>(), py::arg("index_type"))
      .def("set", &SearcherBuilder::set, py::arg("key"), py::arg("value"))
      .def("build", &SearcherBuilder::build);
}