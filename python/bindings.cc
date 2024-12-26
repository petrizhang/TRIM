#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thread>

#include "top/hnsw/hnsw.h"
#include "top/searcher.h"
#include "top/core/object.h"

namespace py = pybind11;
using namespace pybind11::literals; // needed to bring in _a literal

namespace top {
Object pyobject_to_object(const py::object& obj) {
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
    throw std::runtime_error("Unsupported type for conversion to Object");
  }
}
}

inline void get_input_array_shapes(const py::buffer_info &buffer, size_t *rows,
                                   size_t *features) {
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

struct Graph {
  top::Graph<int> graph;

  Graph() = default;

  explicit Graph(const Graph &rhs) : graph(rhs.graph) {}

  explicit Graph(const std::string &filename) { graph.load(filename); }

  explicit Graph(const top::Graph<int> &graph) : graph(graph) {}

  void save(const std::string &filename) { graph.save(filename); }

  void load(const std::string &filename) { graph.load(filename); }
};

struct Searcher {
  std::unique_ptr<top::Searcher> searcher;

  Searcher(const Graph &graph, py::object input, const std::string &metric,
           int level)
      : searcher(std::unique_ptr<top::Searcher>(
            top::create_searcher(graph.graph, metric, level))) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
    auto buffer = items.request();
    size_t rows, features;
    get_input_array_shapes(buffer, &rows, &features);
    float *vector_data = (float *)items.data(0);
    searcher->set_data(vector_data, rows, features);
  }

  py::object ann_search(py::object query, int k) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(query);
    int *ids;
    ids = new int[k];
    searcher->ann_search(items.data(0), k, ids);
    py::capsule free_when_done(ids, [](void *f) { delete[] f; });
    return py::array_t<int>({k}, {sizeof(int)}, ids, free_when_done);
  }

  void set(py::object py_name, py::object py_value) {
    FAISS_THROW_IF_NOT_MSG(py::isinstance<py::str>(py_name), "name must be a string");
    std::string name = py_name.cast<std::string>();
    top::Object value = top::pyobject_to_object(py_value);
    searcher->set(name, value);
  }

  void optimize(int num_threads = 0) { searcher->optimize(num_threads); }
};

PYBIND11_MODULE(topnn, m) {
  py::class_<Graph>(m, "Graph")
      .def(py::init<>())
      .def(py::init<const std::string &>(), py::arg("filename"))
      .def("save", &Graph::save, py::arg("filename"))
      .def("load", &Graph::load, py::arg("filename"));

  py::class_<Searcher>(m, "Searcher")
      .def(py::init<const Graph &, py::object, const std::string &, int>(),
           py::arg("graph"), py::arg("data"), py::arg("metric"),
           py::arg("level"))
      .def("set", &Searcher::set, py::arg("name"), py::arg("value"))
      .def("ann_search", &Searcher::ann_search, py::arg("query"), py::arg("k"))
      .def("optimize", &Searcher::optimize, py::arg("num_threads") = 0);
}