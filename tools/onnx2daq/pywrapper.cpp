#include <pybind11/pybind11.h>

#include <tools/onnx2daq/OnnxConverter.h>

namespace py = pybind11;
using dnn::OnnxConverter;

void convert(const std::string &model_str, const std::string &filepath,
             const std::string &table_file = "") {
    OnnxConverter converter;
    converter.Convert(model_str, filepath, table_file);
}

PYBIND11_MODULE(_onnx2daq, m) {
    m.def("convert", &convert, "");
}
