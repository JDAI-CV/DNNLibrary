#include <pybind11/pybind11.h>

#include "OnnxConverter.h"

namespace py = pybind11;

void convert(const std::string &model_str,
                            const std::string &filepath,
                            const css &table_file="") {

    ONNX_NAMESPACE::ModelProto model_proto;
    bool ret = model_proto.ParseFromString(model_str);
    if (!ret) {
        throw std::invalid_argument("Read protobuf string failed");
    }
    OnnxConverter converter;
    converter.Convert(model_proto, filepath, table_file);
    google::protobuf::ShutdownProtobufLibrary();
}

PYBIND11_MODULE(_onnx2daq, m) {
    m.def("convert", &convert, "");
}
