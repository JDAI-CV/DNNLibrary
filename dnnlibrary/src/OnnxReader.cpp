#include <dnnlibrary/OnnxReader.h>

#include <algorithm>
#include <fstream>
#include <iterator>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <tools/onnx2daq/OnnxConverter.h>

namespace dnn {
void OnnxReader::ReadOnnx(const std::string &filepath, ModelBuilder &builder) {
    ONNX_NAMESPACE::ModelProto model_proto;
    {
        std::ifstream ifs(filepath, std::ios::in | std::ios::binary);
        std::stringstream ss;
        ss << ifs.rdbuf();
        // FIXME: Handle the return value
        model_proto.ParseFromString(ss.str());
        ifs.close();
    }
    ReadOnnx(model_proto, builder);
}

void OnnxReader::ReadOnnx(const uint8_t *buf, const size_t size,
                          ModelBuilder &builder) {
    ONNX_NAMESPACE::ModelProto model_proto;
    // FIXME: Handle the return value
    model_proto.ParseFromArray(buf, size);
    ReadOnnx(model_proto, builder);
}

void OnnxReader::ReadOnnx(const ONNX_NAMESPACE::ModelProto &model_proto,
                          ModelBuilder &builder) {
    OnnxConverter converter;
    converter.Convert(model_proto);
    auto buf = converter.GetBuf();

    DaqReader daq_reader;
    daq_reader.ReadDaq(std::move(buf), builder);
}
}  // namespace dnn
