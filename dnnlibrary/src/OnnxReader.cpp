#include <dnnlibrary/OnnxReader.h>

#include <fstream>

#include <tools/onnx2daq/OnnxConverter.h>

void OnnxReader::ReadOnnx(const std::string &filepath, ModelBuilder &builder) {
    ONNX_NAMESPACE::ModelProto model_proto;
    {
        std::ifstream ifs(filepath, std::ios::in | std::ios::binary);
        model_proto.ParseFromIstream(&ifs);
        ifs.close();
    }
    ReadOnnx(model_proto, builder);
}

void OnnxReader::ReadOnnx(const uint8_t *buf, const size_t size, ModelBuilder &builder) {
    ONNX_NAMESPACE::ModelProto model_proto;
    model_proto.ParseFromArray(buf, size);
    ReadOnnx(model_proto, builder);
}

void OnnxReader::ReadOnnx(const ONNX_NAMESPACE::ModelProto &model_proto, ModelBuilder &builder) {
    OnnxConverter converter;
    converter.Convert(model_proto);
    auto buf = converter.GetBuf();
    
    DaqReader daq_reader;
    daq_reader.ReadDaq(std::move(buf), builder);
}
