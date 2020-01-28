#include <fstream>

#include <common/helper.h>
#include <tools/onnx2daq/OnnxConverter.h>

int main(int argc, char *argv[])
{
    ONNX_NAMESPACE::ModelProto model_proto;
    std::ifstream ifs(argv[1], std::ios::in | std::ios::binary);
    std::stringstream ss;
    ss << ifs.rdbuf();
    // FIXME: Handle the return value
    model_proto.ParseFromString(ss.str());
    dnn::OnnxConverter converter;
    const auto nodes = converter.GetSupportedNodes(model_proto);
    if (nodes) {
        const auto &supported_ops = nodes.value();
        PNT(supported_ops);
        return 0;
    } else {
        const auto &error = nodes.error();
        PNT(error);
        return 1;
    }
}
