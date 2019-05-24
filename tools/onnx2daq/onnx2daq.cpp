#include <fstream>
#include <map>
#include <numeric>
#include <string>

#include <glog/logging.h>
#include "tools/onnx2daq/OnnxConverter.h"

using dnn::OnnxConverter;
using std::string;
using std::vector;

void usage(const std::string &filename) {
    std::cout << "Usage: " << filename
              << " onnx_model output_filename [table_file]" << std::endl;
}

int main(int argc, char **argv) {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    if (argc != 3 && argc != 4) {
        usage(argv[0]);
        return -1;
    }
    const std::string table_file = argc == 4 ? argv[3] : "";

    ONNX_NAMESPACE::ModelProto model_proto;
    std::ifstream ifs(argv[1], std::ios::in | std::ios::binary);
    std::stringstream ss;
    ss << ifs.rdbuf();
    // FIXME: Handle the return value
    model_proto.ParseFromString(ss.str());
    OnnxConverter converter;
    converter.Convert(model_proto, table_file);
    converter.Save(argv[2]);

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
