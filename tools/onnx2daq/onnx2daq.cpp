#include <fstream>
#include <map>
#include <numeric>
#include <string>

#include <glog/logging.h>
#include "OnnxConverter.h"

using std::string;
using std::vector;

void usage(const std::string &filename) {
    std::cout << "Usage: " << filename << " onnx_model output_filename [table_file]" << std::endl;
}

int main(int argc, char **argv) {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    if (argc != 3 && argc != 4) {
        usage(argv[0]);
        return -1;
    }
    css table_file = argc == 4 ? argv[3] : "";
    ONNX_NAMESPACE::ModelProto model_proto;
    {
        std::ifstream ifs(argv[1], std::ios::in | std::ios::binary);
        model_proto.ParseFromIstream(&ifs);
        ifs.close();
    }

    OnnxConverter converter;
    converter.Convert(model_proto, table_file);
    converter.Save(argv[2]);

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
