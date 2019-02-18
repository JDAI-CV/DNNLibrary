#include <fstream>
#include <map>
#include <numeric>
#include <string>

#include <common/StrKeyMap.h>
#include <glog/logging.h>
#include "NodeAttrHelper.h"
#include "OnnxConverter.h"
#include "common/log_helper.h"

using std::string;
using std::vector;

int main(int argc, char **argv) {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    if (argc != 3 && argc != 4) {
        std::cerr << "argc must be 3 or 4" << std::endl;
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
    converter.Convert(model_proto, argv[2], table_file);

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
