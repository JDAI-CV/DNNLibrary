//
// Created by daquexian on 8/1/18.
//

#ifndef PROJECT_ONNXREADER_H
#define PROJECT_ONNXREADER_H

#include <string>
#include <optional>

#include <onnx.proto3.pb.h>
#include "ModelBuilder.h"

class OnnxReader {
public:
    explicit OnnxReader(std::string filepath, ModelBuilder &builder);

private:
    std::pair<std::optional<std::string>, FuseCode> find_activation(const onnx::NodeProto &node);
    void ReadFile(std::string filepath, ModelBuilder &builder);
    onnx::ModelProto model_proto_;
};


#endif //PROJECT_ONNXREADER_H
