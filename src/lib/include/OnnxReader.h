//
// Created by daquexian on 8/1/18.
//

#ifndef PROJECT_ONNXREADER_H
#define PROJECT_ONNXREADER_H

#include <string>
#include <optional>

#include <onnx.proto3.pb.h>

class OnnxReader {
public:
    void ReadFile(std::string filepath);

private:
    std::pair<std::optional<string>, FuseCode> find_activation(const onnx::ModelProto &model,
                                                               const onnx::NodeProto &node);
};


#endif //PROJECT_ONNXREADER_H
