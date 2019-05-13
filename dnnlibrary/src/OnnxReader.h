//
// Created by daquexian on 05/12/19.
//

#ifndef DNNLIBRARY_ONNXREADER_H
#define DNNLIBRARY_ONNXREADER_H

#include <memory>
#include <string>

#include <common/daq_generated.h>
#include "DaqReader.h"
#include <flatbuffers/flatbuffers.h>
#include <ModelBuilder.h>
#include <onnx/onnx_pb.h>

class OnnxReader {
   public:
    void ReadOnnx(const std::string &filepath, ModelBuilder &builder);
    void ReadOnnx(const uint8_t *buf, const size_t size, ModelBuilder &builder);
    void ReadOnnx(const ONNX_NAMESPACE::ModelProto &model_proto, ModelBuilder &builder);
};

#endif  // DNNLIBRARY_ONNXREADER_H
#include <onnx/onnx_pb.h>
