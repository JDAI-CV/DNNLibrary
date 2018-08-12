#include <utility>

//
// Created by daquexian on 8/1/18.
//

#include "OnnxReader.h"

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <memory>
#include <utility>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <onnx.proto3.pb.h>
#include <android_log_helper.h>
#include <ModelBuilder.h>
#include <OnnxReader.h>

#include "NodeAttrHelper.h"

using std::string; using std::vector;

bool read_proto_from_text(string filepath, google::protobuf::Message &message) {
    std::ifstream fs;
    fs.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    bool success;
    try {
        fs.open(filepath.c_str(), std::ifstream::in);

        google::protobuf::io::IstreamInputStream input(&fs);
        success = google::protobuf::TextFormat::Parse(&input, &message);

        fs.close();

    } catch (const std::ifstream::failure &e) {
        LOGE("Open file %s error", filepath.c_str());
        success = false;
    }

    return success;
}

bool read_proto_from_binary(string filepath, google::protobuf::Message &message) {
    std::ifstream fs;
    fs.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    bool success;
    try {
        fs.open(filepath, std::ifstream::in | std::ifstream::binary);

        google::protobuf::io::IstreamInputStream input(&fs);
        google::protobuf::io::CodedInputStream codedstr(&input);

        codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

        success = message.ParseFromCodedStream(&codedstr);

        fs.close();
    } catch (const std::ifstream::failure &e) {
        LOGE("Open file %s error", filepath.c_str());
        success = false;
    }

    return success;
}

void OnnxReader::ReadFile(std::string filepath, ModelBuilder &builder) {
    std::map<string, ModelBuilder::Index> operand_indexes;

    read_proto_from_binary(std::move(filepath), model_proto_);
    for (const auto &tensor : model_proto_.graph().initializer()) {
        if (tensor.data_type() == onnx::TensorProto_DataType_FLOAT) {
            const float *ptr = tensor.float_data().empty() ?
                               reinterpret_cast<const float *>(tensor.raw_data().data()) : tensor.float_data().data();
            ModelBuilder::Shape shape;
            for (auto dim : tensor.dims()) {
                shape.push_back(static_cast<uint32_t>(dim));
            }
            operand_indexes[tensor.name()] = builder.addTensorFromBuffer(ptr, shape);
        }
    }

    vector<string> model_inputs;
    model_proto_.graph().input(0).type().tensor_type().shape();
    for (const auto &input : model_proto_.graph().input()) {
        if (operand_indexes.find(input.name()) != operand_indexes.end()) {
            continue;
        }

        ModelBuilder::Shape shape;     // NCHW order
        for (const auto &dim : input.type().tensor_type().shape().dim()) {
            if (dim.value_case() == onnx::TensorShapeProto_Dimension::kDimValue) {
                shape.push_back(static_cast<uint32_t>(dim.dim_value()));
            } else {
                throw std::invalid_argument("The input of graph doesn't have dim_value");
            }
        }
        operand_indexes[input.name()] = builder.addInput(shape[2], shape[3], shape[1]);
    }

    vector<string> skipped_act;
    for (const auto &node : model_proto_.graph().node()) {
        NodeAttrHelper helper(node);
        const auto &op = node.op_type();
        if (op == "Conv") {
            auto strides = helper.get("strides", vector<int>{1, 1});
            auto pads = helper.get("pads", vector<int>{0, 0, 0, 0});
            auto dilations = helper.get("dilations", vector<int>{1, 1});
            auto group = helper.get("group", 1);
            if (group != 1) {
                // TODO: Support it
                throw std::invalid_argument("group != 1 is not supported");
            }
            auto activation = find_activation(node);
            if (activation.first.has_value()) {
                skipped_act.push_back(activation.first.value());
            }
            std::optional<ModelBuilder::Index> bias_idx;
            if (node.input_size() == 3) {
                bias_idx = operand_indexes.at(node.input(2));
            }
            operand_indexes[node.output(0)] = builder.addConv(operand_indexes.at(node.input(0)), strides[1], strides[0],
                                                              pads[2], pads[3],
                                                              pads[1], pads[0],
                                                              activation.second, operand_indexes.at(node.input(1)),
                                                              bias_idx);
        } else if (op == "AveragePool" || op == "MaxPool") {
            auto strides = helper.get("strides", vector<int>{1, 1});
            auto pads = helper.get("pads", vector<int>{0, 0, 0, 0});
            auto kernel_shape = helper.get("kernel_shape", vector<int>{0, 0});
            auto count_include_pad = helper.get("count_include_pad", 0);
            if (count_include_pad == 1) {
                throw std::invalid_argument("count_include_pad == 1 is not supported");
            }
            auto storage_order = helper.get("storage_order", 0);
            if (storage_order == 1) {
                throw std::invalid_argument("storage_order == 1 is not supported");
            }
            if (helper.has_attr("auto_pad")) {
                throw std::invalid_argument("auto_pad is not supported");
            }
            auto activation = find_activation(node);
            if (activation.first.has_value()) {
                skipped_act.push_back(activation.first.value());
            }
            operand_indexes[node.output(0)] = builder.addPool(operand_indexes.at(node.input(0)), strides[1], strides[0],
                                                              pads[2], pads[3], pads[0], pads[1],
                                                              kernel_shape[0], kernel_shape[1], activation.second,
                                                              op == "AveragePool" ? ModelBuilder::AVE_POOL
                                                                                  : ModelBuilder::MAX_POOL);
        } else if (op == "Relu") {
            operand_indexes[node.output(0)] = builder.addReLU(operand_indexes.at(node.input(0)));
        } else if (op == "Add") {
            auto input1 = operand_indexes.at(node.input(0));
            auto input2 = operand_indexes.at(node.input(1));
            operand_indexes[node.output(1)] = builder.addAddTensor(input1, input2);
        } else if (op == "Gemm") {
            auto transA = helper.get("transA", 0);
            auto transB = helper.get("transB", 0);
            auto alpha = helper.get("alpha", 1.0f);
            auto beta = helper.get("beta", 1.0f);
            if (transA == 0 && transB == 1 && alpha == 1.f && beta == 1.f) {
                auto activation = find_activation(node);
                if (activation.first.has_value()) {
                    skipped_act.push_back(activation.first.value());
                }
                builder.addFC(operand_indexes.at(node.input(0)), activation.second,
                              operand_indexes.at(node.input(1)), operand_indexes.at(node.input(2)));
            } else {
                throw std::invalid_argument(
                        "Only transA == 0, transB == 1, alpha == 1.0 and beta == 1.0 is supported.");
            }
        }
    }
}

std::pair<std::optional<std::string>, FuseCode> OnnxReader::find_activation(const onnx::NodeProto &node) {
    std::pair<std::optional<string>, FuseCode> activation{{}, ANEURALNETWORKS_FUSED_NONE};
    for (const auto &_node : model_proto_.graph().node()) {
        if (node.output(0) == _node.input(0) && _node.op_type() == "Relu") {
            // If there are two branches after a conv/pool and both branches has a relu on the top, we have to add two normal relu layers
            if (activation.second != ANEURALNETWORKS_FUSED_NONE) {
                return {{}, ANEURALNETWORKS_FUSED_NONE};
            }
            activation = std::make_pair(std::make_optional(_node.name()), ANEURALNETWORKS_FUSED_RELU);
        }
    }
    return activation;
}

OnnxReader::OnnxReader(const std::string filepath, ModelBuilder &builder) {
    ReadFile(std::move(filepath), builder);
}
