//
// Created by daquexian on 8/1/18.
//

#include "OnnxReader.h"

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <utility>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <onnx.proto3.pb.h>
#include <ModelBuilder.h>
#include "NodeAttrHelper.h"

using std::string; using std::vector;

bool read_proto_from_text(string filepath, google::protobuf::Message& message)
{
    std::ifstream fs(filepath, std::ifstream::in);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    bool success = google::protobuf::TextFormat::Parse(&input, &message);

    fs.close();

    return success;
}

bool read_proto_from_binary(string filepath, google::protobuf::Message& message)
{
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

    bool success = message.ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

void OnnxReader::ReadFile(std::string filepath) {
    std::map<string, ModelBuilder::Index> operand_indexes;
    ModelBuilder builder;
    onnx::ModelProto model_proto;
    read_proto_from_binary(filepath, model_proto);
    for (auto tensor : model_proto.graph().initializer()) {
        if (tensor.data_type() == onnx::TensorProto_DataType_FLOAT) {
            const float* ptr = tensor.float_data().empty() ?
                    reinterpret_cast<const float *>(tensor.raw_data().data()) : tensor.float_data().data();
            ModelBuilder::Shape shape;
            for (auto dim : tensor.dims()) {
                shape.push_back(static_cast<uint32_t>(dim));
            }
            operand_indexes[tensor.name()] = builder.addTensorFromBuffer(ptr, shape);
        }
    }

    vector<string> model_inputs;
    model_proto.graph().input(0).type().tensor_type().shape();
    for (auto input : model_proto.graph().input()) {
        if (operand_indexes.find(input.name()) != operand_indexes.end()) {
            continue;
        }

        ModelBuilder::Shape shape;     // NCHW order
        for (auto dim : input.type().tensor_type().shape().dim()) {
            if (dim.has_dim_value()) {
                shape.push_back(static_cast<uint32_t>(dim.dim_value()));
            } else {
                throw std::invalid_argument("The input of graph doesn't have dim_value");
            }
        }
        operand_indexes[input.name()] = builder.addInput(shape[2], shape[3], shape[1]);
    }

    vector<string> skipped_act;
    for (auto node : model_proto.graph().node()) {
        NodeAttrHelper helper(node);
        auto op = node.op_type();
        if (op == "Conv") {
            auto strides = helper.get("strides", vector<int>{1, 1});
            auto pads = helper.get("pads", vector<int>{0, 0, 0, 0});
            auto dilations = helper.get("dilations", vector<int>{1, 1});
            auto group = helper.get("group", 1);
            if (group != 1) {
                // TODO: Support it
                throw std::invalid_argument("group != 1 is not supported");
            }

            auto activation = find_activation(model_proto, node);

            if (activation.first.has_value()) {
                skipped_act.push_back(activation.first.value());
            }
            std::optional<ModelBuilder::Index> bias_idx;
            if (node.input_size() == 3) {
                bias_idx = operand_indexes[node.input(2)];
            }
            auto output_idx = builder.addConv(operand_indexes[node.input(0)], strides[1], strides[0], pads[2], pads[3],
                                              pads[1], pads[0],
                                              activation.second, operand_indexes[node.input(1)], bias_idx);
            operand_indexes[node.output(0)] = output_idx;
        } else if (op == "AveragePool") {
            auto strides = helper.get("strides", vector<int>{1, 1});
            auto pads = helper.get("pads", vector<int>{0, 0, 0, 0});
            auto kernel_shape = helper.get("kernel_shape", vector<int>{0, 0});
            auto count_include_pad = helper.get("count_include_pad", 0);
            if (count_include_pad == 1) {
                throw std::invalid_argument("count_include_pad == 1 is not supported");
            }
            if (helper.has_attr("auto_pad")) {
                throw std::invalid_argument("auto_pad is not supported");
            }
            // TODO:
        } else if (op == "Relu") {
            operand_indexes[node.output(0)] = builder.addReLU(operand_indexes[node.input(0)]);
        }
    }
}

std::pair<std::optional<string>, FuseCode> OnnxReader::find_activation(const onnx::ModelProto &model,
                                                                       const onnx::NodeProto &node) {
    std::optional<std::pair<string, FuseCode>> activation;
    for (auto _node : model.graph().node()) {
        if (node.output(0) == _node.input(0) && _node.op_type() == "Relu") {
            // If there are two branches after a conv/pool and both branches has a relu on the top, we have to add two normal relu layers
            if (activation) {
                return {{}, ANEURALNETWORKS_FUSED_NONE};
            }
            activation = std::make_pair({_node.name()}, ANEURALNETWORKS_FUSED_RELU);
        }
    }
    return activation;
}
