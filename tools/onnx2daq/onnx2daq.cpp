#include <string>
#include <fstream>
#include <numeric>
#include <map>

#include <daq_generated.h>
#include <onnx.proto3.pb.h>
#include "NodeAttrHelper.h"
#include "log_helper.h"

using std::string; using std::vector;

#if 0
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
#endif

using Shape = std::vector<int>;

std::map<std::string, std::string> name_map;

std::string m(const std::string &str) {
    std::cout << "key: " << str << std::endl;
    if (name_map.find(str) != name_map.end()) {
        std::cout << "In map, " << name_map[str] << std::endl;
        return name_map[str];
    }
    
    std::cout << "Not in map" << std::endl;
    return str;
}

template <typename T>
uint32_t product(const vector<T> &v) {
    return static_cast<uint32_t> (accumulate(v.begin(), v.end(), 1, std::multiplies<T>()));
}

template <typename T>
struct Tensor {
    std::vector<T> data;
    Shape shape;
};

using FTensor = Tensor<float>;

enum class FuseCode {
    FUSED_NONE,
    FUSED_RELU,
    FUSED_RELU1,
    FUSED_RELU6
};

DNN::FuseCode convert_fuse_code_type(FuseCode fuse_code) {
    switch (fuse_code) {
        case FuseCode::FUSED_NONE:
            return DNN::FuseCode::None;
        case FuseCode::FUSED_RELU:
            return DNN::FuseCode::Relu;
        case FuseCode::FUSED_RELU1:
            return DNN::FuseCode::Relu1;
        case FuseCode::FUSED_RELU6:
            return DNN::FuseCode::Relu6;
    }
    throw std::invalid_argument("Invalid FuseCode");
}

std::pair<std::optional<std::string>, FuseCode> find_activation(const onnx::ModelProto &model_proto, const onnx::NodeProto &node) {
    std::pair<std::optional<string>, FuseCode> activation{{}, FuseCode::FUSED_NONE};
    for (const auto &_node : model_proto.graph().node()) {
        if (node.output(0) == _node.input(0) && _node.op_type() == "Relu") {
            // If there are two branches after a conv/pool and both branches has a relu on the top, we have to add two normal relu layers
            if (activation.second != FuseCode::FUSED_NONE) {
                return {{}, FuseCode::FUSED_NONE};
            }
            activation = std::make_pair(std::make_optional(_node.name()), FuseCode::FUSED_RELU);
        }
    }
    return activation;
}

/**
 * onnx: [filter_out_channel, filter_in_channel, height, width]
 * nnapi: [depth_out, height, width, depth_in]
 */
template <typename T>
Tensor<T> onnx2nnapi(const Tensor<T> &src) {
    Tensor<T> dest;
    dest.data.resize(product(src.shape));
    // t for total
    size_t out_t = src.shape[0], in_t = src.shape[1], h_t = src.shape[2], w_t = src.shape[3];
    for (size_t out = 0; out < out_t; out++) {
        for (size_t in = 0; in < in_t; in++) {
            for (size_t h = 0; h < h_t; h++) {
                for (size_t w = 0; w < w_t; w++) {
                    auto onnx_idx = out * in_t * h_t * w_t + in * h_t * w_t + h * w_t + w;
                    auto nnapi_idx = out * h_t * w_t * in_t + h * w_t * in_t + w * in_t + in;
                    dest.data[nnapi_idx] = src.data[onnx_idx];
                }
            }
        }
    }
    std::cout << "src shape " << src.shape << std::endl;
    dest.shape = {src.shape[0], src.shape[2], src.shape[3], src.shape[1]};
    std::cout << "dest shape " << dest.shape << std::endl;
    std::cout << "dest data " << std::endl << dest.data << std::endl;
    return dest;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "argc must be 3" << std::endl;
        return -1;
    }
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    onnx::ModelProto model_proto;
    {
        std::ifstream ifs(argv[1], std::ios::in | std::ios::binary);
        model_proto.ParseFromIstream(&ifs);
        ifs.close();
    }

    flatbuffers::FlatBufferBuilder builder;

    std::vector<std::string> operands;

    vector<flatbuffers::Offset<DNN::Tensor>> tensors;
    for (const auto &tensor : model_proto.graph().initializer()) {
        if (tensor.data_type() == onnx::TensorProto_DataType_FLOAT) {
            const float *ptr = tensor.float_data().empty() ?
                               reinterpret_cast<const float *>(tensor.raw_data().data()) : tensor.float_data().data();
            Shape shape;
            for (auto dim : tensor.dims()) {
                shape.push_back(static_cast<uint32_t>(dim));
            }
            std::cout << "tensor shape: " << shape << std::endl;
            auto data_vec = vector<float>(ptr, ptr + product(shape));
            
            FTensor onnx_tensor{data_vec, shape};
            FTensor nnapi_tensor;
            if (shape.size() == 4) {
                nnapi_tensor = onnx2nnapi(onnx_tensor);
            } else {
                nnapi_tensor = onnx_tensor;
            }
            auto flat_tensor = DNN::CreateTensorDirect(builder, DNN::DataType::Float32, nullptr, 
                    &nnapi_tensor.data, &nnapi_tensor.shape, tensor.name().c_str());
            tensors.push_back(flat_tensor);
        }
        operands.push_back(tensor.name());
    }

    vector<flatbuffers::Offset<DNN::Input>> inputs;
    for (const auto &input : model_proto.graph().input()) {
        if (std::find(operands.begin(), operands.end(), input.name()) != operands.end()) {
            continue;
        }

        Shape shape;
        for (const auto &dim : input.type().tensor_type().shape().dim()) {
            if (dim.value_case() == onnx::TensorShapeProto_Dimension::kDimValue) {
                shape.push_back(static_cast<uint32_t>(dim.dim_value()));
            } else {
                throw std::invalid_argument("The input of graph doesn't have dim_value");
            }
        }
        auto flat_input = DNN::CreateInputDirect(builder, &shape, input.name().c_str());
        inputs.push_back(flat_input);
    }

    vector<flatbuffers::Offset<DNN::Layer>> layers;
    vector<string> skipped_act;
    bool has_reshape = false;
    for (const auto &node : model_proto.graph().node()) {
        if (has_reshape) {
            throw std::invalid_argument("Reshape can only be the last layer for now");
        }
        NodeAttrHelper helper(node);
        const auto &op = node.op_type();
        if (op == "Conv") {
            auto strides = helper.get("strides", vector<int>{0, 0, 1, 1});
            auto pads = helper.get("pads", vector<int>{0, 0, 0, 0, 0, 0, 0, 0});
            auto dilations = helper.get("dilations", vector<int>{0, 0, 1, 1});
            std::cout << strides << ", " << pads << ", " << dilations << std::endl;
            strides = vector<int>(strides.begin() + 2, strides.end());
            pads = vector<int>(pads.begin() + 4, pads.end());
            dilations = vector<int>(dilations.begin() + 2, dilations.end());
            std::cout << strides << ", " << pads << ", " << dilations << std::endl;
            auto group = helper.get("group", 1);
            if (group != 1) {
                // TODO: Support it
                throw std::invalid_argument("group != 1 is not supported");
            }
            auto activation = find_activation(model_proto, node);
            if (activation.first.has_value()) {
                skipped_act.push_back(activation.first.value());
            }
            auto param = DNN::CreateConv2DDirect(builder, m(node.input(0)).c_str(), m(node.input(1)).c_str(),
                    node.input_size() == 3 ? m(node.input(2)).c_str() : nullptr,
                    &pads, &dilations, &strides, group, convert_fuse_code_type(activation.second), m(node.output(0)).c_str());
            auto layer = DNN::CreateLayer(builder, DNN::LayerType::Conv2D, param);
            layers.push_back(layer);
        } else if (op == "AveragePool" || op == "MaxPool") {
            auto strides = helper.get("strides", vector<int>{0, 0, 1, 1});
            auto pads = helper.get("pads", vector<int>{0, 0, 0, 0, 0, 0, 0, 0});
            auto kernel_shape = helper.get("kernel_shape", vector<int>{0, 0, 0, 0});
            auto count_include_pad = helper.get("count_include_pad", 0);
            strides = vector<int>(strides.begin() + 2, strides.end());
            pads = vector<int>(pads.begin() + 4, pads.end());
            kernel_shape = vector<int>(kernel_shape.begin() + 2, kernel_shape.end());
            std::cout << strides << ", " << pads << ", " << kernel_shape << std::endl;
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
            auto activation = find_activation(model_proto, node);
            if (activation.first.has_value()) {
                skipped_act.push_back(activation.first.value());
            }
            flatbuffers::Offset<DNN::Layer> layer;
            if (op == "AveragePool") {
                auto param = DNN::CreateAvePoolDirect(builder, m(node.input(0)).c_str(), &kernel_shape, &pads, &strides,
                        convert_fuse_code_type(activation.second), m(node.output(0)).c_str());
                layer = DNN::CreateLayer(builder, DNN::LayerType::AvePool, 0, param);
            } else {
                auto param = DNN::CreateMaxPoolDirect(builder, m(node.input(0)).c_str(), &kernel_shape, &pads, &strides,
                                                      convert_fuse_code_type(activation.second), m(node.output(0)).c_str());
                layer = DNN::CreateLayer(builder, DNN::LayerType::MaxPool, 0, 0, param);
            }
            layers.push_back(layer);
            // operand_indexes[node.output(0)] = builder.addPool(operand_indexes.at(node.input(0)), strides[1], strides[0],
                                                              // pads[2], pads[3], pads[0], pads[1],
                                                              // kernel_shape[0], kernel_shape[1], activation.second,
                                                              // op == "AveragePool" ? ModelBuilder::AVE_POOL
                                                                                  // : ModelBuilder::MAX_POOL);
        } else if (op == "Relu") {
            auto param = DNN::CreateReluDirect(builder, m(node.input(0)).c_str(), m(node.output(0)).c_str());
            auto layer = DNN::CreateLayer(builder, DNN::LayerType::Relu, 0, 0, 0, param);
            layers.push_back(layer);
            // operand_indexes[node.output(0)] = builder.addReLU(operand_indexes.at(node.input(0)));
        } else if (op == "Add") {
            auto activation = find_activation(model_proto, node);
            if (activation.first.has_value()) {
                skipped_act.push_back(activation.first.value());
            }
            auto param = DNN::CreateAddDirect(builder, m(node.input(0)).c_str(), m(node.input(1)).c_str(),
                    convert_fuse_code_type(activation.second), m(node.output(0)).c_str());
            auto layer = DNN::CreateLayer(builder, DNN::LayerType::Add, 0, 0, 0, 0, 0, 0, param);
            layers.push_back(layer);
            // auto input1 = operand_indexes.at(node.input(0));
            // auto input2 = operand_indexes.at(node.input(1));
            // operand_indexes[node.output(1)] = builder.addAddTensor(input1, input2);
        } else if (op == "Gemm") {
            auto transA = helper.get("transA", 0);
            auto transB = helper.get("transB", 0);
            auto alpha = helper.get("alpha", 1.0f);
            auto beta = helper.get("beta", 1.0f);
            if (transA == 0 && transB == 1 && alpha == 1.f && beta == 1.f) {
                auto activation = find_activation(model_proto, node);
                if (activation.first.has_value()) {
                    skipped_act.push_back(activation.first.value());
                }
                auto param = DNN::CreateFCDirect(builder, m(node.input(0)).c_str(), m(node.input(1)).c_str(),
                                                 node.input_size() == 3 ? m(node.input(2)).c_str() : nullptr,
                                                 convert_fuse_code_type(activation.second), m(node.output(0)).c_str()
                );
                auto layer = DNN::CreateLayer(builder, DNN::LayerType::FC, 0, 0, 0, 0, 0, param, 0);
                layers.push_back(layer);
                // builder.addFC(operand_indexes.at(node.input(0)), activation.second,
                              // operand_indexes.at(node.input(1)), operand_indexes.at(node.input(2)));
            } else {
                throw std::invalid_argument(
                        "Only transA == 0, transB == 1, alpha == 1.0 and beta == 1.0 is supported.");
            }
        } else if (op == "Softmax") {
            // simply ignore attribute "axis", because nnapi softmax didn't has this attr, and we will check the equality of the two ops in DaqReader.cpp
            auto param = DNN::CreateSoftmaxDirect(builder, m(node.input(0)).c_str(), m(node.output(0)).c_str());
            auto layer = DNN::CreateLayer(builder, DNN::LayerType::Softmax, 0, 0, 0, 0, param);
            layers.push_back(layer);
        } else if (op == "Concat") {
            vector<flatbuffers::Offset<flatbuffers::String>> inputs;
            for (auto onnx_input : node.input()) {
                auto flat_input = builder.CreateString(m(onnx_input).c_str(), onnx_input.size());
                inputs.push_back(flat_input);
            }
            auto axis = helper.get("axis", 1);
            uint32_t axis_nchw_to_nhwc[4]{0, 3, 1, 2};
            auto param = DNN::CreateConcatDirect(builder, &inputs, axis_nchw_to_nhwc[axis], m(node.output(0)).c_str());
            auto layer = DNN::CreateLayer(builder, DNN::LayerType::Concat, 0, 0, 0, 0, 0, 0, 0, param);
            layers.push_back(layer);
        } else if (op == "Dropout") {
            // Dropout does nothing, so the output is the same as the input
            name_map[node.output(0)] = m(node.input(0));
        } else if (op == "Reshape") {
            has_reshape = true;
        } else {
            throw std::invalid_argument("Unsupported operator " + op);
        }
    }
    auto flat_layers = builder.CreateVector(layers);
    auto flat_inputs = builder.CreateVector(inputs);
    auto flat_tensors = builder.CreateVector(tensors);
    auto flat_model = DNN::CreateModel(builder, flat_layers, flat_tensors, flat_inputs);

    builder.Finish(flat_model);

    std::ofstream ofs(argv[2]);
    ofs.write(reinterpret_cast<char *>(builder.GetBufferPointer()), builder.GetSize());
    ofs.close();

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
