#include <common/data_types.h>
#include <common/Shaper.h>
#include <common/StrKeyMap.h>
#include <common/helper.h>
#include <common/internal_vars.h>
#include <glog/logging.h>
#include <onnx/optimizer/optimize.h>
#include <onnx/shape_inference/implementation.h>
#include <tools/onnx2daq/OnnxConverter.h>

#include <fstream>
#include <map>
#include <numeric>
#include <string>
#ifdef __ANDROID__
#include <dnnlibrary/nnapi_implementation.h>
#endif
#include "NodeAttrHelper.h"

using std::string;
using std::vector;
using Shape = Shaper::Shape;

namespace dnn {
std::string OnnxConverter::m(const std::string &str) const {
    if (name_map_.find(str) != name_map_.end()) {
        return name_map_.at(str);
    }

    return str;
}

DNN::FuseCode OnnxConverter::ConvertFuseCodeType(FuseCode fuse_code) {
    switch (fuse_code) {
        case FuseCode::NONE:
            return DNN::FuseCode::None;
        case FuseCode::RELU:
            return DNN::FuseCode::Relu;
        case FuseCode::RELU1:
            return DNN::FuseCode::Relu1;
        case FuseCode::RELU6:
            return DNN::FuseCode::Relu6;
    }
    throw std::invalid_argument("Invalid FuseCode");
}

std::pair<dnn::optional<std::pair<int, ONNX_NAMESPACE::NodeProto>>, FuseCode>
OnnxConverter::FindActivation(const ONNX_NAMESPACE::ModelProto &model_proto,
                              css &output_name) {
    std::pair<dnn::optional<std::pair<int, ONNX_NAMESPACE::NodeProto>>,
              FuseCode>
        activation{{}, FuseCode::NONE};
    int i = 0;
    for (const auto &_node : model_proto.graph().node()) {
        if (!_node.input().empty() && output_name == _node.input(0) &&
            _node.op_type() == "Relu") {
            // If there are two branches after a conv/pool and both branches has
            // a relu on the top, we have to add two normal relu layers
            if (activation.second != FuseCode::NONE) {
                return {{}, FuseCode::NONE};
            }
            const auto node_pair = std::make_pair(i, _node);
            activation =
                std::make_pair(dnn::make_optional(node_pair), FuseCode::RELU);
        }
        i++;
    }
    if (activation.first.has_value()) {
        skipped_act_.push_back(activation.first.value().first);
        name_map_[activation.first.value().second.output(0)] = output_name;
    }
    return activation;
}

void OnnxConverter::CreateTensorFb(const Tensor &tensor,
                                   const DNN::DataType &data_type) {
    CreateTensorFb(tensor.name, tensor, data_type);
}

void OnnxConverter::CreateTensorFb(const std::string &name,
                                   const Tensor &tensor) {
    switch (tensor.data_type) {
        case Tensor::DataType::FLOAT32: {
            CreateTensorFb(name, tensor, DNN::DataType::Float32);
            break;
        }
        case Tensor::DataType::INT32: {
            CreateTensorFb(name, tensor, DNN::DataType::Int32);
            break;
        }
        case Tensor::DataType::UINT8: {
            const auto quant_info = quant_infos_.at(tensor.name);
            DNN::DataType daq_data_type;
            if (quant_info.scales.size() == 1 &&
                quant_info.zero_point.has_value()) {
                daq_data_type = DNN::DataType::QUANT8_ASYMM;
            } else if (quant_info.scales.size() == 1 &&
                       !quant_info.zero_point.has_value()) {
                daq_data_type = DNN::DataType::QUANT8_SYMM;
            } else {
                daq_data_type = DNN::DataType::QUANT8_SYMM_PER_CHANNEL;
            }
            CreateTensorFb(name, tensor, daq_data_type);
            break;
        }
    }
}

void OnnxConverter::CreateTensorFb(const std::string &name,
                                   const Tensor &tensor,
                                   const DNN::DataType &data_type) {
    flatbuffers::Offset<DNN::Tensor> fb_tensor;
    switch (tensor.data_type) {
        case Tensor::DataType::FLOAT32: {
            const auto data = tensor.float_data();
            fb_tensor = DNN::CreateTensorDirect(
                builder_, data_type, nullptr, &data, &tensor.shape,
                name.c_str(), nullptr, nullptr, nullptr);
            break;
        }
        case Tensor::DataType::INT32: {
            const auto data = tensor.int32_data();
            fb_tensor = DNN::CreateTensorDirect(
                builder_, data_type, nullptr, nullptr, &tensor.shape,
                name.c_str(), nullptr, nullptr, &data);
            break;
        }
        case Tensor::DataType::UINT8: {
            const auto data = tensor.uint8_data();
            fb_tensor = DNN::CreateTensorDirect(
                builder_, data_type, &data, nullptr, &tensor.shape,
                name.c_str(), nullptr, nullptr, nullptr);
            break;
        }
    }
    tensors_.push_back(fb_tensor);
}

std::vector<flatbuffers::Offset<flatbuffers::String>>
OnnxConverter::FbStrVector(const std::vector<std::string> &std_str_vector) {
    std::vector<flatbuffers::Offset<flatbuffers::String>> fb_str_vector;
    for (const auto &onnx_input : std_str_vector) {
        const auto flat_input =
            builder_.CreateString(m(onnx_input).c_str(), m(onnx_input).size());
        fb_str_vector.push_back(flat_input);
    }
    return fb_str_vector;
}

/**
 * onnx: [filter_out_channel, filter_in_channel / group, height, width]
 * nnapi: [1, height, width, depth_out]
 */
OnnxConverter::Tensor OnnxConverter::OnnxToNnapiAxes1230(const Tensor &src) {
    if (src.shape.size() != 4) {
        return src;
    }
    Tensor dest = src;
    size_t elemsize = 0;
    if (src.data_type == Tensor::DataType::UINT8) {
        elemsize = 1;
    } else if (src.data_type == Tensor::DataType::FLOAT32) {
        elemsize = 4;
    }
    dest.data.resize(Product(src.shape) * elemsize);
    // t for total
    auto out_t = src.shape[0], in_t = src.shape[1], h_t = src.shape[2],
         w_t = src.shape[3];
    CHECK_EQ(in_t, 1u);
    for (uint32_t out = 0; out < out_t; out++) {
        for (uint32_t in = 0; in < in_t; in++) {
            for (uint32_t h = 0; h < h_t; h++) {
                for (uint32_t w = 0; w < w_t; w++) {
                    auto onnx_idx =
                        out * in_t * h_t * w_t + in * h_t * w_t + h * w_t + w;
                    auto nnapi_idx = h * w_t * out_t + w * out_t + out;
                    FORZ(i, elemsize) {
                        dest.data[elemsize * nnapi_idx + i] =
                            src.data[elemsize * onnx_idx + i];
                    }
                }
            }
        }
    }
    dest.shape = {in_t, h_t, w_t, out_t};
    return dest;
}

OnnxConverter::Tensor OnnxConverter::OnnxToNnapiAxes0231(const Tensor &src) {
    if (src.shape.size() != 4) {
        return src;
    }
    Tensor dest = src;
    size_t elemsize = 0;
    if (src.data_type == Tensor::DataType::UINT8) {
        elemsize = 1;
    } else if (src.data_type == Tensor::DataType::FLOAT32) {
        elemsize = 4;
    }
    dest.data.resize(Product(src.shape) * elemsize);
    // t for total
    auto out_t = src.shape[0], in_t = src.shape[1], h_t = src.shape[2],
         w_t = src.shape[3];
    for (uint32_t out = 0; out < out_t; out++) {
        for (uint32_t in = 0; in < in_t; in++) {
            for (uint32_t h = 0; h < h_t; h++) {
                for (uint32_t w = 0; w < w_t; w++) {
                    auto onnx_idx =
                        out * in_t * h_t * w_t + in * h_t * w_t + h * w_t + w;
                    auto nnapi_idx =
                        out * h_t * w_t * in_t + h * w_t * in_t + w * in_t + in;
                    FORZ(i, elemsize) {
                        dest.data[elemsize * nnapi_idx + i] =
                            src.data[elemsize * onnx_idx + i];
                    }
                }
            }
        }
    }
    dest.shape = {out_t, h_t, w_t, in_t};
    return dest;
}

OnnxConverter::Tensor OnnxConverter::OnnxToNnapiIdentity(const Tensor &src) {
    return src;
}

void OnnxConverter::SetIdentity(css &input_name, css &output_name) {
    // Dropout does nothing, so the output is the same as the input
    shaper_.Eltwise(input_name, output_name);
    name_map_[output_name] = m(input_name);
}

// The reason that we only store weights rather than directly add them in daq
// model is that there may be different transform (nchw->nhwc or not) on the
// weights
void OnnxConverter::HandleInitializer() {
    for (const auto &tensor : model_proto_.graph().initializer()) {
        DNN_ASSERT(tensor.has_name(), "");
        css name = tensor.name();
        Shape shape;
        for (auto dim : tensor.dims()) {
            shape.push_back(static_cast<uint32_t>(dim));
        }
        if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            const char *ptr = tensor.float_data().empty()
                                  ? tensor.raw_data().data()
                                  : reinterpret_cast<const char *>(
                                        tensor.float_data().data());
            const auto data_vec =
                vector<char>(ptr, ptr + Product(shape) * sizeof(float));
            onnx_tensors_[name] = {name, data_vec, shape,
                                   Tensor::DataType::FLOAT32};
        } else if (tensor.data_type() ==
                   ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
            const auto *ptr =
                reinterpret_cast<const char *>(tensor.raw_data().data());
            const auto data_vec = vector<char>(ptr, ptr + Product(shape));
            onnx_tensors_[name] = {name, data_vec, shape,
                                   Tensor::DataType::UINT8};
        } else if (tensor.data_type() ==
                   ONNX_NAMESPACE::TensorProto_DataType_INT32) {
            const char *ptr = tensor.int32_data().empty()
                                  ? tensor.raw_data().data()
                                  : reinterpret_cast<const char *>(
                                        tensor.int32_data().data());
            const auto data_vec =
                vector<char>(ptr, ptr + Product(shape) * sizeof(int32_t));
            onnx_tensors_[name] = {name, data_vec, shape,
                                   Tensor::DataType::INT32};
        } else if (tensor.data_type() ==
                   ONNX_NAMESPACE::TensorProto_DataType_INT64) {
            // TODO: shape of reshape layer
        } else {
            DNN_ASSERT(false, "The data type \"" + std::to_string(tensor.data_type()) +
                                  "\" of tensor \"" +
                                  tensor.name() + "\" is not supported");
        }
        operands_.push_back(name);
    }
}

std::vector<flatbuffers::Offset<DNN::Input>>
OnnxConverter::GetInputOfOnnxModel() {
    vector<flatbuffers::Offset<DNN::Input>> inputs;

    for (const auto &input : model_proto_.graph().input()) {
        if (std::find(operands_.begin(), operands_.end(), input.name()) !=
            operands_.end()) {
            continue;
        }

        Shape shape;
        for (const auto &dim : input.type().tensor_type().shape().dim()) {
            if (dim.value_case() ==
                ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimValue) {
                shape.push_back(static_cast<uint32_t>(dim.dim_value()));
            } else {
                throw std::invalid_argument(
                    "The input of graph doesn't have dim_value");
            }
        }
        Shape nnapi_shape;
        if (shape.size() == 4) {
            nnapi_shape = Shape{shape[0], shape[2], shape[3], shape[1]};
        } else {
            nnapi_shape = shape;
        }
        shaper_.AddShape(input.name(), nnapi_shape);
        const auto flat_input = DNN::CreateInputDirect(builder_, &nnapi_shape,
                                                       input.name().c_str());
        inputs.push_back(flat_input);
    }

    return inputs;
}

std::vector<flatbuffers::Offset<flatbuffers::String>>
OnnxConverter::GetOutputOfOnnxModel() {
    std::vector<std::string> outputs;
    for (const auto &output : model_proto_.graph().output()) {
        outputs.push_back(m(output.name()));
    }
    return FbStrVector(outputs);
}

void OnnxConverter::ReadTableFile(css &table_file) {
    std::ifstream ifsf(table_file);
    DNN_ASSERT(!ifsf.fail(), "");
    for (std::string line; std::getline(ifsf, line);) {
        if (line.substr(0, 18) == "dequantize after: ") {
            dequantize_after_.push_back(line.substr(18));
            continue;
        }
        std::stringstream ss;
        ss << line;
        std::string name;
        int scale_num, zero_point_num;
        std::vector<float> scales;
        dnn::optional<int> zero_point;

        ss >> name >> scale_num;
        if (scale_num > 0) {
            FORZ(i, scale_num) {
                float scale;
                ss >> scale;
                scales.push_back(scale);
            }
        } else {
            scale_num = -scale_num;
            FORZ(j, scale_num) {
                std::string mul;
                ss >> mul;
                if (j == 0) {
                    scales = std::vector<float>(
                        quant_infos_.at(mul).scales.size(), 1.f);
                }
                FORZ(i, scales.size()) {
                    scales[i] *= quant_infos_.at(mul).scales[i];
                }
            }
        }
        ss >> zero_point_num;
        if (zero_point_num > 0) {
            int tmp;
            ss >> tmp;
            zero_point = tmp;
        }
        std::string quant_type_str;
        QuantInfo::Type quant_type;
        ss >> quant_type_str;
        if (quant_type_str == "quant8_asymm") {
            quant_type = QuantInfo::Type::QUANT8_ASYMM;
        } else if (quant_type_str == "int32") {
            quant_type = QuantInfo::Type::INT32;
        } else {
            throw std::invalid_argument(
                name + " has unknown quant type: " + quant_type_str);
        }
        quant_infos_[name] = {scales, zero_point, quant_type};
    }
}

std::vector<flatbuffers::Offset<DNN::QuantInfo>>
OnnxConverter::ConvertQuantInfosToFbs() {
    std::map<QuantInfo::Type, DNN::DataType> mapping_table{
        {QuantInfo::Type::INT32, DNN::DataType::Int32},
        {QuantInfo::Type::QUANT8_ASYMM, DNN::DataType::QUANT8_ASYMM}};
    std::vector<flatbuffers::Offset<DNN::QuantInfo>> ret;
    for (const auto x : quant_infos_) {
        const auto &quant_info = x.second;
        ret.push_back(DNN::CreateQuantInfoDirect(
            builder_, x.first.c_str(), mapping_table.at(quant_info.type),
            &quant_info.scales, quant_info.zero_point.value_or(0)));
    }
    return ret;
}

void OnnxConverter::Convert(const std::string &model_str,
                            const std::string &filepath,
                            const css &table_file) {
    ONNX_NAMESPACE::ModelProto model_proto;
    bool ret = model_proto.ParseFromString(model_str);
    if (!ret) {
        throw std::invalid_argument("Read protobuf string failed");
    }
    Convert(model_proto, table_file);
    Save(filepath);
}

dnn::optional<Shaper::Shape> GetShape(
    const ONNX_NAMESPACE::ModelProto &model_proto, const std::string &name) {
    for (const auto &value_info : model_proto.graph().value_info()) {
        if (value_info.name() == name) {
            if (!value_info.has_type()) {
                return dnn::nullopt;
            } else if (!value_info.type().has_tensor_type()) {
                return dnn::nullopt;
            } else if (!value_info.type().tensor_type().has_shape()) {
                return dnn::nullopt;
            } else if (value_info.type().tensor_type().shape().dim_size() ==
                       0) {
                return dnn::nullopt;
            }

            Shape shape;
            for (const auto &dim :
                 value_info.type().tensor_type().shape().dim()) {
                if (dim.has_dim_value()) {
                    shape.push_back(dim.dim_value());
                } else {
                    return dnn::nullopt;
                }
            }

            return shape;
        }
    }

    return dnn::nullopt;
}

std::pair<bool, std::string> OnnxConverter::IsNodeSupported(
    const ONNX_NAMESPACE::ModelProto &model_proto,
    const ONNX_NAMESPACE::NodeProto &node) const {
#ifdef __ANDROID__
    if (GetAndroidSdkVersion() < 27) {
        return {false, "Android API level is lower than 27"};
    }
#endif
    NodeAttrHelper helper(node);
    const auto &op = node.op_type();
    std::map<std::string, int> supported_ops{{"Conv", 27},
                                             {"AveragePool", 27},
                                             {"MaxPool", 27},
                                             {"GlobalAveragePool", 27},
                                             {"GlobalMaxPool", 27},
                                             {"Relu", 27},
                                             {"PRelu", 27},
                                             {"Add", 27},
                                             {"Mul", 27},
                                             {"Gemm", 27},
                                             {"Softmax", 27},
                                             {"Concat", 27},
                                             {"Dropout", 27},
                                             {"BatchNormalization", 27},
                                             {"Reshape", 27},
                                             {"LRN", 27},
                                             {"Identity", 27},
                                             {"Tanh", 27},
                                             {"Floor", 27},
                                             {"Sigmoid", 27},
                                             {"Abs", 29},
                                             {"Exp", 29},
                                             {"Sub", 27}};
    if (supported_ops.find(op) == supported_ops.end()) {
        return {false, "Unsupported operator " + op};
    }
#ifdef __ANDROID__
    if (supported_ops[op] > GetAndroidSdkVersion()) {
        return {false, "Operator " + op + " is only supported on API > " +
                           std::to_string(supported_ops[op])};
    }
#endif
    if (op == "Conv") {
        const auto strides = helper.get("strides", vector<int>{1, 1});
        const auto pads = helper.get("pads", vector<int>{0, 0, 0, 0});
        const auto dilations = helper.get("dilations", vector<int>{1, 1});
        const auto group = helper.get("group", 1);
        // if (dilations != vector<int>{1, 1} && strides != vector<int>{1, 1}) {
#ifdef __ANDROID__
        if (dilations != vector<int>{1, 1} && GetAndroidSdkVersion() <= 29) {
            return {
                false,
                // "Both dilations and strides > 1 is not supported for now"};
                "Dilations > 1 is not supported for API < 29 now"};
        }
#endif
        const auto weight_name = m(node.input(1));
        if (onnx_tensors_.has(weight_name)) {
            const auto &onnx_weight = onnx_tensors_.at(weight_name);
            if (group != 1 && onnx_weight.shape[1] != 1) {
                return {false, "group != 1 is not supported"};
            }
            if (onnx_weight.shape.size() != 4) {
                return {false, "Only conv 2d is supported."};
            }
        } else {
            return {false, "The weight of convolution must be known"};
        }
    } else if (op == "AveragePool" || op == "MaxPool") {
        const auto count_include_pad = helper.get("count_include_pad", 0);
        if (count_include_pad == 1) {
            return {false, "count_include_pad == 1 is not supported"};
        }
        const auto storage_order = helper.get("storage_order", 0);
        if (storage_order == 1) {
            return {false, "storage_order == 1 is not supported"};
        }
        if (helper.get("auto_pad", "NOTSET") != "NOTSET") {
            return {false, "auto_pad is not supported"};
        }
        if (helper.get("kernel_shape", std::vector<int>{1, 1}).size() != 2) {
            return {false, "Only pooling 2d is supported"};
        }
        if (helper.get("ceil_mode", 0) == 1) {
            return {false, "ceil_mode == 1 is not supported for pooling"};
        }
        if (helper.get("dilations", std::vector<int>{1, 1}) !=
            std::vector<int>{1, 1}) {
            return {false, "Dilations of pooling is not supported"};
        }
        if (node.output_size() != 1) {
            return {false, "Argmax in maxpooling is not supported"};
        }
    } else if (op == "GlobalAveragePool" || op == "GlobalMaxPool") {
        const auto &input_shape = GetShape(model_proto, node.input(0));
        if (!input_shape.has_value() || input_shape.value().size() != 4) {
            return {false, "Only rank-4 tensor is supported in " + op};
        }
    } else if (op == "PRelu") {
        const auto slope_name = m(node.input(1));
        if (onnx_tensors_.has(slope_name)) {
            if (onnx_tensors_.at(slope_name).shape != Shape{1}) {
                // TODO: support it
                return {false, "PRelu only support one element slope."};
            }
        } else {
            return {false, "PRelu slope must be known"};
        }
    } else if (op == "Gemm") {
        const auto transA = helper.get("transA", 0);
        const auto transB = helper.get("transB", 0);
        const auto alpha = helper.get("alpha", 1.0f);
        const auto beta = helper.get("beta", 1.0f);
        if (!(transA == 0 && transB == 1 && alpha == 1.f && beta == 1.f)) {
            return {false,
                    "Only transA == 0, transB == 1, alpha == 1.0 and beta == "
                    "1.0 is supported."};
        }
    } else if (op == "BatchNormalization") {
        if (node.output_size() != 1) {
            return {false,
                    "Your onnx model may be in training mode, please export "
                    "it in test mode."};
        }
        const auto scale_name = m(node.input(1));
        const auto b_name = m(node.input(2));
        const auto mean_name = m(node.input(3));
        const auto var_name = m(node.input(4));
        if (!onnx_tensors_.has(scale_name)) {
            return {false, "Scale of BN must be known"};
        }
        if (!onnx_tensors_.has(b_name)) {
            return {false, "B of BN must be known"};
        }
        if (!onnx_tensors_.has(mean_name)) {
            return {false, "Mean of BN must be known"};
        }
        if (!onnx_tensors_.has(var_name)) {
            return {false, "Var of BN must be known"};
        }
    } else if (op == "LRN") {
        const auto size = helper.get("size", 1);
        if (size % 2 == 0) {
            return {false, "NNAPI only support odd size for LRN"};
        }
    } else if (op == "Reshape") {
        const auto output_name = node.output(0);
        for (const auto another_node : model_proto_.graph().node()) {
            for (const auto input_name : another_node.input()) {
                if (input_name == output_name &&
                    another_node.op_type() != "Gemm") {
                    return {false,
                            "Reshape can only be the last layer or precede a "
                            "gemm layer for now"};
                }
            }
        }
    } else if (op == "Softmax") {
        const auto axis = helper.get("axis", 1);
        if (axis != 1) {
            return {false, "Only axis == 1 is supported in Softmax"};
        }
        const auto &input_shape = GetShape(model_proto, node.input(0));
        if (!input_shape.has_value() || input_shape.value().size() != 4) {
            return {false, "Only rank-4 tensor is supported in Softmax"};
        }
    } else if (op == "Concat") {
        const auto &input_shape = GetShape(model_proto, node.input(0));
        if (!input_shape.has_value() || input_shape.value().size() != 4) {
            return {false, "Only rank-4 tensor is supported in Softmax"};
        }
    }
    return {true, ""};
}

bool IsValidSupportedNodesVec(const std::vector<int> &supported_node_vec,
                              const ONNX_NAMESPACE::ModelProto &model_proto) {
    if (!supported_node_vec.empty()) {
        if (supported_node_vec.size() == 1) {
            const auto &node = model_proto.graph().node(supported_node_vec[0]);
            // Reshape and Dropout are simply ignored in DNNLibrary, causing the
            // input == output, which is not allowed in NNAPI
            if (node.op_type() == "Reshape" || node.op_type() == "Dropout" ||
                node.op_type() == "Identity") {
                return false;
            }
        }
        return true;
    }
    return false;
}

expected<std::vector<std::vector<int>>, std::string> OnnxConverter::GetSupportedNodes(
    ONNX_NAMESPACE::ModelProto model_proto) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    ONNX_NAMESPACE::shape_inference::InferShapes(model_proto);
    model_proto_ = model_proto;
    try {
        HandleInitializer();

        std::vector<std::vector<int>> supported_node_vecs;
        std::vector<int> supported_node_vec;
        for (int i = 0; i < model_proto.graph().node_size(); i++) {
            bool supported;
            std::string error_msg;
            std::tie(supported, error_msg) =
                IsNodeSupported(model_proto, model_proto.graph().node(i));
            if (supported) {
                supported_node_vec.push_back(i);
            } else {
                if (IsValidSupportedNodesVec(supported_node_vec, model_proto)) {
                    supported_node_vecs.push_back(supported_node_vec);
                    supported_node_vec.clear();
                }
            }
        }
        if (IsValidSupportedNodesVec(supported_node_vec, model_proto)) {
            supported_node_vecs.push_back(supported_node_vec);
        }
        Clear();
        return supported_node_vecs;
    } catch (std::exception &e) {
        return make_unexpected(e.what());
    }
}

void OnnxConverter::Convert(const ONNX_NAMESPACE::ModelProto &model_proto,
                            const css &table_file) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    model_proto_ = ONNX_NAMESPACE::optimization::OptimizeFixed(
        model_proto,
        vector<string>{
            "eliminate_deadend", "eliminate_identity", "eliminate_nop_dropout",
            "eliminate_nop_monotone_argmax", "eliminate_nop_pad",
            "extract_constant_to_initializer", "eliminate_unused_initializer",
            "eliminate_nop_transpose", "fuse_add_bias_into_conv",
            "fuse_bn_into_conv", "fuse_consecutive_concats",
            "fuse_consecutive_log_softmax", "fuse_consecutive_reduce_unsqueeze",
            "fuse_consecutive_squeezes", "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm", "fuse_pad_into_conv",
            "fuse_transpose_into_gemm"});

    if (!table_file.empty()) {
        ReadTableFile(table_file);
    }

    HandleInitializer();

    const auto inputs = GetInputOfOnnxModel();

    bool has_reshape = false;
    for (int i = 0; i < model_proto_.graph().node_size(); i++) {
        const auto &node = model_proto_.graph().node(i);
        NodeAttrHelper helper(node);
        const auto &op = node.op_type();
        VLOG(5) << "Node " << node.name();
        if (std::find(skipped_act_.begin(), skipped_act_.end(), i) !=
            skipped_act_.end()) {
            VLOG(5) << "Skip layer " << node.name();
            continue;
        }
        if (has_reshape && op != "Gemm") {
            throw std::invalid_argument(
                "Reshape can only be the last layer or precede a gemm layer "
                "for now");
        }
        if (op == "Conv") {
            VLOG(5) << "Start converting Conv";
            // onnx strides are in the order height, width
            // while nnapi strides are in the order width, height
            const auto onnx_strides = helper.get("strides", vector<int>{1, 1});
            // onnx pads are in the order top, left, bottom, right
            // while nnapi pads is in the order left, right, top, bottom
            const auto onnx_pads = helper.get("pads", vector<int>{0, 0, 0, 0});
            // onnx dilations is in the order height, width
            // while nnapi dilations are in the order width, height
            const auto onnx_dilations =
                helper.get("dilations", vector<int>{1, 1});
            CHECK_EQ(onnx_pads.size(), 4ul);
            CHECK_EQ(onnx_strides.size(), 2ul);
            CHECK_EQ(onnx_dilations.size(), 2ul);
            const decltype(onnx_strides) nnapi_strides{onnx_strides[1],
                                                       onnx_strides[0]};
            const decltype(onnx_pads) nnapi_pads{onnx_pads[1], onnx_pads[3],
                                                 onnx_pads[0], onnx_pads[2]};
            const decltype(onnx_dilations) nnapi_dilations{onnx_dilations[1],
                                                           onnx_dilations[0]};
            const auto group = helper.get("group", 1);
            dnn::optional<string> bias_name;
            if (node.input_size() >= 3) {
                bias_name = m(node.input(2));
            }
            const auto input_name = m(node.input(0));
            const auto output_name = m(node.output(0));

            const auto ori_weight_name = m(node.input(1));
            if (!onnx_tensors_.has(ori_weight_name)) {
                throw std::invalid_argument(
                    "The weight of convolution must be known");
            }
            const auto &onnx_weight = onnx_tensors_.at(ori_weight_name);
            const auto act = FindActivation(model_proto_, output_name);
            if (group == 1) {
                VLOG(5) << "Vanilla conv";
                WriteDaqLayer_CONV_2D(
                    input_name, ori_weight_name, bias_name, onnx_pads[1],
                    onnx_pads[3], onnx_pads[0], onnx_pads[2], onnx_strides[1],
                    onnx_strides[0], act.second, false, 1, 1, output_name);
            } else if (onnx_weight.shape[1] == 1) {  // depthwise
                VLOG(5) << "Depthwise conv";
                WriteDaqLayer_DEPTHWISE_CONV_2D(
                    input_name, ori_weight_name, bias_name, onnx_pads[1],
                    onnx_pads[3], onnx_pads[0], onnx_pads[2], onnx_strides[1],
                    onnx_strides[0], onnx_weight.shape[0] / group, act.second,
                    output_name);
            } else {
                // TODO: Support it
                throw std::invalid_argument("group != 1 is not supported");
            }
            VLOG(5) << "Converting Conv completed";
        } else if (op == "AveragePool" || op == "MaxPool" ||
                   op == "GlobalAveragePool" || op == "GlobalMaxPool") {
            VLOG(5) << "Start converting Pool";
            const auto input_name = m(node.input(0));
            const auto output_name = m(node.output(0));
            const auto act = FindActivation(model_proto_, output_name).second;
            if (op == "AveragePool" || op == "MaxPool") {
                vector<int> nnapi_strides, nnapi_pads, kernel_shape;
                kernel_shape = helper.get("kernel_shape", vector<int>{0, 0});
                const auto count_include_pad =
                    helper.get("count_include_pad", 0);
                const auto onnx_strides =
                    helper.get("strides", vector<int>{1, 1});
                const auto onnx_pads =
                    helper.get("pads", vector<int>{0, 0, 0, 0});
                nnapi_strides = {onnx_strides[1], onnx_strides[0]};
                nnapi_pads = {onnx_pads[1], onnx_pads[3], onnx_pads[0],
                              onnx_pads[2]};
                if (count_include_pad == 1) {
                    throw std::invalid_argument(
                        "count_include_pad == 1 is not supported");
                }
                const auto storage_order = helper.get("storage_order", 0);
                if (storage_order == 1) {
                    throw std::invalid_argument(
                        "storage_order == 1 is not supported");
                }
                if (helper.get("auto_pad", "NOTSET") != "NOTSET") {
                    throw std::invalid_argument("auto_pad is not supported");
                }
                CHECK_EQ(nnapi_pads.size(), 4ul);
                CHECK_EQ(kernel_shape.size(), 2ul);
                CHECK_EQ(nnapi_strides.size(), 2ul);
                // kernel_shape of onnx model is [height, width]
                if (op == "AveragePool") {
                    WriteDaqLayer_AVERAGE_POOL_2D(
                        input_name, onnx_pads[1], onnx_pads[3], onnx_pads[0],
                        onnx_pads[2], onnx_strides[1], onnx_strides[0],
                        kernel_shape[1], kernel_shape[0], act, output_name);
                } else {
                    WriteDaqLayer_MAX_POOL_2D(
                        input_name, onnx_pads[1], onnx_pads[3], onnx_pads[0],
                        onnx_pads[2], onnx_strides[1], onnx_strides[0],
                        kernel_shape[1], kernel_shape[0], act, output_name);
                }
            } else {
                const auto input_height = shaper_[input_name][1];
                const auto input_width = shaper_[input_name][2];
                if (op == "GlobalAveragePool") {
                    WriteDaqLayer_AVERAGE_POOL_2D(input_name, 0, 0, 0, 0, 1, 1,
                                                  input_width, input_height,
                                                  act, output_name);
                } else {
                    WriteDaqLayer_MAX_POOL_2D(input_name, 0, 0, 0, 0, 1, 1,
                                              input_width, input_height, act,
                                              output_name);
                }
            }
            VLOG(5) << "Converting Pool completed";
        } else if (op == "Relu") {
            VLOG(5) << "Start converting Relu";
            const auto input_name = m(node.input(0));
            const auto output_name = m(node.output(0));
            WriteDaqLayer_RELU(input_name, output_name);
            VLOG(5) << "Converting Relu completed";
        } else if (op == "PRelu") {
            VLOG(5) << "Start converting PRelu";
            const auto input_name = m(node.input(0));
            const auto slope_name = m(node.input(1));
            const auto output_name = m(node.output(0));
            WriteDaqLayer_PRELU(input_name, slope_name, output_name);
            VLOG(5) << "Converting PRelu completed";
        } else if (op == "Add") {
            VLOG(5) << "Start converting Add";
            const auto input1_name = m(node.input(0));
            const auto input2_name = m(node.input(1));
            const auto output_name = m(node.output(0));
            const auto act = FindActivation(model_proto_, output_name).second;
            WriteDaqLayer_ADD(input1_name, input2_name, act, output_name);
            VLOG(5) << "Converting Add completed";
        } else if (op == "Mul") {
            VLOG(5) << "Start converting Mul";
            const auto input1_name = m(node.input(0));
            const auto input2_name = m(node.input(1));
            const auto output_name = m(node.output(0));
            const auto act = FindActivation(model_proto_, output_name).second;
            WriteDaqLayer_MUL(input1_name, input2_name, act, output_name);
            VLOG(5) << "Converting Mul completed";
        } else if (op == "Gemm") {
            VLOG(5) << "Start converting Gemm";
            const auto input_name = m(node.input(0));
            const auto weight_name = m(node.input(1));
            const auto output_name = m(node.output(0));
            const auto act = FindActivation(model_proto_, output_name).second;
            dnn::optional<string> bias_name;
            if (node.input_size() >= 3) {
                bias_name = m(node.input(2));
            }
            const auto transA = helper.get("transA", 0);
            const auto transB = helper.get("transB", 0);
            const auto alpha = helper.get("alpha", 1.0f);
            const auto beta = helper.get("beta", 1.0f);
            if (transA == 0 && transB == 1 && alpha == 1.f && beta == 1.f) {
                WriteDaqLayer_FULLY_CONNECTED(input_name, weight_name,
                                              bias_name, act, output_name);
            } else {
                throw std::invalid_argument(
                    "Only transA == 0, transB == 1, alpha == 1.0 and beta == "
                    "1.0 is "
                    "supported.");
            }
            has_reshape = false;
            VLOG(5) << "Converting Gemm completed";
        } else if (op == "Softmax") {
            VLOG(5) << "Start converting Softmax";
            const auto input_name = m(node.input(0));
            const auto output_name = m(node.output(0));
            WriteDaqLayer_SOFTMAX(input_name, 1.f, output_name);
            VLOG(5) << "Converting Softmax completed";
        } else if (op == "Concat") {
            VLOG(5) << "Start converting Concat";
            vector<std::string> concat_inputs_str;
            for (const auto &onnx_input : node.input()) {
                concat_inputs_str.push_back(m(onnx_input));
            }
            const uint32_t axis_nchw_to_nhwc[4]{0, 3, 1, 2};
            const auto axis = helper.get("axis", 1);
            const auto output_name = m(node.output(0));
            WriteDaqLayer_CONCATENATION(concat_inputs_str,
                                        axis_nchw_to_nhwc[axis], output_name);
            VLOG(5) << "Converting Concat completed";
        } else if (op == "Dropout") {
            VLOG(5) << "Start converting Dropout";
            SetIdentity(node.input(0), node.output(0));
            VLOG(5) << "Converting Dropout completed";

        } else if (op == "BatchNormalization") {
            VLOG(5) << "Start converting BatchNormalization";
            DNN_ASSERT(node.output_size() == 1,
                       "Your onnx model may be in training mode, please export "
                       "it in test mode.")
            const auto input_name = m(node.input(0));

            const auto scale_name = m(node.input(1));
            const auto bias_name = m(node.input(2));
            const auto mean_name = m(node.input(3));
            const auto var_name = m(node.input(4));

            const auto scale_tensor = onnx_tensors_.at(scale_name);
            const auto bias_tensor = onnx_tensors_.at(bias_name);
            const auto mean_tensor = onnx_tensors_.at(mean_name);
            const auto var_tensor = onnx_tensors_.at(var_name);

            const auto eps = helper.get("epsilon", 1e-5f);

            const auto output_name = m(node.output(0));
            const auto act = FindActivation(model_proto_, output_name).second;

            vector<float> a, b;
            FORZ(i, scale_tensor.shape[0]) {
                a.push_back(scale_tensor.float_data()[i] /
                            sqrt(var_tensor.float_data()[i] + eps));
                b.push_back((scale_tensor.float_data()[i] *
                             -mean_tensor.float_data()[i]) /
                                sqrt(var_tensor.float_data()[i] + eps) +
                            bias_tensor.float_data()[i]);
            }

            const auto tensor_a_name = output_name + "_imm_a";
            const auto tensor_b_name = output_name + "_imm_b";
            const auto tensor_imm_product_name = output_name + "_imm_mul";
            const auto flat_tensor_a = DNN::CreateTensorDirect(
                builder_, DNN::DataType::Float32, nullptr, &a,
                &scale_tensor.shape, tensor_a_name.c_str());
            const auto flat_tensor_b = DNN::CreateTensorDirect(
                builder_, DNN::DataType::Float32, nullptr, &b,
                &scale_tensor.shape, tensor_b_name.c_str());
            shaper_.AddShape(tensor_a_name, scale_tensor.shape);
            shaper_.AddShape(tensor_b_name, scale_tensor.shape);
            tensors_.push_back(flat_tensor_a);
            tensors_.push_back(flat_tensor_b);
            WriteDaqLayer_MUL(input_name, tensor_a_name, FuseCode::NONE,
                              tensor_imm_product_name);
            WriteDaqLayer_ADD(tensor_imm_product_name, tensor_b_name, act,
                              output_name);

            VLOG(5) << "Converting BatchNormalization completed";
        } else if (op == "Reshape") {
            VLOG(5) << "Start converting Reshape";
            has_reshape = true;
            SetIdentity(node.input(0), node.output(0));
            VLOG(5) << "Converting Reshape completed";
        } else if (op == "LRN") {
            VLOG(5) << "Start converting LRN";
            if (!helper.has_attr("size")) {
                throw std::invalid_argument(
                    "Invalid ONNX model, attribute \"size\" is required in "
                    "LRN");
            }
            const auto size = helper.get("size", 1);
            auto alpha = helper.get("alpha", 0.0001f);
            const auto beta = helper.get("beta", 0.75f);
            const auto bias = helper.get("bias", 1.f);
            if (size % 2 == 0) {
                std::invalid_argument("NNAPI only support odd size for LRN");
            }
            const auto radius = (size - 1) / 2;
            alpha /= size;  // The implementation of ONNX LRN is not the same as
                            // that of NNAPI LRN
            WriteDaqLayer_LOCAL_RESPONSE_NORMALIZATION(
                node.input(0), radius, bias, alpha, beta, node.output(0));
            VLOG(5) << "Converting LRN completed";
        } else if (op == "Tanh") {
            VLOG(5) << "Start converting Tanh";
            const auto input_name = m(node.input(0));
            const auto output_name = m(node.output(0));
            WriteDaqLayer_TANH(input_name, output_name);
            VLOG(5) << "Converting Tanh completed";
        } else if (op == "Floor") {
            VLOG(5) << "Start converting Floor";
            const auto input_name = m(node.input(0));
            const auto output_name = m(node.output(0));
            WriteDaqLayer_FLOOR(input_name, output_name);
            VLOG(5) << "Converting Floor completed";
        } else if (op == "Sigmoid") {
            VLOG(5) << "Start converting Sigmoid";
            const auto input_name = m(node.input(0));
            const auto output_name = m(node.output(0));
            WriteDaqLayer_LOGISTIC(input_name, output_name);
            VLOG(5) << "Converting Sigmoid completed";
        } else if (op == "Abs") {
            VLOG(5) << "Start converting Abs";
            const auto input_name = m(node.input(0));
            const auto output_name = m(node.output(0));
            WriteDaqLayer_ABS(input_name, output_name);
            VLOG(5) << "Converting Abs completed";
        } else if (op == "Exp") {
            VLOG(5) << "Start converting Exp";
            const auto input_name = m(node.input(0));
            const auto output_name = m(node.output(0));
            WriteDaqLayer_EXP(input_name, output_name);
            VLOG(5) << "Converting Exp completed";
        } else if (op == "Sub") {
            VLOG(5) << "Start converting Sub";
            const auto input1_name = m(node.input(0));
            const auto input2_name = m(node.input(1));
            const auto output_name = m(node.output(0));
            const auto act = FindActivation(model_proto_, output_name).second;
            WriteDaqLayer_SUB(input1_name, input2_name, act, output_name);
            VLOG(5) << "Converting Sub completed";
        } else {
            throw std::invalid_argument("Unsupported operator " + op);
        }
        FORZ(i, node.output_size()) {
            const auto output = node.output(i);
            if (std::find(dequantize_after_.begin(), dequantize_after_.end(),
                          output) != dequantize_after_.end()) {
                css dequant_output = output + "_dequant";
                WriteDaqLayer_DEQUANTIZE(output, dequant_output);
                name_map_[output] = dequant_output;
            }
        }
    }
    const auto outputs = GetOutputOfOnnxModel();

    const auto flat_layers = builder_.CreateVector(layers_);
    const auto flat_inputs = builder_.CreateVector(inputs);
    const auto flat_tensors = builder_.CreateVector(tensors_);
    const auto flat_quant_infos =
        builder_.CreateVector(ConvertQuantInfosToFbs());
    const auto flat_outputs = builder_.CreateVector(outputs);
    const auto flat_model = DNN::CreateModel(
        builder_, flat_layers, flat_tensors, flat_inputs, flat_quant_infos,
        flat_outputs, dnn::CURRENT_MODEL_VERSION);

    builder_.Finish(flat_model);

    VLOG(5) << "Shapes: ";
    VLOG(5) << shaper_;

    Clear();
}

void OnnxConverter::Clear() {
    skipped_act_.clear();
    layers_.clear();
    operands_.clear();
    tensors_.clear();
    name_map_.clear();
    nnapi_tensors_.clear();
    onnx_tensors_.clear();
    shaper_.Clear();
}

void OnnxConverter::Save(const std::string &filename) {
    std::ofstream ofs(filename);
    ofs.write(reinterpret_cast<char *>(builder_.GetBufferPointer()),
              builder_.GetSize());
    ofs.close();
}

std::unique_ptr<uint8_t[]> OnnxConverter::GetBuf() {
    std::unique_ptr<uint8_t[]> ptr(new uint8_t[builder_.GetSize()]);
    memcpy(ptr.get(), builder_.GetBufferPointer(), builder_.GetSize());
    return ptr;
}
}  // namespace dnn
