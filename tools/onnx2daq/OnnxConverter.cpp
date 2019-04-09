#include "OnnxConverter.h"

#include <fstream>
#include <map>
#include <numeric>
#include <string>

#include <common/Shaper.h>
#include <common/StrKeyMap.h>
#include <glog/logging.h>
#include <onnx/optimizer/optimize.h>
#include "NodeAttrHelper.h"

using std::string;
using std::vector;
using Shape = Shaper::Shape;

std::string OnnxConverter::m(const std::string &str) {
    if (name_map_.find(str) != name_map_.end()) {
        return name_map_[str];
    }

    return str;
}

DNN::FuseCode OnnxConverter::ConvertFuseCodeType(FuseCode fuse_code) {
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

std::pair<nonstd::optional<std::string>, OnnxConverter::FuseCode>
OnnxConverter::FindActivation(const ONNX_NAMESPACE::ModelProto &model_proto,
                              css &output_name) {
    std::pair<nonstd::optional<string>, FuseCode> activation{
        {}, FuseCode::FUSED_NONE};
    for (const auto &_node : model_proto.graph().node()) {
        if (!_node.input().empty() && output_name == _node.input(0) &&
            _node.op_type() == "Relu") {
            // If there are two branches after a conv/pool and both branches has
            // a relu on the top, we have to add two normal relu layers
            if (activation.second != FuseCode::FUSED_NONE) {
                return {{}, FuseCode::FUSED_NONE};
            }
            activation = std::make_pair(nonstd::make_optional(_node.name()),
                                        FuseCode::FUSED_RELU);
        }
    }
    return activation;
}

void OnnxConverter::CreateTensorFb(const Tensor &tensor, const DNN::DataType &data_type) {
    CreateTensorFb(tensor.name, tensor, data_type);
}

void OnnxConverter::CreateTensorFb(const std::string &name, const Tensor &tensor) {
    switch (tensor.data_type) {
        case Tensor::DataType::FLOAT32:
        {
            CreateTensorFb(name, tensor, DNN::DataType::Float32);
            break;
        }
        case Tensor::DataType::INT32:
        {
            CreateTensorFb(name, tensor, DNN::DataType::Int32);
            break;
        }
        case Tensor::DataType::UINT8:
        {
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

void OnnxConverter::CreateTensorFb(const std::string &name, const Tensor &tensor, const DNN::DataType &data_type) {
    flatbuffers::Offset<DNN::Tensor> fb_tensor;
    switch (tensor.data_type) {
        case Tensor::DataType::FLOAT32:
        {
            const auto data = tensor.float_data();
            fb_tensor = DNN::CreateTensorDirect(builder_, data_type, nullptr, &data, &tensor.shape, name.c_str(), nullptr, nullptr, nullptr);
            break;
        }
        case Tensor::DataType::INT32:
        {
            const auto data = tensor.int32_data();
            fb_tensor = DNN::CreateTensorDirect(builder_, data_type, nullptr, nullptr, &tensor.shape, name.c_str(), nullptr, nullptr, &data);
            break;
        }
        case Tensor::DataType::UINT8:
        {
            const auto data = tensor.uint8_data();
            fb_tensor = DNN::CreateTensorDirect(builder_, data_type, &data, nullptr, &tensor.shape, name.c_str(), nullptr, nullptr, nullptr);
            break;
        }
    }
    tensors_.push_back(fb_tensor);
}

std::vector<flatbuffers::Offset<flatbuffers::String>> OnnxConverter::FbStrVector(const std::vector<std::string> &std_str_vector) {
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
OnnxConverter::Tensor OnnxConverter::OnnxToNnapiDwConvWeight(const Tensor &src) {
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
                    auto onnx_idx = out * in_t * h_t * w_t +
                                    in * h_t * w_t + h * w_t + w;
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
    dest.name = src.name + "_conv_w";
    return dest;
}

OnnxConverter::Tensor OnnxConverter::OnnxToNnapiVanillaConvWeight(const Tensor &src) {
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
                    auto onnx_idx = out * in_t * h_t * w_t +
                                    in * h_t * w_t + h * w_t + w;
                    auto nnapi_idx = out * h_t * w_t * in_t +
                                     h * w_t * in_t + w * in_t + in;
                    FORZ(i, elemsize) {
                        dest.data[elemsize * nnapi_idx + i] =
                            src.data[elemsize * onnx_idx + i];
                    }
                }
            }
        }
    }
    dest.shape = {out_t, h_t, w_t, in_t};
    dest.name = src.name + "_conv_w";
    return dest;
}

void OnnxConverter::AddConv(
    const string &input_name, const std::vector<int> &strides,
    const std::vector<int> &pads, const std::vector<int> &dilations, int group,
    const std::pair<nonstd::optional<std::string>, FuseCode> &activation,
    const string &ori_weight_name,
    const nonstd::optional<std::string> &bias_name, const string &output_name) {
    flatbuffers::Offset<DNN::Layer> layer;
    if (dilations != vector<int>{1, 1}) {
        if (strides != vector<int>{1, 1}) {
            throw std::invalid_argument(
                "Both dilations and strides > 1 is not supported for now");
        }
        LOG(INFO) << "Dilations of conv: " << dilations << ", converting..";
        const auto s2b_name = input_name + "_s2b";
        const auto im_name = input_name + "_conv_imm";
        const auto b2s_name = input_name + "_b2s";
        std::vector<int> new_pads = pads;
        const auto input_shape = shaper_[input_name];
        new_pads[1] = (input_shape[1] + pads[1] + (dilations[0] - 1)) /
                          dilations[0] * dilations[0] -
                      input_shape[1];
        new_pads[3] = (input_shape[2] + pads[3] + (dilations[1] - 1)) /
                          dilations[1] * dilations[1] -
                      input_shape[2];
        LOG(INFO) << input_shape << ", " << pads << ", " << dilations << ", "
                  << new_pads;
        {
            shaper_.SpaceToBatch(input_name, dilations, new_pads, s2b_name);
            const auto param = DNN::CreateSpaceToBatchDirect(
                builder_, m(input_name).c_str(), &dilations, &new_pads,
                s2b_name.c_str());
            layer = DNN::CreateLayer(builder_, DNN::LayerType::SpaceToBatch, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);
        }
        {
            // paddings are applied in spacetobatch
            AddConv(s2b_name, strides, vector<int>{0, 0, 0, 0},
                    vector<int>{1, 1}, group, activation, ori_weight_name,
                    bias_name, im_name);
        }
        {
            shaper_.BatchToSpace(im_name, dilations, b2s_name);
            const auto param = DNN::CreateBatchToSpaceDirect(
                builder_, im_name.c_str(), &dilations, b2s_name.c_str());
            layer = DNN::CreateLayer(builder_, DNN::LayerType::BatchToSpace, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);
        }
        {
            const auto b2s_shape = shaper_[b2s_name];
            const std::vector<int32_t> starts{0, 0, 0, 0};
            const std::vector<int32_t> ends{
                static_cast<int32_t>(b2s_shape[0]),
                static_cast<int32_t>(b2s_shape[1]) - (new_pads[1] - pads[0]),
                static_cast<int32_t>(b2s_shape[2]) - (new_pads[3] - pads[3]),
                static_cast<int32_t>(b2s_shape[3])};
            const std::vector<int32_t> strides_in_ss{1, 1, 1, 1};
            const int32_t begin_mask = 0;
            const int32_t end_mask = 0;
            const int32_t shrink_axis_mask = 0;
            shaper_.StridedSlice(b2s_name, starts, ends, strides_in_ss,
                                 begin_mask, end_mask, shrink_axis_mask,
                                 output_name);
            const auto param = DNN::CreateStridedSliceDirect(
                builder_, b2s_name.c_str(), &starts, &ends, &strides_in_ss,
                begin_mask, end_mask, shrink_axis_mask, output_name.c_str());
            layer = DNN::CreateLayer(builder_, DNN::LayerType::StridedSlice, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);
        }
        return;
    }

    const auto &onnx_weight = onnx_tensors_.at(ori_weight_name);
    string weight_name;
    Tensor weight_tensor;
    if (group == 1) {
        LOG(INFO) << "Vanilla conv";
        weight_tensor = OnnxToNnapiVanillaConvWeight(onnx_weight);
        weight_name = weight_tensor.name;
        shaper_.AddShape(weight_name, weight_tensor.shape);
        shaper_.Conv(input_name, strides[1], strides[0], 1, 1, pads[2], pads[3],
                     pads[0], pads[1], weight_name, output_name);
        nnapi_tensors_[weight_name] = weight_tensor;

        const auto param = DNN::CreateConv2DDirect(
            builder_, m(input_name).c_str(), weight_name.c_str(),
            bias_name ? bias_name.value().c_str() : nullptr, &pads, &strides,
            ConvertFuseCodeType(activation.second), output_name.c_str());
        layer = DNN::CreateLayer(builder_, DNN::LayerType::Conv2D, param);
    } else if (onnx_weight.shape[1] == 1) {  // depthwise
        LOG(INFO) << "Depthwise conv";
        weight_tensor = OnnxToNnapiDwConvWeight(onnx_weight);
        weight_name = weight_tensor.name;
        shaper_.AddShape(weight_name, weight_tensor.shape);
        shaper_.DepthwiseConv(input_name, strides[1], strides[0], 1, 1, pads[2],
                              pads[3], pads[0], pads[1], weight_name,
                              output_name);
        nnapi_tensors_[weight_name] = weight_tensor;
        const auto multiplier = nnapi_tensors_.at(weight_name).shape[3] / group;
        const auto param = DNN::CreateDepthwiseConv2DDirect(
            builder_, m(input_name).c_str(), weight_name.c_str(),
            bias_name ? bias_name.value().c_str() : nullptr, &pads, &strides,
            multiplier, ConvertFuseCodeType(activation.second),
            output_name.c_str());
        layer = DNN::CreateLayer(builder_, DNN::LayerType::DepthwiseConv2D, 0,
                                 0, 0, 0, 0, 0, 0, 0, param);
    } else {
        // TODO: Support it
        throw std::invalid_argument("group != 1 is not supported");
    }
    if (weight_tensor.data_type == Tensor::DataType::FLOAT32) {
        CreateTensorFb(weight_name, weight_tensor, DNN::DataType::Float32);
    } else if (weight_tensor.data_type == Tensor::DataType::UINT8) {
        const auto quant_info = quant_infos_.at(weight_tensor.name);
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
        CreateTensorFb(weight_name, weight_tensor, daq_data_type);
    } else {
        DNN_ASSERT(false, "Unknown data type of tensor");
    }
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerPool(css &op, css &input_name,
                                 const std::vector<int> &kernel_shape,
                                 const std::vector<int> &pads,
                                 const std::vector<int> &strides,
                                 css &output_name) {
    const auto activation = FindActivation(model_proto_, output_name);
    if (activation.first.has_value()) {
        skipped_act_.push_back(activation.first.value());
        name_map_[activation.first.value()] = output_name;
    }
    shaper_.Pool(input_name, strides[1], strides[0], pads[2], pads[3], pads[0],
                 pads[1], kernel_shape[0], kernel_shape[1], output_name);
    flatbuffers::Offset<DNN::Layer> layer;
    if (op == "AveragePool" || op == "GlobalAveragePool") {
        const auto param = DNN::CreateAvePoolDirect(
            builder_, m(input_name).c_str(), &kernel_shape, &pads, &strides,
            ConvertFuseCodeType(activation.second), output_name.c_str());
        layer = DNN::CreateLayer(builder_, DNN::LayerType::AvePool, 0, param);
    } else {
        const auto param = DNN::CreateMaxPoolDirect(
            builder_, m(input_name).c_str(), &kernel_shape, &pads, &strides,
            ConvertFuseCodeType(activation.second), output_name.c_str());
        layer =
            DNN::CreateLayer(builder_, DNN::LayerType::MaxPool, 0, 0, param);
    }
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerRelu(css &input_name, css &output_name) {
    shaper_.Relu(input_name, output_name);
    const auto param = DNN::CreateReluDirect(builder_, m(input_name).c_str(),
                                             output_name.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Relu, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerAdd(css &input1_name, css &input2_name,
                                css &output_name) {
    shaper_.Eltwise(input1_name, output_name);
    const auto activation = FindActivation(model_proto_, output_name);
    if (activation.first.has_value()) {
        skipped_act_.push_back(activation.first.value());
        name_map_[activation.first.value()] = output_name;
    }
    const auto param = DNN::CreateAddDirect(
        builder_, m(input1_name).c_str(), m(input2_name).c_str(),
        ConvertFuseCodeType(activation.second), output_name.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Add, 0, 0, 0,
                                        0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerAdd(css &input1_name, float input2,
                                css &output_name) {
    shaper_.Eltwise(input1_name, output_name);
    const auto activation = FindActivation(model_proto_, output_name);
    if (activation.first.has_value()) {
        skipped_act_.push_back(activation.first.value());
        name_map_[activation.first.value()] = output_name;
    }
    const auto param = DNN::CreateAddScalarDirect(
        builder_, m(input1_name).c_str(), input2,
        ConvertFuseCodeType(activation.second), output_name.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::AddScalar, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, param, 0);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerMul(css &input1_name, css &input2_name,
                                css &output_name) {
    shaper_.Eltwise(input1_name, output_name);
    const auto activation = FindActivation(model_proto_, output_name);
    if (activation.first.has_value()) {
        skipped_act_.push_back(activation.first.value());
        name_map_[activation.first.value()] = output_name;
    }
    const auto param = DNN::CreateMulDirect(
        builder_, m(input1_name).c_str(), m(input2_name).c_str(),
        ConvertFuseCodeType(activation.second), output_name.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Mul, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerMul(css &input1_name, float input2,
                                css &output_name) {
    shaper_.Eltwise(input1_name, output_name);
    const auto activation = FindActivation(model_proto_, output_name);
    if (activation.first.has_value()) {
        skipped_act_.push_back(activation.first.value());
        name_map_[activation.first.value()] = output_name;
    }
    const auto param = DNN::CreateMulScalarDirect(
        builder_, m(input1_name).c_str(), input2,
        ConvertFuseCodeType(activation.second), output_name.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::MulScalar, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerGemm(css &input_name, css &weight_name,
                                 nonstd::optional<std::string> bias_name,
                                 const int transA, const int transB,
                                 const float alpha, const float beta,
                                 css &output_name) {
    if (transA == 0 && transB == 1 && alpha == 1.f && beta == 1.f) {
        {
            nnapi_tensors_[weight_name] = onnx_tensors_.at(weight_name);
            const auto &weight_tensor = nnapi_tensors_[weight_name];
            shaper_.AddShape(weight_name, weight_tensor.shape);
            CreateTensorFb(weight_name, weight_tensor, DNN::DataType::Float32);
        }
        if (bias_name.has_value()) {
            nnapi_tensors_[bias_name.value()] =
                onnx_tensors_.at(bias_name.value());
            const auto &bias_tensor = nnapi_tensors_[bias_name.value()];
            CreateTensorFb(bias_name.value(), bias_tensor, DNN::DataType::Float32);
        }
        const auto activation = FindActivation(model_proto_, output_name);
        if (activation.first.has_value()) {
            skipped_act_.push_back(activation.first.value());
            name_map_[activation.first.value()] = output_name;
        }
        shaper_.FC(input_name, weight_name, output_name);
        const auto param = DNN::CreateFCDirect(
            builder_, m(input_name).c_str(), weight_name.c_str(),
            bias_name.has_value() ? bias_name.value().c_str() : nullptr,
            ConvertFuseCodeType(activation.second), output_name.c_str());
        const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::FC, 0, 0,
                                            0, 0, 0, param, 0);
        layers_.push_back(layer);
    } else {
        throw std::invalid_argument(
            "Only transA == 0, transB == 1, alpha == 1.0 and beta == 1.0 is "
            "supported.");
    }
}

void OnnxConverter::AddLayerSoftmax(css &input_name, css &output_name) {
    shaper_.Softmax(input_name, output_name);
    // simply ignore attribute "axis", because nnapi softmax didn't has this
    // attr, and we will check the equality of the two ops in DaqReader.cpp
    const auto param = DNN::CreateSoftmaxDirect(builder_, m(input_name).c_str(),
                                                output_name.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Softmax, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

// axis here is for onnx nchw
void OnnxConverter::AddLayerConcat(const std::vector<std::string> &inputs,
                                   css &output_name, const int axis) {
    const auto concat_inputs = FbStrVector(inputs);
    DNN_ASSERT(axis < 4, axis);
    const uint32_t axis_nchw_to_nhwc[4]{0, 3, 1, 2};
    shaper_.Concat(inputs, axis, output_name);
    const auto param = DNN::CreateConcatDirect(
        builder_, &concat_inputs, axis_nchw_to_nhwc[axis], output_name.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Concat, 0, 0,
                                        0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerDequantize(css &input_name, css &output_name) {
    shaper_.Eltwise(input_name, output_name);
    const auto param = DNN::CreateDequantizeDirect(
        builder_, m(input_name).c_str(), output_name.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Dequantize, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
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
            PNT(tensor.name(), tensor.data_type());
            DNN_ASSERT(false, "");
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
        const Shape nnapi_shape{shape[0], shape[2], shape[3], shape[1]};
        shaper_.AddShape(input.name(), nnapi_shape);
        const auto flat_input = DNN::CreateInputDirect(builder_, &nnapi_shape,
                                                       input.name().c_str());
        inputs.push_back(flat_input);
    }

    return inputs;
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
        nonstd::optional<int> zero_point;

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

void OnnxConverter::Convert(const ONNX_NAMESPACE::ModelProto &model_proto,
                            const std::string &filepath,
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
    for (const auto &node : model_proto_.graph().node()) {
        NodeAttrHelper helper(node);
        const auto &op = node.op_type();
        LOG(INFO) << "Node " << node.name();
        if (std::find(skipped_act_.begin(), skipped_act_.end(), node.name()) !=
            skipped_act_.end()) {
            LOG(INFO) << "Skip layer " << node.name();
            continue;
        }
        if (has_reshape && op != "Gemm") {
            throw std::invalid_argument(
                "Reshape can only be the last layer or precede a gemm layer "
                "for now");
        }
        if (op == "Conv") {
            LOG(INFO) << "Start converting Conv";
            const auto strides = helper.get("strides", vector<int>{1, 1});
            const auto pads = helper.get("pads", vector<int>{0, 0, 0, 0});
            const auto dilations = helper.get("dilations", vector<int>{1, 1});
            CHECK_EQ(pads.size(), 4ul);
            CHECK_EQ(strides.size(), 2ul);
            CHECK_EQ(dilations.size(), 2ul);
            const auto group = helper.get("group", 1);
            const auto activation =
                FindActivation(model_proto_, node.output(0));
            if (activation.first.has_value()) {
                skipped_act_.push_back(activation.first.value());
                name_map_[activation.first.value()] = node.name();
            }
            nonstd::optional<string> bias_name;
            if (node.input_size() >= 3) {
                const auto ori_bias_name = m(node.input(2));
                bias_name = ori_bias_name + "_conv_b";
                nnapi_tensors_[bias_name.value()] =
                    onnx_tensors_.at(ori_bias_name);
                flatbuffers::Offset<DNN::Tensor> flat_tensor;
                const auto bias_name_str = bias_name.value();
                const auto &bias_tensor = nnapi_tensors_[bias_name_str];
                if (bias_tensor.data_type == Tensor::DataType::FLOAT32) {
                    CreateTensorFb(bias_name_str, bias_tensor, DNN::DataType::Float32);
                } else if (bias_tensor.data_type == Tensor::DataType::INT32) {
                    CreateTensorFb(bias_name_str, bias_tensor, DNN::DataType::Int32);
                } else {
                    std::invalid_argument("Unknown data type");
                }
            }

            const auto ori_weight_name = m(node.input(1));
            AddConv(m(node.input(0)), strides, pads, dilations, group,
                    activation, ori_weight_name, bias_name, m(node.output(0)));
            LOG(INFO) << "Converting Conv completed";
        } else if (op == "AveragePool" || op == "MaxPool" ||
                   op == "GlobalAveragePool" || op == "GlobalMaxPool") {
            LOG(INFO) << "Start converting Pool";
            const auto input_name = m(node.input(0));
            const auto output_name = m(node.output(0));
            vector<int> strides, pads, kernel_shape;
            if (op == "AveragePool" || op == "MaxPool") {
                strides = helper.get("strides", vector<int>{1, 1});
                pads = helper.get("pads", vector<int>{0, 0, 0, 0});
                kernel_shape = helper.get("kernel_shape", vector<int>{0, 0});
                const auto count_include_pad =
                    helper.get("count_include_pad", 0);
                if (count_include_pad == 1) {
                    throw std::invalid_argument(
                        "count_include_pad == 1 is not supported");
                }
                const auto storage_order = helper.get("storage_order", 0);
                if (storage_order == 1) {
                    throw std::invalid_argument(
                        "storage_order == 1 is not supported");
                }
                if (helper.has_attr("auto_pad")) {
                    throw std::invalid_argument("auto_pad is not supported");
                }
            } else {
                strides = {0, 0};
                pads = {0, 0, 0, 0};
                kernel_shape = {-1, -1};  // -1 for global
            }
            CHECK_EQ(pads.size(), 4ul);
            CHECK_EQ(kernel_shape.size(), 2ul);
            CHECK_EQ(strides.size(), 2ul);
            AddLayerPool(op, input_name, kernel_shape, pads, strides,
                         output_name);
            LOG(INFO) << "Converting Pool completed";
        } else if (op == "Relu") {
            LOG(INFO) << "Start converting Relu";
            const auto input_name = m(node.input(0));
            const auto output_name = m(node.output(0));
            AddLayerRelu(input_name, output_name);
            LOG(INFO) << "Converting Relu completed";

        } else if (op == "PRelu") {
            LOG(INFO) << "Start converting PRelu";
            const auto input_name = m(node.input(0));
            const auto slope_name = m(node.input(1));
            const auto output_name = m(node.output(0));
            const auto imm1_name = output_name + "_imm1";
            const auto imm2_name = output_name + "_imm2";
            const auto imm3_name = output_name + "_imm3";
            const auto imm4_name = output_name + "_imm4";
            if (onnx_tensors_[slope_name].shape != Shape{1}) {
                // TODO: support it
                throw std::invalid_argument("Only support one element slope.");
            }
            AddLayerRelu(input_name, imm1_name);
            AddLayerMul(input_name, -onnx_tensors_[slope_name].data[0],
                        imm2_name);
            AddLayerRelu(imm2_name, imm3_name);
            AddLayerMul(imm3_name, -1.f, imm4_name);
            AddLayerAdd(imm1_name, imm4_name, output_name);
            // TODO:
            LOG(INFO) << "Converting PRelu completed";
        } else if (op == "Add") {
            LOG(INFO) << "Start converting Add";
            const auto input1_name = m(node.input(0));
            const auto input2_name = m(node.input(1));
            const auto output_name = m(node.output(0));
            AddLayerAdd(input1_name, input2_name, output_name);
            LOG(INFO) << "Converting Add completed";
        } else if (op == "Mul") {
            LOG(INFO) << "Start converting Mul";
            const auto input1_name = m(node.input(0));
            const auto input2_name = m(node.input(1));
            const auto output_name = m(node.output(0));
            AddLayerMul(input1_name, input2_name, output_name);
            LOG(INFO) << "Converting Mul completed";
        } else if (op == "Gemm") {
            LOG(INFO) << "Start converting Gemm";
            const auto input_name = m(node.input(0));
            const auto weight_name = m(node.input(1));
            const auto output_name = m(node.output(0));
            nonstd::optional<string> bias_name;
            if (node.input_size() >= 3) {
                bias_name = m(node.input(2));
            }
            const auto transA = helper.get("transA", 0);
            const auto transB = helper.get("transB", 0);
            const auto alpha = helper.get("alpha", 1.0f);
            const auto beta = helper.get("beta", 1.0f);
            AddLayerGemm(input_name, weight_name, bias_name, transA, transB,
                         alpha, beta, output_name);
            has_reshape = false;
            LOG(INFO) << "Converting Gemm completed";
        } else if (op == "Softmax") {
            LOG(INFO) << "Start converting Softmax";
            const auto input_name = m(node.input(0));
            const auto output_name = m(node.output(0));
            AddLayerSoftmax(input_name, output_name);
            LOG(INFO) << "Converting Softmax completed";
        } else if (op == "Concat") {
            LOG(INFO) << "Start converting Concat";
            vector<std::string> concat_inputs_str;
            for (const auto &onnx_input : node.input()) {
                concat_inputs_str.push_back(m(onnx_input));
            }
            const auto axis = helper.get("axis", 1);
            const auto output_name = m(node.output(0));
            AddLayerConcat(concat_inputs_str, output_name, axis);
            LOG(INFO) << "Converting Concat completed";
        } else if (op == "Dropout") {
            LOG(INFO) << "Start converting Dropout";
            SetIdentity(node.input(0), node.output(0));
            LOG(INFO) << "Converting Dropout completed";

        } else if (op == "BatchNormalization") {
            LOG(INFO) << "Start converting BatchNormalization";
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

            vector<float> a, b;
            FORZ(i, scale_tensor.shape[0]) {
                a.push_back(scale_tensor.float_data()[i] /
                            sqrt(var_tensor.float_data()[i] + eps));
                b.push_back((scale_tensor.float_data()[i] *
                             -mean_tensor.float_data()[i]) /
                                sqrt(var_tensor.float_data()[i] + eps) +
                            bias_tensor.float_data()[i]);
            }

            const auto flat_tensor_a = DNN::CreateTensorDirect(
                builder_, DNN::DataType::Float32, nullptr, &a,
                &scale_tensor.shape, (output_name + "_imm_a").c_str());
            const auto flat_tensor_b = DNN::CreateTensorDirect(
                builder_, DNN::DataType::Float32, nullptr, &b,
                &scale_tensor.shape, (output_name + "_imm_b").c_str());
            tensors_.push_back(flat_tensor_a);
            tensors_.push_back(flat_tensor_b);
            AddLayerMul(input_name, output_name + "_imm_a",
                        output_name + "_imm_mul");
            AddLayerAdd(output_name + "_imm_mul", output_name + "_imm_b",
                        output_name);

            LOG(INFO) << "Converting BatchNormalization completed";
        } else if (op == "Reshape") {
            LOG(INFO) << "Start converting Reshape";
            has_reshape = true;
            SetIdentity(node.input(0), node.output(0));
            LOG(INFO) << "Converting Reshape completed";
        } else {
            throw std::invalid_argument("Unsupported operator " + op);
        }
        FORZ(i, node.output_size()) {
            const auto output = node.output(i);
            if (std::find(dequantize_after_.begin(), dequantize_after_.end(),
                          output) != dequantize_after_.end()) {
                css dequant_output = output + "_dequant";
                AddLayerDequantize(output, dequant_output);
                name_map_[output] = dequant_output;
            }
        }
    }
    const auto flat_layers = builder_.CreateVector(layers_);
    const auto flat_inputs = builder_.CreateVector(inputs);
    const auto flat_tensors = builder_.CreateVector(tensors_);
    const auto flat_quant_infos =
        builder_.CreateVector(ConvertQuantInfosToFbs());
    const auto flat_model = DNN::CreateModel(
        builder_, flat_layers, flat_tensors, flat_inputs, flat_quant_infos);

    builder_.Finish(flat_model);

    LOG(INFO) << "Shapes: ";
    LOG(INFO) << shaper_;

    std::ofstream ofs(filepath);
    ofs.write(reinterpret_cast<char *>(builder_.GetBufferPointer()),
              builder_.GetSize());
    ofs.close();

    skipped_act_.clear();
    layers_.clear();
    operands_.clear();
    tensors_.clear();
    name_map_.clear();
    nnapi_tensors_.clear();
    onnx_tensors_.clear();
    shaper_.Clear();
}
