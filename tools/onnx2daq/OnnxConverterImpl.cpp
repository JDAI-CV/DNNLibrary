#include <common/helper.h>
#include <tools/onnx2daq/OnnxConverter.h>

#include "NodeAttrHelper.h"

using std::string;
using std::vector;
using Shape = Shaper::Shape;

namespace dnn {
// void OnnxConverter::AddConv(const string &input_name,
//                             const string &ori_weight_name,
//                             const dnn::optional<std::string> &bias_name,
//                             const int32_t padding_left,
//                             const int32_t padding_right,
//                             const int32_t padding_top,
//                             const int32_t padding_bottom,
//                             const int32_t stride_width,
//                             const int32_t stride_height,
//                             const int32_t dilation_width,
//                             const int32_t dilation_height,
//                             int group,
//                             const string &output_name) {
//     flatbuffers::Offset<DNN::Layer> layer;
//     if (dilation_width != 1 || dilation_height != 1) {
//         if (stride_width != 1 || stride_height != 1) {
//             throw std::invalid_argument(
//                 "Both dilations and strides > 1 is not supported for now");
//         }
//         if (!(padding_left == padding_right && padding_right == padding_top
//         && padding_top == padding_bottom)) {
//             throw std::invalid_argument(
//                 "Both dilations and asymmetric pads is not supported for
//                 now");
//         }
//         VLOG(5) << "Dilations of conv: " << dilation_width << ", " <<
//         dilation_height << ", converting.."; const auto s2b_name = input_name
//         + "_s2b"; const auto im_name = input_name + "_conv_imm"; const auto
//         b2s_name = input_name + "_b2s"; std::vector<int> new_pads = pads;
//         const auto input_shape = shaper_[input_name];
//         new_pads[1] = (input_shape[2] + pads[1] + (dilations[0] - 1)) /
//                           dilations[0] * dilations[0] -
//                       input_shape[2];
//         new_pads[3] = (input_shape[1] + pads[3] + (dilations[1] - 1)) /
//                           dilations[1] * dilations[1] -
//                       input_shape[1];
//         VLOG(5) << input_shape << ", " << pads << ", " << dilations << ", "
//                 << new_pads;
//         // Why "AllowShortBlocksOnASingleLine: false" doesn't work on it?
//         // clang-format off
//         {
//             AddLayerSPACE_TO_BATCH_ND(input_name, {dilation_height,
//             dilation_width}, new_pads, s2b_name);
//         }
//         // clang-format on
//         {
//             // paddings are applied in spacetobatch
//             AddConv(s2b_name, strides, vector<int>{0, 0, 0, 0},
//                     vector<int>{1, 1}, group, ori_weight_name, bias_name,
//                     im_name);
//         }
//         // clang-format off
//         {
//             AddLayerBATCH_TO_SPACE_ND(im_name, dilations, b2s_name);
//         }
//         // clang-format on
//         {
//             const auto b2s_shape = shaper_[b2s_name];
//             const std::vector<int32_t> starts{0, 0, 0, 0};
//             const std::vector<int32_t> ends{
//                 static_cast<int32_t>(b2s_shape[0]),
//                 static_cast<int32_t>(b2s_shape[1]) - (new_pads[1] - pads[1]),
//                 static_cast<int32_t>(b2s_shape[2]) - (new_pads[3] - pads[3]),
//                 static_cast<int32_t>(b2s_shape[3])};
//             const std::vector<int32_t> strides_in_ss{1, 1, 1, 1};
//             const int32_t begin_mask = 0;
//             const int32_t end_mask = 0;
//             const int32_t shrink_axis_mask = 0;
//             AddLayerSTRIDED_SLICE(b2s_name, starts, ends, strides_in_ss,
//                                   begin_mask, end_mask, shrink_axis_mask,
//                                   output_name);
//         }
//         return;
//     }
//
//     if (!onnx_tensors_.has(ori_weight_name)) {
//         throw std::invalid_argument("The weight of convolution must be
//         known");
//     }
//     const auto &onnx_weight = onnx_tensors_.at(ori_weight_name);
//     if (group == 1) {
//         VLOG(5) << "Vanilla conv";
//         AddLayerConvImpl(input_name, ori_weight_name, bias_name, pads,
//         strides,
//                          output_name);
//     } else if (onnx_weight.shape[1] == 1) {  // depthwise
//         VLOG(5) << "Depthwise conv";
//         AddLayerDepthwiseConvImpl(input_name, ori_weight_name, bias_name,
//         pads,
//                                   strides, onnx_weight.shape[0] / group,
//                                   output_name);
//     } else {
//         // TODO: Support it
//         throw std::invalid_argument("group != 1 is not supported");
//     }
// }

// OnnxConverter auto generated methods start
void OnnxConverter::WriteDaqLayer_CONV_2D(
    const std::string &input, const std::string &weight,
    const dnn::optional<std::string> &bias, int32_t padding_left,
    int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
    int32_t stride_x, int32_t stride_y, const std::string &output) {
    const auto activation = FindActivation(model_proto_, output);

    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    {
        const auto name = weight;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    if (bias.has_value()) {
        const auto name = bias.value();

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiIdentity(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Conv(m(input), m(weight), padding_left, padding_right, padding_top,
                 padding_bottom, stride_x, stride_y, output);
    const auto param = DNN::CreateCONV_2DDirect(
        builder_, m(input).c_str(), m(weight).c_str(),
        bias.has_value() ? bias.value().c_str() : nullptr, padding_left,
        padding_right, padding_top, padding_bottom, stride_x, stride_y,
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::CONV_2D, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_AVERAGE_POOL_2D(
    const std::string &input, int32_t padding_left, int32_t padding_right,
    int32_t padding_top, int32_t padding_bottom, int32_t stride_x,
    int32_t stride_y, int32_t kernel_width, int32_t kernel_height,
    const std::string &output) {
    const auto activation = FindActivation(model_proto_, output);

    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Pool(m(input), padding_left, padding_right, padding_top,
                 padding_bottom, stride_x, stride_y, kernel_width,
                 kernel_height, output);
    const auto param = DNN::CreateAVERAGE_POOL_2DDirect(
        builder_, m(input).c_str(), padding_left, padding_right, padding_top,
        padding_bottom, stride_x, stride_y, kernel_width, kernel_height,
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::AVERAGE_POOL_2D, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_MAX_POOL_2D(
    const std::string &input, int32_t padding_left, int32_t padding_right,
    int32_t padding_top, int32_t padding_bottom, int32_t stride_x,
    int32_t stride_y, int32_t kernel_width, int32_t kernel_height,
    const std::string &output) {
    const auto activation = FindActivation(model_proto_, output);

    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Pool(m(input), padding_left, padding_right, padding_top,
                 padding_bottom, stride_x, stride_y, kernel_width,
                 kernel_height, output);
    const auto param = DNN::CreateMAX_POOL_2DDirect(
        builder_, m(input).c_str(), padding_left, padding_right, padding_top,
        padding_bottom, stride_x, stride_y, kernel_width, kernel_height,
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::MAX_POOL_2D, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_RELU(const std::string &input,
                                       const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Identity(m(input), output);
    const auto param =
        DNN::CreateRELUDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::RELU, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_SOFTMAX(const std::string &input, float beta,
                                          const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Identity(m(input), output);
    const auto param = DNN::CreateSOFTMAXDirect(builder_, m(input).c_str(),
                                                beta, output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::SOFTMAX, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_FULLY_CONNECTED(
    const std::string &input, const std::string &weight,
    const dnn::optional<std::string> &bias, const std::string &output) {
    const auto activation = FindActivation(model_proto_, output);

    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    {
        const auto name = weight;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiIdentity(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    if (bias.has_value()) {
        const auto name = bias.value();

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiIdentity(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.FC(m(input), m(weight), output);
    const auto param = DNN::CreateFULLY_CONNECTEDDirect(
        builder_, m(input).c_str(), m(weight).c_str(),
        bias.has_value() ? bias.value().c_str() : nullptr,
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer = DNN::CreateLayer(
        builder_, DNN::LayerType::FULLY_CONNECTED, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_ADD(const std::string &input1,
                                      const std::string &input2,
                                      const std::string &output) {
    const auto activation = FindActivation(model_proto_, output);

    {
        const auto name = input1;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    {
        const auto name = input2;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Eltwise(m(input1), m(input2), output);
    const auto param = DNN::CreateADDDirect(
        builder_, m(input1).c_str(), m(input2).c_str(),
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::ADD, 0, 0, 0,
                                        0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_CONCATENATION(
    const std::vector<std::string> &inputs, int32_t axis,
    const std::string &output) {
    for (const auto &name : inputs) {
        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    const auto inputs_fb = FbStrVector(inputs);
    shaper_.Concat(inputs, axis, output);
    const auto param = DNN::CreateCONCATENATIONDirect(builder_, &inputs_fb,
                                                      axis, output.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::CONCATENATION,
                                        0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_DEPTHWISE_CONV_2D(
    const std::string &input, const std::string &weight,
    const dnn::optional<std::string> &bias, int32_t padding_left,
    int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
    int32_t stride_x, int32_t stride_y, int32_t depth_multiplier,
    const std::string &output) {
    const auto activation = FindActivation(model_proto_, output);

    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    {
        const auto name = weight;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes1230(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    if (bias.has_value()) {
        const auto name = bias.value();

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiIdentity(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.DepthwiseConv(m(input), m(weight), padding_left, padding_right,
                          padding_top, padding_bottom, stride_x, stride_y,
                          output);
    const auto param = DNN::CreateDEPTHWISE_CONV_2DDirect(
        builder_, m(input).c_str(), m(weight).c_str(),
        bias.has_value() ? bias.value().c_str() : nullptr, padding_left,
        padding_right, padding_top, padding_bottom, stride_x, stride_y,
        depth_multiplier, ConvertFuseCodeType(activation.second),
        output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::DEPTHWISE_CONV_2D, 0, 0, 0,
                         0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_BATCH_TO_SPACE_ND(
    const std::string &input, const std::vector<int32_t> &block_sizes,
    const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.BatchToSpace(m(input), block_sizes, output);
    const auto param = DNN::CreateBATCH_TO_SPACE_NDDirect(
        builder_, m(input).c_str(), &block_sizes, output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::BATCH_TO_SPACE_ND, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_SPACE_TO_BATCH_ND(
    const std::string &input, const std::vector<int32_t> &block_sizes,
    const std::vector<int32_t> &pads, const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.SpaceToBatch(m(input), block_sizes, pads, output);
    const auto param = DNN::CreateSPACE_TO_BATCH_NDDirect(
        builder_, m(input).c_str(), &block_sizes, &pads, output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::SPACE_TO_BATCH_ND, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_STRIDED_SLICE(
    const std::string &input, const std::vector<int32_t> &starts,
    const std::vector<int32_t> &ends, const std::vector<int32_t> &strides,
    int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask,
    const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.StridedSlice(m(input), starts, ends, strides, begin_mask, end_mask,
                         shrink_axis_mask, output);
    const auto param = DNN::CreateSTRIDED_SLICEDirect(
        builder_, m(input).c_str(), &starts, &ends, &strides, begin_mask,
        end_mask, shrink_axis_mask, output.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::STRIDED_SLICE,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_MUL(const std::string &input1,
                                      const std::string &input2,
                                      const std::string &output) {
    const auto activation = FindActivation(model_proto_, output);

    {
        const auto name = input1;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    {
        const auto name = input2;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Eltwise(m(input1), m(input2), output);
    const auto param = DNN::CreateMULDirect(
        builder_, m(input1).c_str(), m(input2).c_str(),
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::MUL, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_DEQUANTIZE(const std::string &input,
                                             const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Identity(m(input), output);
    const auto param =
        DNN::CreateDEQUANTIZEDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::DEQUANTIZE, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_LOCAL_RESPONSE_NORMALIZATION(
    const std::string &input, int32_t radius, float bias, float alpha,
    float beta, const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Identity(m(input), output);
    const auto param = DNN::CreateLOCAL_RESPONSE_NORMALIZATIONDirect(
        builder_, m(input).c_str(), radius, bias, alpha, beta, output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::LOCAL_RESPONSE_NORMALIZATION,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_TANH(const std::string &input,
                                       const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Identity(m(input), output);
    const auto param =
        DNN::CreateTANHDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::TANH, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_FLOOR(const std::string &input,
                                        const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Identity(m(input), output);
    const auto param =
        DNN::CreateFLOORDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::FLOOR, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_LOGISTIC(const std::string &input,
                                           const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Identity(m(input), output);
    const auto param =
        DNN::CreateLOGISTICDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::LOGISTIC, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_PRELU(const std::string &input,
                                        const std::string &alpha,
                                        const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Identity(m(input), output);
    const auto param = DNN::CreatePRELUDirect(builder_, m(input).c_str(),
                                              m(alpha).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::PRELU, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_POW(const std::string &input,
                                      const std::string &exp,
                                      const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Identity(m(input), output);
    const auto param = DNN::CreatePOWDirect(builder_, m(input).c_str(),
                                            m(exp).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::POW, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_NEG(const std::string &input,
                                      const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Identity(m(input), output);
    const auto param =
        DNN::CreateNEGDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::NEG, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_MINIMUM(const std::string &input1,
                                          const std::string &input2,
                                          const std::string &output) {
    {
        const auto name = input1;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    {
        const auto name = input2;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Eltwise(m(input1), m(input2), output);
    const auto param = DNN::CreateMINIMUMDirect(
        builder_, m(input1).c_str(), m(input2).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::MINIMUM, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_MAXIMUM(const std::string &input1,
                                          const std::string &input2,
                                          const std::string &output) {
    {
        const auto name = input1;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    {
        const auto name = input2;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Eltwise(m(input1), m(input2), output);
    const auto param = DNN::CreateMAXIMUMDirect(
        builder_, m(input1).c_str(), m(input2).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::MAXIMUM, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_LOG(const std::string &input,
                                      const std::string &output) {
    {
        const auto name = input;

        if (onnx_tensors_.has(name)) {
            const auto &onnx_tensor = onnx_tensors_.at(name);
            const auto new_tensor = OnnxToNnapiAxes0231(onnx_tensor);
            shaper_.AddShape(name, new_tensor.shape);
            nnapi_tensors_[name] = new_tensor;
            CreateTensorFb(name, new_tensor);
        }
    }

    shaper_.Identity(m(input), output);
    const auto param =
        DNN::CreateLOGDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::LOG, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

// OnnxConverter auto generated methods end

}  // namespace dnn
