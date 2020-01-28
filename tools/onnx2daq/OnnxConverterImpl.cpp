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
    int32_t stride_x, int32_t stride_y, FuseCode fuse_code, bool nchw,
    int32_t dilation_x, int32_t dilation_y, const std::string &output) {
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
                 padding_bottom, stride_x, stride_y, nchw, dilation_x,
                 dilation_y, output);
    const auto input_param = DNN::CreateCONV_2D_InputDirect(
        builder_, m(input).c_str(), m(weight).c_str(),
        bias.has_value() ? bias.value().c_str() : nullptr, padding_left,
        padding_right, padding_top, padding_bottom, stride_x, stride_y,
        ConvertFuseCodeType(fuse_code), nchw, dilation_x, dilation_y);
    const auto output_param =
        DNN::CreateCONV_2D_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateCONV_2D(builder_, input_param, output_param);
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::CONV_2D, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_AVERAGE_POOL_2D(
    const std::string &input, int32_t padding_left, int32_t padding_right,
    int32_t padding_top, int32_t padding_bottom, int32_t stride_x,
    int32_t stride_y, int32_t kernel_width, int32_t kernel_height,
    FuseCode fuse_code, const std::string &output) {
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
    const auto input_param = DNN::CreateAVERAGE_POOL_2D_InputDirect(
        builder_, m(input).c_str(), padding_left, padding_right, padding_top,
        padding_bottom, stride_x, stride_y, kernel_width, kernel_height,
        ConvertFuseCodeType(fuse_code));
    const auto output_param =
        DNN::CreateAVERAGE_POOL_2D_OutputDirect(builder_, output.c_str());
    const auto param =
        DNN::CreateAVERAGE_POOL_2D(builder_, input_param, output_param);
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::AVERAGE_POOL_2D, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_MAX_POOL_2D(
    const std::string &input, int32_t padding_left, int32_t padding_right,
    int32_t padding_top, int32_t padding_bottom, int32_t stride_x,
    int32_t stride_y, int32_t kernel_width, int32_t kernel_height,
    FuseCode fuse_code, const std::string &output) {
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
    const auto input_param = DNN::CreateMAX_POOL_2D_InputDirect(
        builder_, m(input).c_str(), padding_left, padding_right, padding_top,
        padding_bottom, stride_x, stride_y, kernel_width, kernel_height,
        ConvertFuseCodeType(fuse_code));
    const auto output_param =
        DNN::CreateMAX_POOL_2D_OutputDirect(builder_, output.c_str());
    const auto param =
        DNN::CreateMAX_POOL_2D(builder_, input_param, output_param);
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
    const auto input_param =
        DNN::CreateRELU_InputDirect(builder_, m(input).c_str());
    const auto output_param =
        DNN::CreateRELU_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateRELU(builder_, input_param, output_param);
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
    const auto input_param =
        DNN::CreateSOFTMAX_InputDirect(builder_, m(input).c_str(), beta);
    const auto output_param =
        DNN::CreateSOFTMAX_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateSOFTMAX(builder_, input_param, output_param);
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::SOFTMAX, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_FULLY_CONNECTED(
    const std::string &input, const std::string &weight,
    const dnn::optional<std::string> &bias, FuseCode fuse_code,
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
    const auto input_param = DNN::CreateFULLY_CONNECTED_InputDirect(
        builder_, m(input).c_str(), m(weight).c_str(),
        bias.has_value() ? bias.value().c_str() : nullptr,
        ConvertFuseCodeType(fuse_code));
    const auto output_param =
        DNN::CreateFULLY_CONNECTED_OutputDirect(builder_, output.c_str());
    const auto param =
        DNN::CreateFULLY_CONNECTED(builder_, input_param, output_param);
    const auto layer = DNN::CreateLayer(
        builder_, DNN::LayerType::FULLY_CONNECTED, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_ADD(const std::string &input1,
                                      const std::string &input2,
                                      FuseCode fuse_code,
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
    const auto input_param = DNN::CreateADD_InputDirect(
        builder_, m(input1).c_str(), m(input2).c_str(),
        ConvertFuseCodeType(fuse_code));
    const auto output_param =
        DNN::CreateADD_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateADD(builder_, input_param, output_param);
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
    const auto input_param =
        DNN::CreateCONCATENATION_InputDirect(builder_, &inputs_fb, axis);
    const auto output_param =
        DNN::CreateCONCATENATION_OutputDirect(builder_, output.c_str());
    const auto param =
        DNN::CreateCONCATENATION(builder_, input_param, output_param);
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::CONCATENATION,
                                        0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_DEPTHWISE_CONV_2D(
    const std::string &input, const std::string &weight,
    const dnn::optional<std::string> &bias, int32_t padding_left,
    int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
    int32_t stride_x, int32_t stride_y, int32_t depth_multiplier,
    FuseCode fuse_code, const std::string &output) {
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
    const auto input_param = DNN::CreateDEPTHWISE_CONV_2D_InputDirect(
        builder_, m(input).c_str(), m(weight).c_str(),
        bias.has_value() ? bias.value().c_str() : nullptr, padding_left,
        padding_right, padding_top, padding_bottom, stride_x, stride_y,
        depth_multiplier, ConvertFuseCodeType(fuse_code));
    const auto output_param =
        DNN::CreateDEPTHWISE_CONV_2D_OutputDirect(builder_, output.c_str());
    const auto param =
        DNN::CreateDEPTHWISE_CONV_2D(builder_, input_param, output_param);
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
    const auto input_param = DNN::CreateBATCH_TO_SPACE_ND_InputDirect(
        builder_, m(input).c_str(), &block_sizes);
    const auto output_param =
        DNN::CreateBATCH_TO_SPACE_ND_OutputDirect(builder_, output.c_str());
    const auto param =
        DNN::CreateBATCH_TO_SPACE_ND(builder_, input_param, output_param);
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
    const auto input_param = DNN::CreateSPACE_TO_BATCH_ND_InputDirect(
        builder_, m(input).c_str(), &block_sizes, &pads);
    const auto output_param =
        DNN::CreateSPACE_TO_BATCH_ND_OutputDirect(builder_, output.c_str());
    const auto param =
        DNN::CreateSPACE_TO_BATCH_ND(builder_, input_param, output_param);
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
    const auto input_param = DNN::CreateSTRIDED_SLICE_InputDirect(
        builder_, m(input).c_str(), &starts, &ends, &strides, begin_mask,
        end_mask, shrink_axis_mask);
    const auto output_param =
        DNN::CreateSTRIDED_SLICE_OutputDirect(builder_, output.c_str());
    const auto param =
        DNN::CreateSTRIDED_SLICE(builder_, input_param, output_param);
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::STRIDED_SLICE,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_MUL(const std::string &input1,
                                      const std::string &input2,
                                      FuseCode fuse_code,
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
    const auto input_param = DNN::CreateMUL_InputDirect(
        builder_, m(input1).c_str(), m(input2).c_str(),
        ConvertFuseCodeType(fuse_code));
    const auto output_param =
        DNN::CreateMUL_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateMUL(builder_, input_param, output_param);
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
    const auto input_param =
        DNN::CreateDEQUANTIZE_InputDirect(builder_, m(input).c_str());
    const auto output_param =
        DNN::CreateDEQUANTIZE_OutputDirect(builder_, output.c_str());
    const auto param =
        DNN::CreateDEQUANTIZE(builder_, input_param, output_param);
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
    const auto input_param =
        DNN::CreateLOCAL_RESPONSE_NORMALIZATION_InputDirect(
            builder_, m(input).c_str(), radius, bias, alpha, beta);
    const auto output_param =
        DNN::CreateLOCAL_RESPONSE_NORMALIZATION_OutputDirect(builder_,
                                                             output.c_str());
    const auto param = DNN::CreateLOCAL_RESPONSE_NORMALIZATION(
        builder_, input_param, output_param);
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
    const auto input_param =
        DNN::CreateTANH_InputDirect(builder_, m(input).c_str());
    const auto output_param =
        DNN::CreateTANH_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateTANH(builder_, input_param, output_param);
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
    const auto input_param =
        DNN::CreateFLOOR_InputDirect(builder_, m(input).c_str());
    const auto output_param =
        DNN::CreateFLOOR_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateFLOOR(builder_, input_param, output_param);
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
    const auto input_param =
        DNN::CreateLOGISTIC_InputDirect(builder_, m(input).c_str());
    const auto output_param =
        DNN::CreateLOGISTIC_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateLOGISTIC(builder_, input_param, output_param);
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
    const auto input_param = DNN::CreatePRELU_InputDirect(
        builder_, m(input).c_str(), m(alpha).c_str());
    const auto output_param =
        DNN::CreatePRELU_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreatePRELU(builder_, input_param, output_param);
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
    const auto input_param =
        DNN::CreatePOW_InputDirect(builder_, m(input).c_str(), m(exp).c_str());
    const auto output_param =
        DNN::CreatePOW_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreatePOW(builder_, input_param, output_param);
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
    const auto input_param =
        DNN::CreateNEG_InputDirect(builder_, m(input).c_str());
    const auto output_param =
        DNN::CreateNEG_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateNEG(builder_, input_param, output_param);
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
    const auto input_param = DNN::CreateMINIMUM_InputDirect(
        builder_, m(input1).c_str(), m(input2).c_str());
    const auto output_param =
        DNN::CreateMINIMUM_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateMINIMUM(builder_, input_param, output_param);
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
    const auto input_param = DNN::CreateMAXIMUM_InputDirect(
        builder_, m(input1).c_str(), m(input2).c_str());
    const auto output_param =
        DNN::CreateMAXIMUM_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateMAXIMUM(builder_, input_param, output_param);
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
    const auto input_param =
        DNN::CreateLOG_InputDirect(builder_, m(input).c_str());
    const auto output_param =
        DNN::CreateLOG_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateLOG(builder_, input_param, output_param);
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::LOG, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_ABS(const std::string &input,
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
    const auto input_param =
        DNN::CreateABS_InputDirect(builder_, m(input).c_str());
    const auto output_param =
        DNN::CreateABS_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateABS(builder_, input_param, output_param);
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::ABS, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_EXP(const std::string &input,
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
    const auto input_param =
        DNN::CreateEXP_InputDirect(builder_, m(input).c_str());
    const auto output_param =
        DNN::CreateEXP_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateEXP(builder_, input_param, output_param);
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::EXP, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::WriteDaqLayer_SUB(const std::string &input1,
                                      const std::string &input2,
                                      FuseCode fuse_code,
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
    const auto input_param = DNN::CreateSUB_InputDirect(
        builder_, m(input1).c_str(), m(input2).c_str(),
        ConvertFuseCodeType(fuse_code));
    const auto output_param =
        DNN::CreateSUB_OutputDirect(builder_, output.c_str());
    const auto param = DNN::CreateSUB(builder_, input_param, output_param);
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::SUB, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

// OnnxConverter auto generated methods end

}  // namespace dnn
