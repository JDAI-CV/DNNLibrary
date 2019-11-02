#include <tools/onnx2daq/OnnxConverter.h>

#include <common/helper.h>
#include "NodeAttrHelper.h"

using std::string;
using std::vector;
using Shape = Shaper::Shape;

namespace dnn {
void OnnxConverter::AddConv(const string &input_name,
                            // Strides here are in the order: width, height
                            const std::vector<int> &strides,
                            // The order of pads here is the same as nnapi:
                            // left, right, top, bottom
                            const std::vector<int> &pads,
                            // Dilations here are in the order: width, height
                            const std::vector<int> &dilations, int group,
                            const string &ori_weight_name,
                            const dnn::optional<std::string> &bias_name,
                            const string &output_name) {
    flatbuffers::Offset<DNN::Layer> layer;
    if (dilations != vector<int>{1, 1}) {
        if (strides != vector<int>{1, 1}) {
            throw std::invalid_argument(
                "Both dilations and strides > 1 is not supported for now");
        }
        if (!(pads[0] == pads[1] && pads[1] == pads[2] && pads[2] == pads[3])) {
            throw std::invalid_argument(
                "Both dilations and asymmetric pads is not supported for now");
        }
        VLOG(5) << "Dilations of conv: " << dilations << ", converting..";
        const auto s2b_name = input_name + "_s2b";
        const auto im_name = input_name + "_conv_imm";
        const auto b2s_name = input_name + "_b2s";
        std::vector<int> new_pads = pads;
        const auto input_shape = shaper_[input_name];
        new_pads[1] = (input_shape[2] + pads[1] + (dilations[0] - 1)) /
                          dilations[0] * dilations[0] -
                      input_shape[2];
        new_pads[3] = (input_shape[1] + pads[3] + (dilations[1] - 1)) /
                          dilations[1] * dilations[1] -
                      input_shape[1];
        VLOG(5) << input_shape << ", " << pads << ", " << dilations << ", "
                << new_pads;
        // Why "AllowShortBlocksOnASingleLine: false" doesn't work on it?
        // clang-format off
        {
            AddLayerSpaceToBatchND(input_name, dilations, new_pads, s2b_name);
        }
        // clang-format on
        {
            // paddings are applied in spacetobatch
            AddConv(s2b_name, strides, vector<int>{0, 0, 0, 0},
                    vector<int>{1, 1}, group, ori_weight_name, bias_name,
                    im_name);
        }
        // clang-format off
        {
            AddLayerBatchToSpaceND(im_name, dilations, b2s_name);
        }
        // clang-format on
        {
            const auto b2s_shape = shaper_[b2s_name];
            const std::vector<int32_t> starts{0, 0, 0, 0};
            const std::vector<int32_t> ends{
                static_cast<int32_t>(b2s_shape[0]),
                static_cast<int32_t>(b2s_shape[1]) - (new_pads[1] - pads[1]),
                static_cast<int32_t>(b2s_shape[2]) - (new_pads[3] - pads[3]),
                static_cast<int32_t>(b2s_shape[3])};
            const std::vector<int32_t> strides_in_ss{1, 1, 1, 1};
            const int32_t begin_mask = 0;
            const int32_t end_mask = 0;
            const int32_t shrink_axis_mask = 0;
            AddLayerStridedSlice(b2s_name, starts, ends, strides_in_ss,
                                 begin_mask, end_mask, shrink_axis_mask,
                                 output_name);
        }
        return;
    }

    if (!onnx_tensors_.has(ori_weight_name)) {
        throw std::invalid_argument("The weight of convolution must be known");
    }
    const auto &onnx_weight = onnx_tensors_.at(ori_weight_name);
    if (group == 1) {
        VLOG(5) << "Vanilla conv";
        AddLayerConvImpl(input_name, ori_weight_name, bias_name, pads, strides,
                         output_name);
    } else if (onnx_weight.shape[1] == 1) {  // depthwise
        VLOG(5) << "Depthwise conv";
        AddLayerDepthwiseConvImpl(input_name, ori_weight_name, bias_name, pads,
                                  strides, onnx_weight.shape[0] / group,
                                  output_name);
    } else {
        // TODO: Support it
        throw std::invalid_argument("group != 1 is not supported");
    }
}

void OnnxConverter::AddLayerPool(css &op, css &input_name,
                                 const std::vector<int> &kernel_shape,
                                // The order of pads here is the same as nnapi:
                                // left, right, top, bottom
                                 const std::vector<int> &pads,
                                // Strides here are in the order: width, height
                                 const std::vector<int> &strides,
                                 css &output_name) {
    if (op == "AveragePool" || op == "GlobalAveragePool") {
        AddLayerAvePoolImpl(input_name, kernel_shape, pads, strides,
                            output_name);
    } else {
        AddLayerMaxPoolImpl(input_name, kernel_shape, pads, strides,
                            output_name);
    }
}

// OnnxConverter auto generated methods start
void OnnxConverter::AddLayerConvImpl(const std::string &input,
                                     const std::string &weight,
                                     const dnn::optional<std::string> &bias,
                                    // The order of pads here is the same as nnapi:
                                    // left, right, top, bottom
                                     const std::vector<int32_t> &pads,
                                    // Strides here are in the order: width, height
                                     const std::vector<int32_t> &strides,
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

    shaper_.Conv(m(input), m(weight), pads, strides, output);
    const auto param = DNN::CreateConv2DDirect(
        builder_, m(input).c_str(), m(weight).c_str(),
        bias.has_value() ? bias.value().c_str() : nullptr, &pads, &strides,
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Conv2D, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerAvePoolImpl(
    const std::string &input, const std::vector<int32_t> &kernel_shape,
    // The order of pads here is the same as nnapi:
    // left, right, top, bottom
    // Strides here are in the order: width, height
    const std::vector<int32_t> &pads, const std::vector<int32_t> &strides,
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

    shaper_.Pool(m(input), kernel_shape, pads, strides, output);
    const auto param = DNN::CreateAvePoolDirect(
        builder_, m(input).c_str(), &kernel_shape, &pads, &strides,
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::AvePool, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerMaxPoolImpl(
    const std::string &input, const std::vector<int32_t> &kernel_shape,
    // The order of pads here is the same as nnapi:
    // left, right, top, bottom
    // Strides here are in the order: width, height
    const std::vector<int32_t> &pads, const std::vector<int32_t> &strides,
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

    shaper_.Pool(m(input), kernel_shape, pads, strides, output);
    const auto param = DNN::CreateMaxPoolDirect(
        builder_, m(input).c_str(), &kernel_shape, &pads, &strides,
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::MaxPool, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerReLU(const std::string &input,
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

    shaper_.Relu(m(input), output);
    const auto param =
        DNN::CreateReluDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Relu, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerSoftmax(const std::string &input,
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

    shaper_.Softmax(m(input), output);
    const auto param =
        DNN::CreateSoftmaxDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Softmax, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerFC(const std::string &input,
                               const std::string &weight,
                               const dnn::optional<std::string> &bias,
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
    const auto param = DNN::CreateFCDirect(
        builder_, m(input).c_str(), m(weight).c_str(),
        bias.has_value() ? bias.value().c_str() : nullptr,
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::FC, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerAdd(const std::string &input1,
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
    const auto param = DNN::CreateAddDirect(
        builder_, m(input1).c_str(), m(input2).c_str(),
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Add, 0, 0, 0,
                                        0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerConcat(const std::vector<std::string> &inputs,
                                   int32_t axis, const std::string &output) {
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
    const auto param =
        DNN::CreateConcatDirect(builder_, &inputs_fb, axis, output.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Concat, 0, 0,
                                        0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerDepthwiseConvImpl(
    const std::string &input, const std::string &weight,
    // The order of pads here is the same as nnapi:
    // left, right, top, bottom
    const dnn::optional<std::string> &bias, const std::vector<int32_t> &pads,
    // Strides here are in the order: width, height
    const std::vector<int32_t> &strides, int32_t depth_multiplier,
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

    shaper_.DepthwiseConv(m(input), m(weight), pads, strides, output);
    const auto param = DNN::CreateDepthwiseConv2DDirect(
        builder_, m(input).c_str(), m(weight).c_str(),
        bias.has_value() ? bias.value().c_str() : nullptr, &pads, &strides,
        depth_multiplier, ConvertFuseCodeType(activation.second),
        output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::DepthwiseConv2D, 0, 0, 0, 0,
                         0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerBatchToSpaceND(
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
    const auto param = DNN::CreateBatchToSpaceDirect(
        builder_, m(input).c_str(), &block_sizes, output.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::BatchToSpace,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerSpaceToBatchND(
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
    const auto param = DNN::CreateSpaceToBatchDirect(
        builder_, m(input).c_str(), &block_sizes, &pads, output.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::SpaceToBatch,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerStridedSlice(const std::string &input,
                                         const std::vector<int32_t> &starts,
                                         const std::vector<int32_t> &ends,
                                         const std::vector<int32_t> &strides,
                                         int32_t begin_mask, int32_t end_mask,
                                         int32_t shrink_axis_mask,
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
    const auto param = DNN::CreateStridedSliceDirect(
        builder_, m(input).c_str(), &starts, &ends, &strides, begin_mask,
        end_mask, shrink_axis_mask, output.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::StridedSlice,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerMul(const std::string &input1,
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
    const auto param = DNN::CreateMulDirect(
        builder_, m(input1).c_str(), m(input2).c_str(),
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Mul, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerAdd(const std::string &input, float scalar,
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

    shaper_.Eltwise(m(input), output);
    const auto param = DNN::CreateAddScalarDirect(
        builder_, m(input).c_str(), scalar,
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::AddScalar, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerMul(const std::string &input, float scalar,
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

    shaper_.Eltwise(m(input), output);
    const auto param = DNN::CreateMulScalarDirect(
        builder_, m(input).c_str(), scalar,
        ConvertFuseCodeType(activation.second), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::MulScalar, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerDequantize(const std::string &input,
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
        DNN::CreateDequantizeDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Dequantize, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerLRN(const std::string &input, int32_t radius,
                                float bias, float alpha, float beta,
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
    const auto param = DNN::CreateLRNDirect(builder_, m(input).c_str(), radius,
                                            bias, alpha, beta, output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::LRN, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerTanh(const std::string &input,
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
        DNN::CreateTanhDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Tanh, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerFloor(const std::string &input,
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
        DNN::CreateFloorDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Floor, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerLogistic(const std::string &input,
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
        DNN::CreateLogisticDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Logistic, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerPReLU(const std::string &input,
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
    const auto param = DNN::CreatePReLUDirect(builder_, m(input).c_str(),
                                              m(alpha).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::PReLU, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerPow(const std::string &input,
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
    const auto param = DNN::CreatePowDirect(builder_, m(input).c_str(),
                                            m(exp).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Pow, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerNeg(const std::string &input,
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
        DNN::CreateNegDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Neg, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerMinimum(const std::string &input1,
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
    const auto param = DNN::CreateMinimumDirect(
        builder_, m(input1).c_str(), m(input2).c_str(), output.c_str());
    const auto layer =
        DNN::CreateLayer(builder_, DNN::LayerType::Minimum, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerMaximum(const std::string &input1,
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
    const auto param = DNN::CreateMaximumDirect(
        builder_, m(input1).c_str(), m(input2).c_str(), output.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Maximum, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

void OnnxConverter::AddLayerLog(const std::string &input,
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
        DNN::CreateLogDirect(builder_, m(input).c_str(), output.c_str());
    const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Log, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, param);
    layers_.push_back(layer);
}

// OnnxConverter auto generated methods end

}  // namespace dnn
