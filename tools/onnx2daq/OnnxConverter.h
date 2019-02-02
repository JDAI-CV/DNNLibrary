#include <onnx/onnx_pb.h>
#include <glog/logging.h>
#include <common/daq_generated.h>
#include <common/helper.h>
#include <common/StrKeyMap.h>
#include <common/Shaper.h>
#include "optional.h"

class OnnxConverter {
private:
    Shaper shaper_;

    struct Tensor {
        enum class DataType {
            FLOAT32,
            UINT8
        };
        std::vector<char> data;
        Shaper::Shape shape;
        DataType data_type;
        const std::vector<float> float_data() const {
            std::vector<float> float_vec(data.size() / 4);
            memcpy(&float_vec[0], &data[0], data.size());
            return float_vec;
        }
    };

    enum class FuseCode {
        FUSED_NONE,
        FUSED_RELU,
        FUSED_RELU1,
        FUSED_RELU6
    };

    struct QuantInfo {
        std::vector<float> scales;
        nonstd::optional<int32_t> zero_point;
    };
    StrKeyMap<QuantInfo> quant_infos_;

    std::map<std::string, std::string> name_map_;

    std::string m(const std::string &str);

    ONNX_NAMESPACE::ModelProto model_proto_;
    flatbuffers::FlatBufferBuilder builder_;
    std::vector<std::string> skipped_act_;

    std::vector<std::string> operands_;
    StrKeyMap<Tensor> nnapi_tensors_;
    StrKeyMap<Tensor> onnx_tensors_;
    std::vector<flatbuffers::Offset<DNN::Layer>> layers_;

    std::vector<flatbuffers::Offset<DNN::Tensor>> tensors_;

    DNN::FuseCode ConvertFuseCodeType(FuseCode fuse_code);
    std::pair<nonstd::optional<std::string>, FuseCode> FindActivation(const ONNX_NAMESPACE::ModelProto &model_proto, css &output_name);

    void HandleInitializer();
    std::vector<flatbuffers::Offset<DNN::Input>> GetInputOfOnnxModel();
    void ReadTableFile(css &table_file);

    void AddConv(const std::string &input_name, const std::vector<int> &strides, const std::vector<int> &pads, 
            const std::vector<int> &dilations, int group, 
            const std::pair<nonstd::optional<std::string>, FuseCode>& activation,
            const std::string &ori_weight_name, const nonstd::optional<std::string> &bias_name, const std::string &output_name);
    inline void addLayerPool(css &op, css &input_name, const std::vector<int> &kernel_shape, const std::vector<int> &pads, const std::vector<int> &strides, css &output_name) {
        auto activation = FindActivation(model_proto_, output_name);
        if (activation.first.has_value()) {
            skipped_act_.push_back(activation.first.value());
        }
        shaper_.Pool(input_name, strides[1], strides[0], pads[2], pads[3], pads[0], pads[1], kernel_shape[0], kernel_shape[1], output_name);
        flatbuffers::Offset<DNN::Layer> layer;
        if (op == "AveragePool" || op == "GlobalAveragePool") {
            auto param = DNN::CreateAvePoolDirect(builder_, input_name.c_str(), &kernel_shape, &pads, &strides,
                    ConvertFuseCodeType(activation.second), output_name.c_str());
            layer = DNN::CreateLayer(builder_, DNN::LayerType::AvePool, 0, param);
        } else {
            auto param = DNN::CreateMaxPoolDirect(builder_, input_name.c_str(), &kernel_shape, &pads, &strides,
                    ConvertFuseCodeType(activation.second), output_name.c_str());
            layer = DNN::CreateLayer(builder_, DNN::LayerType::MaxPool, 0, 0, param);
        }
        layers_.push_back(layer);
    }
    inline void addLayerRelu(css &input_name, css &output_name) {
        shaper_.Relu(input_name, output_name);
        auto param = DNN::CreateReluDirect(builder_, input_name.c_str(), output_name.c_str());
        auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Relu, 0, 0, 0, param);
        layers_.push_back(layer);
    }
    inline void addLayerAdd(css &input1_name, css &input2_name, css &output_name) {
        shaper_.Eltwise(input1_name, output_name);
        auto activation = FindActivation(model_proto_, output_name);
        if (activation.first.has_value()) {
            skipped_act_.push_back(activation.first.value());
        }
        auto param = DNN::CreateAddDirect(builder_, input1_name.c_str(), input2_name.c_str(),
                ConvertFuseCodeType(activation.second), output_name.c_str());
        auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Add, 0, 0, 0, 0, 0, 0, param);
        layers_.push_back(layer);
    }
    inline void addLayerAdd(css &input1_name, float input2, css &output_name) {
        shaper_.Eltwise(input1_name, output_name);
        const auto activation = FindActivation(model_proto_, output_name);
        if (activation.first.has_value()) {
            skipped_act_.push_back(activation.first.value());
        }
        const auto param = DNN::CreateAddScalarDirect(builder_, input1_name.c_str(), input2,
                ConvertFuseCodeType(activation.second), output_name.c_str());
        const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::AddScalar, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param, 0);
        layers_.push_back(layer);
    }
    inline void addLayerMul(css &input1_name, css &input2_name, css &output_name) {
        shaper_.Eltwise(input1_name, output_name);
        const auto activation = FindActivation(model_proto_, output_name);
        if (activation.first.has_value()) {
            skipped_act_.push_back(activation.first.value());
        }
        const auto param = DNN::CreateMulDirect(builder_, input1_name.c_str(), input2_name.c_str(),
                ConvertFuseCodeType(activation.second), output_name.c_str());
        const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Mul, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
        layers_.push_back(layer);
    }
    inline void addLayerMul(css &input1_name, float input2, css &output_name) {
        shaper_.Eltwise(input1_name, output_name);
        const auto activation = FindActivation(model_proto_, output_name);
        if (activation.first.has_value()) {
            skipped_act_.push_back(activation.first.value());
        }
        const auto param = DNN::CreateMulScalarDirect(builder_, input1_name.c_str(), input2,
                ConvertFuseCodeType(activation.second), output_name.c_str());
        const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::MulScalar, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
        layers_.push_back(layer);
    }
    inline void addLayerGemm(css &input_name, css &weight_name, nonstd::optional<std::string> bias_name, const int transA, const int transB, const float alpha, const float beta, css &output_name) {
        if (transA == 0 && transB == 1 && alpha == 1.f && beta == 1.f) {
            {
                nnapi_tensors_[weight_name] = onnx_tensors_.at(weight_name);
                const auto &weight_tensor = nnapi_tensors_[weight_name];
                const auto weight_data = weight_tensor.float_data();
                shaper_.AddShape(weight_name, weight_tensor.shape);
                auto flat_tensor = DNN::CreateTensorDirect(builder_, DNN::DataType::Float32, nullptr,
                        &weight_data, &weight_tensor.shape,
                        weight_name.c_str());
                tensors_.push_back(flat_tensor);
            }
            if (bias_name.has_value()) {
                nnapi_tensors_[bias_name.value()] = onnx_tensors_.at(bias_name.value());
                const auto &bias_tensor = nnapi_tensors_[bias_name.value()];
                const auto bias_data = bias_tensor.float_data();
                auto flat_tensor = DNN::CreateTensorDirect(builder_, DNN::DataType::Float32, nullptr,
                        &bias_data, &bias_tensor.shape, bias_name.value().c_str());
                tensors_.push_back(flat_tensor);
            }
            auto activation = FindActivation(model_proto_, output_name);
            if (activation.first.has_value()) {
                skipped_act_.push_back(activation.first.value());
            }
            shaper_.FC(input_name, weight_name, output_name);
            auto param = DNN::CreateFCDirect(builder_, input_name.c_str(), weight_name.c_str(),
                    bias_name.has_value() ? bias_name.value().c_str() : nullptr,
                    ConvertFuseCodeType(activation.second), output_name.c_str()
                    );
            auto layer = DNN::CreateLayer(builder_, DNN::LayerType::FC, 0, 0, 0, 0, 0, param, 0);
            layers_.push_back(layer);
        } else {
            throw std::invalid_argument(
                    "Only transA == 0, transB == 1, alpha == 1.0 and beta == 1.0 is supported.");
        }
    }
    inline void addLayerSoftmax(css &input_name, css &output_name) {
        shaper_.Softmax(input_name, output_name);
        // simply ignore attribute "axis", because nnapi softmax didn't has this attr, and we will check the equality of the two ops in DaqReader.cpp
        auto param = DNN::CreateSoftmaxDirect(builder_, input_name.c_str(), output_name.c_str());
        auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Softmax, 0, 0, 0, 0, param);
        layers_.push_back(layer);
    }
    // axis here is for onnx nchw
    inline void addLayerConcat(const std::vector<std::string> &inputs, css &output_name, const int axis) {
        std::vector<flatbuffers::Offset<flatbuffers::String>> concat_inputs;
        for (const auto &onnx_input : inputs) {
            auto flat_input = builder_.CreateString(m(onnx_input).c_str(), onnx_input.size());
            concat_inputs.push_back(flat_input);
        }
        uint32_t axis_nchw_to_nhwc[4]{0, 3, 1, 2};
        shaper_.Concat(inputs, axis, output_name);
        auto param = DNN::CreateConcatDirect(builder_, &concat_inputs, axis_nchw_to_nhwc[axis], output_name.c_str());
        auto layer = DNN::CreateLayer(builder_, DNN::LayerType::Concat, 0, 0, 0, 0, 0, 0, 0, param);
        layers_.push_back(layer);
    }

    /**
     * onnx: [filter_out_channel, filter_in_channel / group, height, width]
     * nnapi: [1, height, width, depth_out]
     */
    Tensor OnnxToNnapiDw(const Tensor &src) {
        Tensor dest;
        size_t elemsize = 0;
        if (src.data_type == Tensor::DataType::UINT8) {
            elemsize = 1;
        } else if (src.data_type == Tensor::DataType::FLOAT32) {
            elemsize = 4;
        }
        dest.data.resize(Product(src.shape) * elemsize);
        // t for total
        auto out_t = src.shape[0], in_t = src.shape[1], h_t = src.shape[2], w_t = src.shape[3];
        CHECK_EQ(in_t, 1u);
        for (uint32_t out = 0; out < out_t; out++) {
            for (uint32_t in = 0; in < in_t; in++) {
                for (uint32_t h = 0; h < h_t; h++) {
                    for (uint32_t w = 0; w < w_t; w++) {
                        auto onnx_idx = out * in_t * h_t * w_t + in * h_t * w_t + h * w_t + w;
                        auto nnapi_idx = h * w_t * out_t + w * out_t + out;
                        FORZ(i, elemsize) {
                            dest.data[elemsize * nnapi_idx + i] = src.data[elemsize * onnx_idx + i];
                        }
                    }
                }
            }
        }
        dest.shape = {in_t, h_t, w_t, out_t};
        dest.data_type = src.data_type;
        return dest;
    }

    /**
     * onnx: [filter_out_channel, filter_in_channel, height, width]
     * nnapi: [depth_out, height, width, depth_in]
     */
    Tensor OnnxToNnapiVanilla(const Tensor &src) {
        Tensor dest;
        size_t elemsize = 0;
        if (src.data_type == Tensor::DataType::UINT8) {
            elemsize = 1;
        } else if (src.data_type == Tensor::DataType::FLOAT32) {
            elemsize = 4;
        }
        dest.data.resize(Product(src.shape) * elemsize);
        // t for total
        auto out_t = src.shape[0], in_t = src.shape[1], h_t = src.shape[2], w_t = src.shape[3];
        for (uint32_t out = 0; out < out_t; out++) {
            for (uint32_t in = 0; in < in_t; in++) {
                for (uint32_t h = 0; h < h_t; h++) {
                    for (uint32_t w = 0; w < w_t; w++) {
                        auto onnx_idx = out * in_t * h_t * w_t + in * h_t * w_t + h * w_t + w;
                        auto nnapi_idx = out * h_t * w_t * in_t + h * w_t * in_t + w * in_t + in;
                        FORZ(i, elemsize) {
                            dest.data[elemsize * nnapi_idx + i] = src.data[elemsize * onnx_idx + i];
                        }
                    }
                }
            }
        }
        dest.shape = {out_t, h_t, w_t, in_t};
        dest.data_type = src.data_type;
        return dest;
    }

public:
    void Convert(const ONNX_NAMESPACE::ModelProto &model, const std::string &filepath, const css &table_file="");
};
