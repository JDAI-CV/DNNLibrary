/**
 * It is an example showing how to use the ModelBuilder API to build an model
 */
#include <chrono>
#include <iostream>
#include <vector>

#include <common/helper.h>
#include <dnnlibrary/ModelBuilder.h>
#include <glog/logging.h>

using namespace android::nn::wrapper;
using dnn::Model;

int main() {
    Model model;
    const bool quant8 = true;
    uint8_t weight_buf[999]{100, 200, 150, 20, 166, 22};
    uint8_t bias_buf[999]{99, 13, 235, 131};
    if (quant8) {
        model.AddInput("data",
                       {Type::TENSOR_QUANT8_ASYMM, {1, 224, 224, 3}, 1, 0});
        model.AddTensorFromBuffer(
            "weight", weight_buf,
            {Type::TENSOR_QUANT8_ASYMM, {3, 1, 1, 3}, 0.1, 150});
        model.AddTensorFromBuffer("bias", bias_buf,
                                  {Type::TENSOR_INT32, {3}, 0.1, 0});
        model.AddDepthWiseConv("data", 1, 1, 0, 0, 0, 0, Model::ACTIVATION_NONE,
                               1, "weight", "bias", "conv_fwd",
                               std::make_optional<Model::QuantInfo>(
                                   {Type::TENSOR_QUANT8_ASYMM, {0.5}, 100}));
        model.AddReLU("conv_fwd", "relu_fwd");
        model.AddOperationAdd("data", "relu_fwd", "output",
                              std::make_optional<Model::QuantInfo>(
                                  {Type::TENSOR_QUANT8_ASYMM, {0.05}, 100}));
    } else {
        model.AddInput("data", {Type::TENSOR_FLOAT32, {1, 224, 224, 3}});
        model.AddTensorFromBuffer("weight", weight_buf,
                                  {Type::TENSOR_FLOAT32, {3, 1, 1, 3}});
        model.AddTensorFromBuffer("bias", bias_buf,
                                  {Type::TENSOR_FLOAT32, {3}});
        model.AddConv("data", 1, 1, 0, 0, 0, 0, Model::ACTIVATION_NONE,
                      "weight", "bias", "output");
    }
    model.AddOutput("output").Compile(Model::PREFERENCE_FAST_SINGLE_ANSWER);
    if (quant8) {
        uint8_t input[1 * 3 * 224 * 224]{29, 100, 66, 166, 188, 222};
        uint8_t output[1 * 3 * 224 * 224];
        model.SetOutputBuffer(0, output);
        model.Predict(std::vector<uint8_t *>{input});
        LOG(INFO) << static_cast<int>(output[0]);
    } else {
        float input[1 * 3 * 224 * 224]{29, 100, 66, 166, 188, 222};
        float output[1 * 3 * 224 * 224];
        model.SetOutputBuffer(0, output);
        model.Predict(std::vector<float *>{input});
        LOG(INFO) << output[0];
    }
}
