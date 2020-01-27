/**
 * It is an example showing how to use the ModelBuilder API to build an model
 */
#include <common/helper.h>
#include <dnnlibrary/ModelBuilder.h>
#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <vector>

using namespace android::nn::wrapper;
using dnn::ModelBuilder;

int main() {
    ModelBuilder builder;
    builder.Prepare();
    const bool quant8 = true;
    uint8_t weight_buf[999]{100, 200, 150, 20, 166, 22};
    uint8_t bias_buf[999]{99, 13, 235, 131};
    if (quant8) {
        builder.AddInput("data",
                         {Type::TENSOR_QUANT8_ASYMM, {1, 224, 224, 3}, 1, 0});
        builder.AddTensorFromBuffer(
            "weight", weight_buf,
            {Type::TENSOR_QUANT8_ASYMM, {3, 1, 1, 3}, 0.1, 150});
        builder.AddTensorFromBuffer("bias", bias_buf,
                                    {Type::TENSOR_INT32, {3}, 0.1, 0});
        builder.AddLayer_DEPTHWISE_CONV_2D(
            "data", "weight", "bias", 1, 1, 0, 0, 0, 0, 1, dnn::FuseCode::NONE,
            "conv_fwd",
            std::make_optional<ModelBuilder::QuantInfo>(
                {Type::TENSOR_QUANT8_ASYMM, {0.5}, 100}));
        builder.AddLayer_RELU("conv_fwd", "relu_fwd");
        builder.AddLayer_ADD("data", "relu_fwd", dnn::FuseCode::NONE, "output",
                             std::make_optional<ModelBuilder::QuantInfo>(
                                 {Type::TENSOR_QUANT8_ASYMM, {0.05}, 100}));
    } else {
        builder.AddInput("data", {Type::TENSOR_FLOAT32, {1, 224, 224, 3}});
        builder.AddTensorFromBuffer("weight", weight_buf,
                                    {Type::TENSOR_FLOAT32, {3, 1, 1, 3}});
        builder.AddTensorFromBuffer("bias", bias_buf,
                                    {Type::TENSOR_FLOAT32, {3}});
        builder.AddLayer_CONV_2D("data", "weight", "bias", 1, 1, 0, 0, 0, 0,
                                 dnn::FuseCode::NONE, false, 1, 1, "output",
                                 dnn::nullopt);
    }
    auto model = builder.AddOutput("output").Compile(
        ModelBuilder::PREFERENCE_FAST_SINGLE_ANSWER);
    if (quant8) {
        uint8_t input[1 * 3 * 224 * 224]{29, 100, 66, 166, 188, 222};
        uint8_t output[1 * 3 * 224 * 224];
        model->SetOutputBuffer(0, output);
        model->Predict(std::vector<uint8_t *>{input});
        LOG(INFO) << static_cast<int>(output[0]);
    } else {
        float input[1 * 3 * 224 * 224]{29, 100, 66, 166, 188, 222};
        float output[1 * 3 * 224 * 224];
        model->SetOutputBuffer(0, output);
        model->Predict(std::vector<float *>{input});
        LOG(INFO) << output[0];
    }
}
