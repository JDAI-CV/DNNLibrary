#include <iostream>
#include <chrono>
#include <vector>

#include <glog/logging.h>
#include "android_log_helper.h"
#include <common/helper.h>
#include "ModelBuilder.h"

int main() {
    using namespace android::nn::wrapper;
    ModelBuilder builder;
    builder.Prepare();
    bool quant8 = false;
    uint8_t weight_buf[999];
    uint8_t bias_buf[999];
    if (quant8) {
        builder.AddInput("data", {Type::TENSOR_QUANT8_ASYMM, {1, 3, 224, 224}, 1, 0});
        builder.AddTensorFromBuffer("weight", weight_buf, {Type::TENSOR_QUANT8_ASYMM, {3, 1, 1, 3}, 0.1, 150});
        builder.AddTensorFromBuffer("bias", bias_buf, {Type::TENSOR_INT32, {3}, 0.1, 0});
        builder.AddConv("data", 1, 1, 0, 0, 0, 0, ModelBuilder::ACTIVATION_NONE, "weight", "bias", "output", std::make_optional<ModelBuilder::QuantInfo>({Type::TENSOR_QUANT8_ASYMM, {0.5}, 100}));
    } else {
        builder.AddInput("data", {Type::TENSOR_FLOAT32, {1, 3, 224, 224}});
        builder.AddTensorFromBuffer("weight", weight_buf, {Type::TENSOR_FLOAT32, {3, 1, 1, 3}});
        builder.AddTensorFromBuffer("bias", bias_buf, {Type::TENSOR_FLOAT32, {3}});
        builder.AddConv("data", 1, 1, 0, 0, 0, 0, ModelBuilder::ACTIVATION_NONE, "weight", "bias", "output");
    }
    builder.AddOutput("output");
    auto model = builder.Compile(ModelBuilder::PREFERENCE_FAST_SINGLE_ANSWER);
    if (quant8) {
        uint8_t input[1*3*224*224];
        uint8_t output[1*3*224*224];
        model->SetOutputBuffer(0, output);
        model->Predict(std::vector<uint8_t *>{input});
        LOG(INFO) << output[0];
    } else {
        float input[1*3*224*224];
        float output[1*3*224*224];
        model->SetOutputBuffer(0, output);
        model->Predict(std::vector<float *>{input});
        LOG(INFO) << output[0];
    }
}
