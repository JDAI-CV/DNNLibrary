#include <dnnlibrary/NeuralNetworksWrapper.h>

#include <common/helper.h>

namespace android {
namespace nn {
namespace wrapper {

OperandType::OperandType(Type type, std::vector<uint32_t> d, float scale,
                         int32_t zeroPoint)
    : type(type), dimensions(std::move(d)), channelQuant(dnn::nullopt) {
    if (dimensions.empty()) {
        if (!isScalarType(type)) {
            dimensions = {1};
        }
    } else {
        DNN_ASSERT(!isScalarType(type), typeToStr(type), " ", dimensions);
    }
    operandType = {
        .type = static_cast<int32_t>(type),
        .dimensionCount = static_cast<uint32_t>(dimensions.size()),
        .dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr,
        .scale = scale,
        .zeroPoint = zeroPoint,
    };
}
OperandType::OperandType(Type type, std::vector<uint32_t> data, float scale,
                         int32_t zeroPoint,
                         SymmPerChannelQuantParams&& channelQuant)
    : type(type),
      dimensions(std::move(data)),
      channelQuant(std::move(channelQuant)) {
    if (dimensions.empty()) {
        DNN_ASSERT(isScalarType(type), "");
    } else {
        DNN_ASSERT(!isScalarType(type), "");
    }
    operandType = {
        .type = static_cast<int32_t>(type),
        .dimensionCount = static_cast<uint32_t>(dimensions.size()),
        .dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr,
        .scale = scale,
        .zeroPoint = zeroPoint,
    };
}

}  // namespace wrapper
}  // namespace nn
}  // namespace android
