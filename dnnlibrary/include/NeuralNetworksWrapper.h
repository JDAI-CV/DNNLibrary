/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Provides C++ classes to more easily use the Neural Networks API.
#ifndef ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_WRAPPER_H
#define ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_WRAPPER_H
#include "NeuralNetworksMock.h"
#include <math.h>
#include <optional>
#include <string>
#include <vector>
#include <common/helper.h>

namespace android {
namespace nn {
namespace wrapper {
enum class Type {
    FLOAT32 = ANEURALNETWORKS_FLOAT32,
    INT32 = ANEURALNETWORKS_INT32,
    UINT32 = ANEURALNETWORKS_UINT32,
    TENSOR_FLOAT32 = ANEURALNETWORKS_TENSOR_FLOAT32,
    TENSOR_INT32 = ANEURALNETWORKS_TENSOR_INT32,
    TENSOR_QUANT8_ASYMM = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
    BOOL = ANEURALNETWORKS_BOOL,
    TENSOR_QUANT16_SYMM = ANEURALNETWORKS_TENSOR_QUANT16_SYMM,
    TENSOR_FLOAT16 = ANEURALNETWORKS_TENSOR_FLOAT16,
    TENSOR_BOOL8 = ANEURALNETWORKS_TENSOR_BOOL8,
    FLOAT16 = ANEURALNETWORKS_FLOAT16,
    TENSOR_QUANT8_SYMM_PER_CHANNEL = ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL,
    TENSOR_QUANT16_ASYMM = ANEURALNETWORKS_TENSOR_QUANT16_ASYMM,
};
enum class ExecutePreference {
    PREFER_LOW_POWER = ANEURALNETWORKS_PREFER_LOW_POWER,
    PREFER_FAST_SINGLE_ANSWER = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER,
    PREFER_SUSTAINED_SPEED = ANEURALNETWORKS_PREFER_SUSTAINED_SPEED
};
enum class Result {
    NO_ERROR = ANEURALNETWORKS_NO_ERROR,
    OUT_OF_MEMORY = ANEURALNETWORKS_OUT_OF_MEMORY,
    INCOMPLETE = ANEURALNETWORKS_INCOMPLETE,
    UNEXPECTED_NULL = ANEURALNETWORKS_UNEXPECTED_NULL,
    BAD_DATA = ANEURALNETWORKS_BAD_DATA,
    OP_FAILED = ANEURALNETWORKS_OP_FAILED,
    UNMAPPABLE = ANEURALNETWORKS_UNMAPPABLE,
    BAD_STATE = ANEURALNETWORKS_BAD_STATE,
    OUTPUT_INSUFFICIENT_SIZE = ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE,
    UNAVAILABLE_DEVICE = ANEURALNETWORKS_UNAVAILABLE_DEVICE,
};
struct SymmPerChannelQuantParams {
    ANeuralNetworksSymmPerChannelQuantParams params;
    std::vector<float> scales;
    SymmPerChannelQuantParams(std::vector<float> scalesVec, uint32_t channelDim)
        : scales(std::move(scalesVec)) {
        params = {
                .channelDim = channelDim,
                .scaleCount = static_cast<uint32_t>(scales.size()),
                .scales = scales.size() > 0 ? scales.data() : nullptr,
        };
    }
};

// A helper function
inline bool isScalarType(const Type &type) {
    return type == Type::FLOAT16 || type == Type::FLOAT32 || type == Type::INT32 || type == Type::BOOL || type == Type::UINT32;
}

inline std::string typeToStr(const Type &type) {
    if (type == Type::FLOAT32) {
        return "FLOAT32";
    } else if (type == Type::INT32) {
        return "INT32";
    } else if (type == Type::UINT32) {
        return "UINT32";
    } else if (type == Type::TENSOR_FLOAT32) {
        return "TENSOR_FLOAT32";
    } else if (type == Type::TENSOR_INT32) {
        return "TENSOR_INT32";
    } else if (type == Type::TENSOR_QUANT8_ASYMM) {
        return "TENSOR_QUANT8_ASYMM";
    } else if (type == Type::TENSOR_QUANT16_ASYMM) {
        return "TENSOR_QUANT16_ASYMM";
    } else if (type == Type::TENSOR_QUANT16_SYMM) {
        return "TENSOR_QUANT16_SYMM";
    } else if (type == Type::BOOL) {
        return "BOOL";
    } else if (type == Type::TENSOR_FLOAT16) {
        return "TENSOR_FLOAT16";
    } else if (type == Type::TENSOR_BOOL8) {
        return "TENSOR_BOOL8";
    } else if (type == Type::FLOAT16) {
        return "FLOAT16";
    } else if (type == Type::FLOAT16) {
        return "FLOAT16";
    } else if (type == Type::TENSOR_QUANT8_SYMM_PER_CHANNEL) {
        return "TENSOR_QUANT8_SYMM_PER_CHANNEL";
    } else {
        return "Unknown type";
    }
}

struct OperandType {
    ANeuralNetworksOperandType operandType;
    Type type;
    std::vector<uint32_t> dimensions;
    std::optional<SymmPerChannelQuantParams> channelQuant;
    OperandType(Type type, std::vector<uint32_t> d = {}, float scale = 0.0f, int32_t zeroPoint = 0)
        : type(type), dimensions(std::move(d)), channelQuant(std::nullopt) {
        if (dimensions.empty()) {
            DNN_ASSERT(isScalarType(type), typeToStr(type));
        } else {
            DNN_ASSERT(!isScalarType(type), typeToStr(type));
        }
        operandType = {
                .type = static_cast<int32_t>(type),
                .dimensionCount = static_cast<uint32_t>(dimensions.size()),
                .dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr,
                .scale = scale,
                .zeroPoint = zeroPoint,
        };
    }
    OperandType(Type type, std::vector<uint32_t> data, float scale, int32_t zeroPoint,
                SymmPerChannelQuantParams&& channelQuant)
        : type(type), dimensions(std::move(data)), channelQuant(std::move(channelQuant)) {
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
    operator ANeuralNetworksOperandType() const {return operandType; }
};
class Memory {
public:
    Memory(size_t size, int protect, int fd, size_t offset) {
        mValid = ANeuralNetworksMemory_createFromFd(size, protect, fd, offset, &mMemory) ==
                 ANEURALNETWORKS_NO_ERROR;
    }
    Memory(AHardwareBuffer* buffer) {
        mValid = ANeuralNetworksMemory_createFromAHardwareBuffer(buffer, &mMemory) ==
                 ANEURALNETWORKS_NO_ERROR;
    }
    ~Memory() { ANeuralNetworksMemory_free(mMemory); }
    // Disallow copy semantics to ensure the runtime object can only be freed
    // once. Copy semantics could be enabled if some sort of reference counting
    // or deep-copy system for runtime objects is added later.
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    // Move semantics to remove access to the runtime object from the wrapper
    // object that is being moved. This ensures the runtime object will be
    // freed only once.
    Memory(Memory&& other) { *this = std::move(other); }
    Memory& operator=(Memory&& other) {
        if (this != &other) {
            ANeuralNetworksMemory_free(mMemory);
            mMemory = other.mMemory;
            mValid = other.mValid;
            other.mMemory = nullptr;
            other.mValid = false;
        }
        return *this;
    }
    ANeuralNetworksMemory* get() const { return mMemory; }
    bool isValid() const { return mValid; }
private:
    ANeuralNetworksMemory* mMemory = nullptr;
    bool mValid = true;
};
class Event {
public:
    Event() {}
    ~Event() { ANeuralNetworksEvent_free(mEvent); }
    // Disallow copy semantics to ensure the runtime object can only be freed
    // once. Copy semantics could be enabled if some sort of reference counting
    // or deep-copy system for runtime objects is added later.
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    // Move semantics to remove access to the runtime object from the wrapper
    // object that is being moved. This ensures the runtime object will be
    // freed only once.
    Event(Event&& other) { *this = std::move(other); }
    Event& operator=(Event&& other) {
        if (this != &other) {
            ANeuralNetworksEvent_free(mEvent);
            mEvent = other.mEvent;
            other.mEvent = nullptr;
        }
        return *this;
    }
    Result wait() { return static_cast<Result>(ANeuralNetworksEvent_wait(mEvent)); }
    // Only for use by Execution
    void set(ANeuralNetworksEvent* newEvent) {
        ANeuralNetworksEvent_free(mEvent);
        mEvent = newEvent;
    }
private:
    ANeuralNetworksEvent* mEvent = nullptr;
};
}  // namespace wrapper
}  // namespace nn
}  // namespace android
#endif  //  ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_WRAPPER_H
