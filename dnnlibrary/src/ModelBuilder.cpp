//
// Created by daquexian on 2017/11/8.
//
#include "ModelBuilder.h"

#include <array>
#include <ctime>
#include <tuple>
#include <sstream>
#include <sys/mman.h>
#include <fstream>
#include <iostream>

#include <glog/logging.h>
#include "android_log_helper.h"
#include <common/helper.h>
#include <operand_helper.h>

#define THROW_ON_ERROR(val) \
    if ((val) != ANEURALNETWORKS_NO_ERROR) {  \
        throw std::invalid_argument(std::string("Error in ") + __FILE__ + std::string(":") + \
                std::to_string(__LINE__) + std::string(", function name: ") + \
                std::string(__func__) + "error, ret: " + GetErrorCause(val));   \
    }

#define THROW_ON_ERROR_WITH_NOTE(val, note) \
    if ((val) != ANEURALNETWORKS_NO_ERROR) {  \
        throw std::invalid_argument(std::string("Error in ") + __FILE__ + std::string(":") + \
                std::to_string(__LINE__) + std::string(", function name: ") + \
                std::string(__func__) + "error, ret: " + GetErrorCause(val) + std::string(", ") + (note));   \
    }

using std::vector; using std::ifstream; using std::streamsize; using std::string; using std::ios;
using std::stringstream; using std::array;
using namespace android::nn::wrapper;

void ModelBuilder::RegisterOperand(const std::string &name, ModelBuilder::Index index, const OperandType &operand_type) {
    operand_indexes_[name] = index;
    ordered_operands_.push_back(name);
    operand_types_.insert({name, operand_type});
}

ModelBuilder::Index ModelBuilder::AddInput(string name, const uint32_t height, const uint32_t width, const uint32_t depth) {
    const vector<uint32_t> dimen{1, width, height, depth};
    return AddInput(name, {Type::TENSOR_FLOAT32, dimen});
}

ModelBuilder::Index ModelBuilder::AddInput(std::string name, const OperandType &operand_type) {
    const auto index = AddNewOperand(operand_type);

    shaper_.AddShape(name, operand_type.dimensions);
    input_index_vec_.push_back(index);
    dnn_model_->AddInput(name, shaper_[name]);
    RegisterOperand(name, index, operand_type);
    return index;
}

ModelBuilder::Index ModelBuilder::AddDepthWiseConv(const string &input_name, int32_t strideX, int32_t strideY,
                                                   int32_t paddingLeft,
                                                   int32_t paddingRight, int32_t paddingBottom, int32_t paddingTop,
                                                   int32_t activation,
                                                   int32_t depthMultiplier, const string &weight_name,
                                                   const std::optional<string> &bias_name,
                                                   const string &output_name,
                                                   const std::optional<QuantInfo> &output_quant_info) {
    const auto input = operand_indexes_[input_name];
    const auto weight = operand_indexes_[weight_name];

    uint32_t biasIndexValue;
    if (!bias_name.has_value()) {
        Shape weightDimen = shaper_[weight_name];     // 1, height, width, num_output
        Shape bias_dims = Shape{weightDimen[3]};
        const auto &weight_type = operand_types_.at(weight_name).type;
        if (weight_type == Type::TENSOR_FLOAT32) {
            biasIndexValue = FillOperand(weight_name + "_b", {Type::TENSOR_FLOAT32, bias_dims}, 0.f);
        } else if (weight_type == Type::TENSOR_QUANT8_ASYMM) {
            const auto input_scale = operand_types_.at(input_name).operandType.scale;
            const auto weight_scale = operand_types_.at(weight_name).operandType.scale;
            biasIndexValue = FillOperand(weight_name + "_b", 
                    {Type::TENSOR_INT32, bias_dims, input_scale * weight_scale}, 0);
        } else {
            throw std::invalid_argument("Unknown type " + typeToStr(weight_type));
        }
    } else {
        biasIndexValue = operand_indexes_[bias_name.value()];
    }
    shaper_.DepthwiseConv(input_name, strideX, strideY, 1, 1, paddingLeft, paddingRight, paddingTop, paddingBottom, weight_name, output_name);
    if (bias_name.has_value() && operand_types_.at(input_name).isQuant()) {
        const auto input_scale = operand_types_.at(input_name).operandType.scale;
        const auto weight_scale = operand_types_.at(weight_name).operandType.scale;
        const auto bias_scale = operand_types_.at(bias_name.value()).operandType.scale;
        DNN_ASSERT(input_scale > 0, "");
        DNN_ASSERT(weight_scale > 0, "");
        DNN_ASSERT(bias_scale > 0, "");
        DNN_ASSERT(input_scale * weight_scale == bias_scale, "");
    }
    IndexSeq input_indexes{input, weight, biasIndexValue};
    AddScalarOperands(input_indexes, paddingLeft, paddingRight, paddingTop, paddingBottom,
                strideX, strideY, depthMultiplier, activation);
    DNN_ASSERT((operand_types_.at(input_name).type == Type::TENSOR_FLOAT32 && operand_types_.at(weight_name).type == Type::TENSOR_FLOAT32) || (operand_types_.at(input_name).type == Type::TENSOR_QUANT8_ASYMM && operand_types_.at(weight_name).type == Type::TENSOR_QUANT8_ASYMM), "");
    OperandType operand_type = GetOperandType(operand_types_.at(input_name).type, shaper_[output_name], output_quant_info);
    const auto output_index = AddOperation(ANEURALNETWORKS_DEPTHWISE_CONV_2D, input_indexes, operand_type)[0];
    RegisterOperand(output_name, output_index, operand_type);
    return output_index;
}

ModelBuilder::Index
ModelBuilder::AddConv(const string &input_name, int32_t strideX, int32_t strideY, int32_t paddingLeft,
                      int32_t paddingRight,
                      int32_t paddingTop, int32_t paddingBottom, int32_t activation, const string &weight_name,
                      const std::optional<string> &bias_name, const string &output_name,
                      const std::optional<QuantInfo> &output_quant_info) {
    const auto input = operand_indexes_[input_name];
    const auto weight = operand_indexes_[weight_name];

    uint32_t biasIndexValue;
    if (!bias_name.has_value()) {
        Shape weightDimen = shaper_[weight_name];     // num_output, height, width, num_input
        Shape bias_dims = Shape{weightDimen[0]};
        const auto &weight_type = operand_types_.at(weight_name).type;
        if (weight_type == Type::TENSOR_FLOAT32) {
            biasIndexValue = FillOperand(weight_name + "_b", {Type::TENSOR_FLOAT32, bias_dims}, 0.f);
        } else if (weight_type == Type::TENSOR_QUANT8_ASYMM) {
            const auto input_scale = operand_types_.at(input_name).operandType.scale;
            const auto weight_scale = operand_types_.at(weight_name).operandType.scale;
            biasIndexValue = FillOperand(weight_name + "_b", 
                    {Type::TENSOR_INT32, bias_dims, input_scale * weight_scale}, 0);
        } else {
            throw std::invalid_argument("Unknown type " + typeToStr(weight_type));
        }
    } else {
        biasIndexValue = operand_indexes_[bias_name.value()];
    }
    shaper_.Conv(input_name, strideX, strideY, 1, 1, paddingLeft, paddingRight, paddingTop, paddingBottom, weight_name, output_name);
    if (bias_name.has_value() && operand_types_.at(input_name).isQuant()) {
        const auto input_scale = operand_types_.at(input_name).operandType.scale;
        const auto weight_scale = operand_types_.at(weight_name).operandType.scale;
        const auto bias_scale = operand_types_.at(bias_name.value()).operandType.scale;
        DNN_ASSERT(input_scale > 0, "");
        DNN_ASSERT(weight_scale > 0, "");
        DNN_ASSERT(bias_scale > 0, "");
        DNN_ASSERT(input_scale * weight_scale == bias_scale, "");
    }
    IndexSeq input_indexes{input, weight, biasIndexValue};
    AddScalarOperands(input_indexes, paddingLeft, paddingRight, paddingTop, paddingBottom, strideX, strideY, activation);
    DNN_ASSERT((operand_types_.at(input_name).type == Type::TENSOR_FLOAT32 && operand_types_.at(weight_name).type == Type::TENSOR_FLOAT32) || (operand_types_.at(input_name).type == Type::TENSOR_QUANT8_ASYMM && operand_types_.at(weight_name).type == Type::TENSOR_QUANT8_ASYMM), "");
    OperandType operand_type = GetOperandType(operand_types_.at(input_name).type, shaper_[output_name], output_quant_info);
    const auto output_index = AddOperation(ANEURALNETWORKS_CONV_2D, input_indexes, operand_type)[0];
    RegisterOperand(output_name, output_index, operand_type);
    return output_index;
}

#if __ANDROID_API__ >= __ANDROID_API_P__

ModelBuilder::Index
ModelBuilder::AddStridedSlice(const string &input_name, const vector<int32_t> &starts, const vector<int32_t> &ends,
                              const vector<int32_t> &strides, int32_t beginMask, int32_t endMask,
                              int32_t shrinkAxisMask, const string &output_name) {

    const auto input = operand_indexes_[input_name];

    const auto startsIndex = AddTensorFromBuffer(output_name + "_starts", &starts[0], {Type::TENSOR_INT32, Shape{static_cast<uint32_t>(starts.size())}});
    const auto endsIndex = AddTensorFromBuffer(output_name + "_ends", &ends[0], {Type::TENSOR_INT32, Shape{static_cast<uint32_t>(ends.size())}});
    const auto stridesIndex = AddTensorFromBuffer(output_name + "_strides", &strides[0], {Type::TENSOR_INT32, Shape{static_cast<uint32_t>(strides.size())}});

    shaper_.StridedSlice(input_name, starts, ends, strides, beginMask, endMask, shrinkAxisMask, output_name);
    IndexSeq input_indexes{input, startsIndex, endsIndex, stridesIndex};
    AddScalarOperands(input_indexes, beginMask, endMask, shrinkAxisMask);

    const OperandType operand_type(operand_types_.at(input_name).type, shaper_[output_name]);
    const auto output_index = AddOperation(ANEURALNETWORKS_STRIDED_SLICE, input_indexes, operand_type)[0];
    RegisterOperand(output_name, output_index, operand_type);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddSpaceToBatchND(const std::string &input_name, const std::vector<int32_t> &block_sizes,
        const std::vector<int32_t> &pads, const std::string &output_name) {
    const auto input = operand_indexes_[input_name];

    const auto block_sizes_idx = AddTensorFromBuffer(output_name + "_bs", &block_sizes[0], {Type::TENSOR_INT32, Shape{static_cast<uint32_t>(block_sizes.size())}});
    const auto pads_idx = AddTensorFromBuffer(output_name + "_pad", &pads[0], {Type::TENSOR_INT32, Shape{static_cast<uint32_t>(pads.size()) / 2, 2}});

    shaper_.SpaceToBatch(input_name, block_sizes, pads, output_name);
    IndexSeq input_indexes{input, block_sizes_idx, pads_idx};
    const OperandType operand_type(operand_types_.at(input_name).type, shaper_[output_name]);
    const auto output_index = AddOperation(ANEURALNETWORKS_SPACE_TO_BATCH_ND, input_indexes, operand_type)[0];
    RegisterOperand(output_name, output_index, operand_type);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddBatchToSpaceND(const std::string &input_name, const std::vector<int32_t> &block_sizes,
        const std::string &output_name) {
    const auto input = operand_indexes_[input_name];

    const auto block_sizes_idx = AddTensorFromBuffer(output_name + "_bs", &block_sizes[0], {Type::TENSOR_INT32, Shape{static_cast<uint32_t>(block_sizes.size())}});

    shaper_.BatchToSpace(input_name, block_sizes, output_name);
    IndexSeq input_indexes{input, block_sizes_idx};
    const OperandType operand_type(operand_types_.at(input_name).type, shaper_[output_name]);
    const auto output_index = AddOperation(ANEURALNETWORKS_BATCH_TO_SPACE_ND, input_indexes, operand_type)[0];
    RegisterOperand(output_name, output_index, operand_type);
    return output_index;
}

#endif

ModelBuilder::Index ModelBuilder::AddPool(const string &input_name, int32_t strideX, int32_t strideY,
                                          int32_t paddingLeft, int32_t paddingRight,
                                          int32_t paddingTop, int32_t paddingBottom, int32_t height, int32_t width,
                                          int32_t activation,
                                          PoolingType poolingType, const string &output_name,
                                          const std::optional<QuantInfo> &output_quant_info) {
    const auto input = operand_indexes_[input_name];

    if (height == -1 && width == -1) {
        VLOG(5) << "Global pool, input: " << input_name;
        const auto inputDimen = shaper_[input_name];
        height = inputDimen[1];
        width = inputDimen[2];
        strideX = width;
        strideY = height;
    }
    shaper_.Pool(input_name, strideX, strideY, paddingLeft, paddingRight, paddingTop, paddingBottom,
            height, width, output_name);
    IndexSeq input_indexes{input};
    AddScalarOperands(input_indexes, 
            paddingLeft, paddingRight, paddingTop, paddingBottom, 
            strideX, strideY, width, height, activation);

    const OperandType operand_type = GetOperandType(operand_types_.at(input_name).type, shaper_[output_name], output_quant_info);
    Index output_index;
    if (poolingType == PoolingType::MAX_POOL) {
        output_index = AddOperation(ANEURALNETWORKS_MAX_POOL_2D, input_indexes, operand_type)[0];
    } else if (poolingType == PoolingType::AVE_POOL) {
        output_index = AddOperation(ANEURALNETWORKS_AVERAGE_POOL_2D, input_indexes, operand_type)[0];
    } else {
        throw std::invalid_argument("Unknown pooling type");
    }
    RegisterOperand(output_name, output_index, operand_type);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddSoftMax(const string &input_name, float beta, const string &output_name) {
    const auto input = operand_indexes_[input_name];

    shaper_.Softmax(input_name, output_name);
    IndexSeq input_indexes{input};
    AddScalarOperands(input_indexes, beta);

    const OperandType operand_type(operand_types_.at(input_name).type, shaper_[output_name]);
    const auto output_index = AddOperation(ANEURALNETWORKS_SOFTMAX, input_indexes, operand_type)[0];
    RegisterOperand(output_name, output_index, operand_type);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddReLU(const string &input_name, const string &output_name) {
    const auto input = operand_indexes_[input_name];

    shaper_.Relu(input_name, output_name);
    IndexSeq input_indexes{input};

    const OperandType operand_type = operand_types_.at(input_name);
    const auto output_index = AddOperation(ANEURALNETWORKS_RELU, input_indexes, operand_type)[0];
    RegisterOperand(output_name, output_index, operand_type);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddConcat(const vector<string> &input_names, int32_t axis, const string &output_name) {
    IndexSeq inputs;
    for (const auto &input_name : input_names) {
        inputs.push_back(operand_indexes_[input_name]);
    }

    shaper_.Concat(input_names, axis, output_name);
    IndexSeq input_indexes(inputs);
    AddScalarOperands(input_indexes, axis);

    const OperandType &input_op_type = operand_types_.at(input_names[0]); 

    const OperandType operand_type(input_op_type.type, shaper_[output_name], input_op_type.operandType.scale, input_op_type.operandType.zeroPoint);
    const auto output_index = AddOperation(ANEURALNETWORKS_CONCATENATION, input_indexes, operand_type)[0];
    RegisterOperand(output_name, output_index, operand_type);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddLRN(const string &input_name, int32_t local_size, float bias, float alpha,
                                         float beta,
                                         const string &output_name) {
    const auto input = operand_indexes_[input_name];

    shaper_.LRN(input_name, output_name);
    IndexSeq input_indexes{input};
    AddScalarOperands(input_indexes, local_size, bias, alpha, beta);

    const OperandType operand_type(operand_types_.at(input_name).type, shaper_[output_name]);
    const auto output_idx = AddOperation(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION, input_indexes, operand_type)[0];
    RegisterOperand(output_name, output_idx, operand_type);
    return output_idx;
}

ModelBuilder::Index ModelBuilder::AddFC(const string &input_name, int32_t activation,
                                        const string &weight_name, const std::optional<string> &bias_name,
                                        const string &output_name, const std::optional<QuantInfo> &output_quant_info) {
    const auto input = operand_indexes_[input_name];
    const auto weight = operand_indexes_[weight_name];
    uint32_t biasIndexValue;
    if (!bias_name.has_value()) {
        auto weightDimen = shaper_[weight_name];
        Shape bias_dims = Shape{weightDimen[0]};
        biasIndexValue = FillOperand(weight_name + "_b", {Type::TENSOR_FLOAT32, bias_dims}, 0.f);
    } else {
        biasIndexValue = operand_indexes_[bias_name.value()];
    }
    shaper_.FC(input_name, weight_name, output_name);
    IndexSeq input_indexes{input, weight, biasIndexValue};
    AddScalarOperands(input_indexes, activation);
    const OperandType operand_type = GetOperandType(operand_types_.at(input_name).type, shaper_[output_name], output_quant_info);
    const auto output_idx = AddOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indexes, operand_type)[0];
    RegisterOperand(output_name, output_idx, operand_type);
    return output_idx;
}

ModelBuilder::Index ModelBuilder::AddOperationAdd(const string &input_name, float scalar, string output_name) {
    const auto input = operand_indexes_[input_name];
    Index scalarIndex = FillOperand(output_name + "_add", {Type::TENSOR_FLOAT32, {1}}, scalar);
    IndexSeq inputOperands{input, scalarIndex, OperandFromScalar(
            ModelBuilder::ACTIVATION_NONE)};
    shaper_.Eltwise(input_name, output_name);
    const OperandType operand_type(operand_types_.at(input_name).type, shaper_[output_name]);
    const auto output_index = AddOperation(ANEURALNETWORKS_ADD, inputOperands, operand_type)[0];
    RegisterOperand(output_name, output_index, operand_type);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddOperationAdd(const string &input1_name, const string &input2_name,
                                               const string &output_name, const std::optional<QuantInfo> &output_quant_info) {
    const auto input1 = operand_indexes_[input1_name];
    const auto input2 = operand_indexes_[input2_name];
    shaper_.Eltwise(input1_name, input2_name, output_name);
    IndexSeq input_indexes{input1, input2};
    AddScalarOperands(input_indexes, ModelBuilder::ACTIVATION_NONE);
    const OperandType operand_type = GetOperandType(operand_types_.at(input1_name).type, shaper_[output_name], output_quant_info);
    const auto output_idx = AddOperation(ANEURALNETWORKS_ADD, input_indexes, operand_type)[0];
    RegisterOperand(output_name, output_idx, operand_type);
    return output_idx;
}

ModelBuilder::Index ModelBuilder::AddMul(const string &input_name, float scalar, const string &output_name) {
    const auto input = operand_indexes_[input_name];
    Index scalarIndex = FillOperand(output_name + "_mul", {Type::TENSOR_FLOAT32, {1}}, scalar);
    IndexSeq inputOperands{input, scalarIndex, OperandFromScalar(
            ModelBuilder::ACTIVATION_NONE)};

    shaper_.Eltwise(input_name, output_name);
    const OperandType operand_type(operand_types_.at(input_name).type, shaper_[output_name]);
    const auto output_index = AddOperation(ANEURALNETWORKS_MUL, inputOperands, operand_type)[0];
    RegisterOperand(output_name, output_index, operand_type);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddMul(const string &input1_name, const string &input2_name,
                                               const string &output_name, const std::optional<QuantInfo> &output_quant_info) {
    const auto input1 = operand_indexes_[input1_name];
    const auto input2 = operand_indexes_[input2_name];
    IndexSeq input_indexes{input1, input2};
    shaper_.Eltwise(input1_name, output_name);
    AddScalarOperands(input_indexes, ModelBuilder::ACTIVATION_NONE);
    const OperandType operand_type = GetOperandType(operand_types_.at(input1_name).type, shaper_[output_name], output_quant_info);
    const auto output_idx = AddOperation(ANEURALNETWORKS_MUL, input_indexes, operand_type)[0];
    RegisterOperand(output_name, output_idx, operand_type);
    return output_idx;
}
//--------------------------------------------------------------------------------------------------//

OperandType ModelBuilder::GetOperandType(const Type &type) {
    return {type, {}};
}

OperandType ModelBuilder::GetOperandType(const Type &type, const Shape &dims, const std::optional<QuantInfo> &quant_info) {
    if (quant_info.has_value()) {
        const auto &quant_info_val = quant_info.value();
        return GetOperandType(quant_info_val, dims);
    }
    return {type, dims};
}

OperandType ModelBuilder::GetOperandType(const QuantInfo &quant_info, const Shape &dims) {
    if (quant_info.type_ == Type::TENSOR_QUANT8_SYMM_PER_CHANNEL) {
        // FIXME: implement it
        throw std::invalid_argument("");
    } else {
        DNN_ASSERT(quant_info.scales_.size() == 1, "");
        return {quant_info.type_, dims, quant_info.scales_[0], quant_info.zero_point_.value_or(0)};
    }
}

#define DEFINE_OPERAND_FROM_SCALAR(scalar_type, map_type, op_type)  \
ModelBuilder::Index ModelBuilder::OperandFromScalar(scalar_type value) {   \
    if (map_type##_operand_map_.find(value) == map_type##_operand_map_.end()) { \
        const auto index = AddNewOperand({Type::op_type}); \
        THROW_ON_ERROR_WITH_NOTE(ANeuralNetworksModel_setOperandValue(dnn_model_->model_, index, &value, sizeof(value)),    \
                "value: " + std::to_string(value)); \
        map_type##_operand_map_[value] = index; \
    }   \
    return map_type##_operand_map_[value];  \
}

DEFINE_OPERAND_FROM_SCALAR(uint32_t, uint32, UINT32);
DEFINE_OPERAND_FROM_SCALAR(int32_t, int32, INT32);
DEFINE_OPERAND_FROM_SCALAR(float, float32, FLOAT32);

#undef DEFINE_OPERAND_FROM_SCALAR

ModelBuilder::Index ModelBuilder::AddMissingOperand(const OperandType &operand_type) {
    const auto index = AddNewOperand(operand_type);
    THROW_ON_ERROR(ANeuralNetworksModel_setOperandValue(dnn_model_->model_, index, nullptr, 0));
    return index;
}

ModelBuilder::Index ModelBuilder::AddNewOperand(const OperandType &operand_type) {
    THROW_ON_ERROR(ANeuralNetworksModel_addOperand(dnn_model_->model_, &operand_type.operandType));
    return next_index_++;
}

// TODO: combine it and AddTensorFromBuffer
ModelBuilder::Index ModelBuilder::AddTensorFromMemory(const string &name, const uint8_t *addr, Shape dimen) {
    throw std::invalid_argument("");
    DNN_ASSERT(!dimen.empty(), "");
    const auto index = AddNewOperand({Type::TENSOR_FLOAT32, dimen});
    THROW_ON_ERROR(ANeuralNetworksModel_setOperandValueFromMemory(
                dnn_model_->model_, index, dnn_model_->memory_, addr - dnn_model_->data_,
                Product(dimen) * sizeof(float)));
    shaper_.AddShape(name, dimen);
    // RegisterOperand(name, index);
    return index;
}

ModelBuilder::Index ModelBuilder::AddTensorFromBuffer(const string &name, const void *buffer, const OperandType &operand_type) {
    DNN_ASSERT(!operand_type.dimensions.empty(), "");
    DNN_ASSERT(!isScalarType(operand_type.type), "");
    size_t element_size;
    switch (operand_type.type) {
        case Type::TENSOR_BOOL8:
            element_size = 1;
            break;
        case Type::TENSOR_FLOAT16:
            element_size = 2;
            break;
        case Type::TENSOR_FLOAT32:
            element_size = 4;
            break;
        case Type::TENSOR_INT32:
            element_size = 4;
            break;
        case Type::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            element_size = 1;
            break;
        case Type::TENSOR_QUANT8_ASYMM:
            element_size = 1;
            break;
        case Type::TENSOR_QUANT16_SYMM:
            element_size = 2;
            break;
        case Type::TENSOR_QUANT16_ASYMM:
            element_size = 2;
            break;
        default:
            throw std::invalid_argument("Wrong type: " + typeToStr(operand_type.type));
    }
    uint32_t index = AddNewOperand(operand_type);
    THROW_ON_ERROR(ANeuralNetworksModel_setOperandValue(dnn_model_->model_, index, buffer, Product(operand_type.dimensions) * element_size));
    shaper_.AddShape(name, operand_type.dimensions);
    RegisterOperand(name, index, operand_type);
    return index;
}

std::unique_ptr<Model> ModelBuilder::Compile(uint32_t preference) {
    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksModel_identifyInputsAndOutputs(
                dnn_model_->model_,
                static_cast<uint32_t>(input_index_vec_.size()), &input_index_vec_[0],
                static_cast<uint32_t>(output_index_vec_.size()), &output_index_vec_[0]
                ), 
            "on identifyInputsAndOutputs");

    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksModel_finish(
                dnn_model_->model_
                ),
            "on model finish");

    ;
    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksCompilation_create(
                dnn_model_->model_, &dnn_model_->compilation_
                ),
            "on create");

    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksCompilation_setPreference(
                dnn_model_->compilation_, preference
                ),
            "on setPreference");

    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksCompilation_finish(
                dnn_model_->compilation_
                ),
            "on compilation finish");

    VLOG(5) << "Finishing.. Here are operands in the model:";
    for (const auto &name : ordered_operands_) {
        VLOG(5) << name << ": " << shaper_[name];
    }
    operand_indexes_.clear();
    ordered_operands_.clear();
    shaper_.Clear();
    return std::move(dnn_model_);
}

void ModelBuilder::RegisterBufferPointer(std::unique_ptr<uint8_t[]> &&pointer) {
    dnn_model_->uint8_buf_pointers_.push_back(std::move(pointer));
}

void ModelBuilder::RegisterBufferPointer(std::unique_ptr<float[]> &&pointer) {
    dnn_model_->float_buf_pointers_.push_back(std::move(pointer));
}

void ModelBuilder::RegisterBufferPointer(std::unique_ptr<int8_t[]> &&pointer) {
    dnn_model_->int8_buf_pointers_.push_back(std::move(pointer));
}

void ModelBuilder::RegisterBufferPointer(std::unique_ptr<int32_t[]> &&pointer) {
    dnn_model_->int32_buf_pointers_.push_back(std::move(pointer));
}

ModelBuilder::IndexSeq ModelBuilder::GetInputIndexes() {
    return input_index_vec_;
}

ModelBuilder::IndexSeq ModelBuilder::GetOutputIndexes() {
    return output_index_vec_;
}

ModelBuilder::Index ModelBuilder::GetBlobIndex(const string &blobName) {
    return operand_indexes_.at(blobName);
}

#define DEFINE_FILL_OPERAND(val_type, op_type)  \
ModelBuilder::Index ModelBuilder::FillOperand(css &name, const OperandType &operand_type, const val_type val) {   \
    DNN_ASSERT(operand_type.type == Type::TENSOR_##op_type, "");  \
    auto buf = std::unique_ptr<val_type[]>(new val_type[Product(operand_type.dimensions)]);   \
    for (size_t i = 0; i < Product(operand_type.dimensions); i++) { \
        buf[i] = val;   \
    }   \
    auto idx = AddTensorFromBuffer(name, buf.get(), {Type::TENSOR_##op_type, operand_type.dimensions});  \
    RegisterBufferPointer(std::move(buf));  \
    return idx; \
}

DEFINE_FILL_OPERAND(float, FLOAT32);
DEFINE_FILL_OPERAND(int32_t, INT32);

#undef DEFINE_FILL_OPERAND

ModelBuilder::Shape ModelBuilder::GetBlobDim(const string &blobName) {
    return shaper_[blobName];
}

ModelBuilder::Shape ModelBuilder::GetBlobDim(uint32_t index) {
    for (const auto &p : operand_indexes_) {
        VLOG(5) << p.second;
        if (p.second == index) {
            return shaper_[p.first];
        }
    }
    throw std::invalid_argument("Wrong index in GetBlobDim");
}

string ModelBuilder::GetErrorCause(int errorCode) {
    switch (errorCode) {
        case ANEURALNETWORKS_OUT_OF_MEMORY:
            return "Out of memory";
        case ANEURALNETWORKS_BAD_DATA:
            return "Bad data";
        case ANEURALNETWORKS_BAD_STATE:
            return "Bad state";
        case ANEURALNETWORKS_INCOMPLETE:
            return "Incomplete";
        case ANEURALNETWORKS_UNEXPECTED_NULL:
            return "Unexpected null";
        case ANEURALNETWORKS_OP_FAILED:
            return "Op failed";
        case ANEURALNETWORKS_UNMAPPABLE:
            return "Unmappable";
        case ANEURALNETWORKS_NO_ERROR:
            return "No error";
        default:
            return "Unknown error code";
    }
}

template<typename... OperandTypes>
ModelBuilder::IndexSeq ModelBuilder::AddOperation(int op, IndexSeq input_indexes, OperandTypes... operand_types) {
    using android::nn::wrapper::OperandType;
    vector<OperandType> types;
    (types.push_back(operand_types), ...);
    IndexSeq output_indexes;
    for (const auto &type : types) {
        auto index = AddNewOperand(type);
        output_indexes.push_back(index);
    }

    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksModel_addOperation(
                dnn_model_->model_, op, input_indexes.size(), &input_indexes[0],
                output_indexes.size(), &output_indexes[0]),
            "op = " + std::to_string(op));
    
    return output_indexes;
}

void ModelBuilder::Prepare() {
    dnn_model_ = std::unique_ptr<Model>(new Model());
    const auto ret = ANeuralNetworksModel_create(&dnn_model_->model_);
    if (ret == ANEURALNETWORKS_OUT_OF_MEMORY) {
        throw std::bad_alloc();
    }
}

void ModelBuilder::SetMemory(int fd, size_t size, size_t offset) {
    ANeuralNetworksMemory *mem = nullptr;
    THROW_ON_ERROR(ANeuralNetworksMemory_createFromFd(size, PROT_READ, fd, offset, &mem));
    dnn_model_->memory_ = mem;
}

void ModelBuilder::SetBasePtr(uint8_t *data) {
    dnn_model_->data_ = data;
}

ModelBuilder &ModelBuilder::AddOutput(const std::string &name) {
    output_index_vec_.push_back(GetBlobIndex(name));
    dnn_model_->AddOutput(name, shaper_[name]);
    return *this;
}

#if __ANDROID_API__ >= __ANDROID_API_P__
ModelBuilder &ModelBuilder::AllowFp16(const bool allowed) {
    ANeuralNetworksModel_relaxComputationFloat32toFloat16(dnn_model_->model_, allowed);
    return *this;
}
#endif
