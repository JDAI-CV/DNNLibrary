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
                std::string(__func__) + "error, ret: " + getErrorCause(val));   \
    }   \

#define THROW_ON_ERROR_WITH_NOTE(val, note) \
    if ((val) != ANEURALNETWORKS_NO_ERROR) {  \
        throw std::invalid_argument(std::string("Error in ") + __FILE__ + std::string(":") + \
                std::to_string(__LINE__) + std::string(", function name: ") + \
                std::string(__func__) + "error, ret: " + getErrorCause(val) + std::string(", ") + (note));   \
    }   \

using std::vector; using std::ifstream; using std::streamsize; using std::string; using std::ios;
using std::stringstream; using std::array;

void ModelBuilder::AppendOperandIndex(const std::string &name, ModelBuilder::Index index) {
    operand_indexes[name] = index;
    ordered_operands.push_back(name);
}

ModelBuilder::Index ModelBuilder::addInput(string name, uint32_t height, uint32_t width, uint32_t depth) {
    vector<uint32_t> dimen{1, width, height, depth};
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);

    shaper.AddShape(name, dimen);
    inputIndexVector.push_back(index);
    dnn_model_->addInput(name, shaper[name]);
    AppendOperandIndex(name, index);
    return index;
}

ModelBuilder::Index ModelBuilder::addSpaceToBatchND(const std::string &input_name, const std::vector<int32_t> &block_sizes,
        const std::vector<int32_t> &pads, const std::string &output_name) {
    auto input = operand_indexes[input_name];

    auto block_sizes_idx = addTensorFromBuffer(output_name + "_bs", &block_sizes[0], Shape{static_cast<uint32_t>(block_sizes.size())});
    auto pads_idx = addTensorFromBuffer(output_name + "_pad", &pads[0], Shape{static_cast<uint32_t>(pads.size()) / 2, 2});

    shaper.SpaceToBatch(input_name, block_sizes, pads, output_name);
    IndexSeq input_indexes{input, block_sizes_idx, pads_idx};
    auto output_index = addOperation(ANEURALNETWORKS_SPACE_TO_BATCH_ND, input_indexes, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::addBatchToSpaceND(const std::string &input_name, const std::vector<int32_t> &block_sizes,
        const std::string &output_name) {
    auto input = operand_indexes[input_name];

    auto block_sizes_idx = addTensorFromBuffer(output_name + "_bs", &block_sizes[0], Shape{static_cast<uint32_t>(block_sizes.size())});

    shaper.BatchToSpace(input_name, block_sizes, output_name);
    IndexSeq input_indexes{input, block_sizes_idx};
    auto output_index = addOperation(ANEURALNETWORKS_BATCH_TO_SPACE_ND, input_indexes, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::addDepthWiseConv(const string &input_name, int32_t strideX, int32_t strideY,
                                                   int32_t paddingLeft,
                                                   int32_t paddingRight, int32_t paddingBottom, int32_t paddingTop,
                                                   int32_t activation,
                                                   int32_t depthMultiplier, const string &weight_name,
                                                   const std::optional<string> &bias_name,
                                                   const string &output_name) {
    auto input = operand_indexes[input_name];
    auto weight = operand_indexes[weight_name];

    uint32_t biasIndexValue;
    if (!bias_name.has_value()) {
        Shape weightDimen = shaper[weight_name];     // 1, height, width, num_output
        Shape bias_dims = Shape{weightDimen[3]};
        biasIndexValue = addFloat32ZeroOperandWithDims(bias_dims);
    } else {
        biasIndexValue = operand_indexes[bias_name.value()];
    }
    shaper.DepthwiseConv(input_name, strideX, strideY, 1, 1, paddingLeft, paddingRight, paddingTop, paddingBottom, weight_name, output_name);
    IndexSeq input_indexes{input, weight, biasIndexValue};
    addOperands(input_indexes, paddingLeft, paddingRight, paddingTop, paddingBottom,
                strideX, strideY, depthMultiplier, activation);
    auto output_index = addOperation(ANEURALNETWORKS_DEPTHWISE_CONV_2D, input_indexes, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index
ModelBuilder::addConv(const string &input_name, int32_t strideX, int32_t strideY, int32_t paddingLeft,
                      int32_t paddingRight,
                      int32_t paddingTop, int32_t paddingBottom, int32_t activation, const string &weight_name,
                      const std::optional<string> &bias_name, const string &output_name) {
    auto input = operand_indexes[input_name];
    auto weight = operand_indexes[weight_name];

    uint32_t biasIndexValue;
    if (!bias_name.has_value()) {
        Shape weightDimen = shaper[weight_name];     // num_output, height, width, num_input
        Shape bias_dims = Shape{weightDimen[0]};
        biasIndexValue = addFloat32ZeroOperandWithDims(bias_dims);
    } else {
        biasIndexValue = operand_indexes[bias_name.value()];
    }
    shaper.Conv(input_name, strideX, strideY, 1, 1, paddingLeft, paddingRight, paddingTop, paddingBottom, weight_name, output_name);
    IndexSeq input_indexes{input, weight, biasIndexValue};
    addOperands(input_indexes, paddingLeft, paddingRight, paddingTop, paddingBottom, strideX, strideY, activation);
    auto output_index = addOperation(ANEURALNETWORKS_CONV_2D, input_indexes, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

#if __ANDROID_API__ >= __ANDROID_API_P__

ModelBuilder::Index
ModelBuilder::addStridedSlice(const string &input_name, const vector<int32_t> &starts, const vector<int32_t> &ends,
                              const vector<int32_t> &strides, int32_t beginMask, int32_t endMask,
                              int32_t shrinkAxisMask, const string &output_name) {

    auto input = operand_indexes[input_name];

    uint32_t startsIndex = addTensorFromBuffer(output_name + "_starts", &starts[0], vector<uint32_t>{static_cast<uint32_t>(starts.size())});
    uint32_t endsIndex = addTensorFromBuffer(output_name + "_ends", &ends[0], vector<uint32_t>{static_cast<uint32_t>(ends.size())});
    uint32_t stridesIndex = addTensorFromBuffer(output_name + "_strides", &strides[0],
                                                   vector<uint32_t>{static_cast<uint32_t>(strides.size())});

    shaper.StridedSlice(input_name, starts, ends, strides, beginMask, endMask, shrinkAxisMask, output_name);
    IndexSeq input_indexes{input, startsIndex, endsIndex, stridesIndex};
    addOperands(input_indexes, beginMask, endMask, shrinkAxisMask);

    auto output_index = addOperation(ANEURALNETWORKS_STRIDED_SLICE, input_indexes, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

#endif

ModelBuilder::Index ModelBuilder::addPool(const string &input_name, int32_t strideX, int32_t strideY,
                                          int32_t paddingLeft, int32_t paddingRight,
                                          int32_t paddingTop, int32_t paddingBottom, int32_t height, int32_t width,
                                          int32_t activation,
                                          uint32_t poolingType, const string &output_name) {
    auto input = operand_indexes[input_name];

    if (height == -1 && width == -1) {
        LOG(INFO) << "Global pool, input: " << input_name;
        auto inputDimen = shaper[input_name];
        height = inputDimen[1];
        width = inputDimen[2];
        strideX = width;
        strideY = height;
    }
    shaper.Pool(input_name, strideX, strideY, paddingLeft, paddingRight, paddingTop, paddingBottom,
            height, width, output_name);
    IndexSeq input_indexes{input};
    addOperands(input_indexes, 
            paddingLeft, paddingRight, paddingTop, paddingBottom, 
            strideX, strideY, width, height, activation);

    Index output_index;
    if (poolingType == MAX_POOL) {  // TODO: use strong typed enum here
        output_index = addOperation(ANEURALNETWORKS_MAX_POOL_2D, input_indexes, shaper[output_name])[0];
    } else if (poolingType == AVE_POOL) {
        output_index = addOperation(ANEURALNETWORKS_AVERAGE_POOL_2D, input_indexes, shaper[output_name])[0];
    }
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::addSoftMax(const string &input_name, float beta, const string &output_name) {
    auto input = operand_indexes[input_name];

    shaper.Softmax(input_name, output_name);
    IndexSeq input_indexes{input};
    addOperands(input_indexes, beta);

    auto output_index = addOperation(ANEURALNETWORKS_SOFTMAX, input_indexes, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::addReLU(const string &input_name, const string &output_name) {
    auto input = operand_indexes[input_name];

    shaper.Relu(input_name, output_name);
    IndexSeq input_indexes{input};

    auto output_index = addOperation(ANEURALNETWORKS_RELU, input_indexes, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::addConcat(const vector<string> &input_names, uint32_t axis, const string &output_name) {
    IndexSeq inputs;
    for (const auto &input_name : input_names) {
        inputs.push_back(operand_indexes[input_name]);
    }

    shaper.Concat(input_names, axis, output_name);
    IndexSeq input_indexes(inputs);
    addOperands(input_indexes, axis);

    auto output_index = addOperation(ANEURALNETWORKS_CONCATENATION, input_indexes, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::addLRN(const string &input_name, uint32_t local_size, float bias, float alpha,
                                         float beta,
                                         const string &output_name) {
    auto input = operand_indexes[input_name];

    shaper.LRN(input_name, output_name);
    IndexSeq input_indexes{input};
    addOperands(input_indexes, local_size, bias, alpha, beta);

    auto output_idx = addOperation(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION, input_indexes, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_idx);
    return output_idx;
}

ModelBuilder::Index ModelBuilder::addFC(const string &input_name, int32_t activation,
                                        const string &weight_name, const std::optional<string> &bias_name,
                                        const string &output_name) {
    auto input = operand_indexes[input_name];
    auto weight = operand_indexes[weight_name];
    uint32_t biasIndexValue;
    if (!bias_name.has_value()) {
        auto weightDimen = shaper[weight_name];
        Shape bias_dims = Shape{weightDimen[0]};
        biasIndexValue = addFloat32ZeroOperandWithDims(bias_dims);
    } else {
        biasIndexValue = operand_indexes[bias_name.value()];
    }
    shaper.FC(input_name, weight_name, output_name);
    IndexSeq input_indexes{input, weight, biasIndexValue};
    addOperands(input_indexes, activation);
    auto output_idx = addOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indexes, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_idx);
    return output_idx;
}

ModelBuilder::Index ModelBuilder::addAddScalar(const string &input_name, float scalar, string output_name) {
    auto input = operand_indexes[input_name];
    uint32_t scalarIndex = addFloat32AsTensorOperand(scalar);
    IndexSeq inputOperands{input, scalarIndex, addOperand(
            ModelBuilder::ACTIVATION_NONE)};
    shaper.Eltwise(input_name, output_name);
    auto output_index = addOperation(ANEURALNETWORKS_ADD, inputOperands, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::addAddTensor(const string &input1_name, const string &input2_name,
                                               const string &output_name) {
    auto input1 = operand_indexes[input1_name];
    auto input2 = operand_indexes[input2_name];
    shaper.Eltwise(input1_name, input2_name, output_name);
    IndexSeq input_indexes{input1, input2};
    addOperands(input_indexes, ModelBuilder::ACTIVATION_NONE);
    auto output_idx = addOperation(ANEURALNETWORKS_ADD, input_indexes, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_idx);
    return output_idx;
}

ModelBuilder::Index ModelBuilder::addMulScalar(const string &input_name, float scalar, const string &output_name) {
    auto input = operand_indexes[input_name];
    Index scalarIndex = addFloat32AsTensorOperand(scalar);
    IndexSeq inputOperands{input, scalarIndex, addOperand(
            ModelBuilder::ACTIVATION_NONE)};

    shaper.Eltwise(input_name, output_name);
    auto output_index = addOperation(ANEURALNETWORKS_MUL, inputOperands, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::addMulTensor(const string &input1_name, const string &input2_name,
                                               const string &output_name) {
    auto input1 = operand_indexes[input1_name];
    auto input2 = operand_indexes[input2_name];
    IndexSeq input_indexes{input1, input2};
    addOperands(input_indexes, ModelBuilder::ACTIVATION_NONE);
    auto output_idx = addOperation(ANEURALNETWORKS_MUL, input_indexes, shaper[output_name])[0];
    AppendOperandIndex(output_name, output_idx);
    return output_idx;
}
//--------------------------------------------------------------------------------------------------//

ANeuralNetworksOperandType ModelBuilder::getInt32OperandTypeWithDims(Shape &dims) {
    ANeuralNetworksOperandType type;
    type.type = ANEURALNETWORKS_TENSOR_INT32;
    type.scale = 0.f;
    type.zeroPoint = 0;
    type.dimensionCount = static_cast<uint32_t>(dims.size());
    type.dimensions = &dims[0];

    return type;
}

ANeuralNetworksOperandType ModelBuilder::getFloat32OperandTypeWithDims(Shape &dims) {
    ANeuralNetworksOperandType type;
    type.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    type.scale = 0.f;
    type.zeroPoint = 0;
    type.dimensionCount = static_cast<uint32_t>(dims.size());
    type.dimensions = &dims[0];

    return type;
}

ANeuralNetworksOperandType ModelBuilder::getInt32OperandType() {
    ANeuralNetworksOperandType type;
    type.type = ANEURALNETWORKS_INT32;
    type.scale = 0.f;
    type.zeroPoint = 0;
    type.dimensionCount = 0;
    type.dimensions = nullptr;

    return type;
}

ANeuralNetworksOperandType ModelBuilder::getFloat32OperandType() {
    ANeuralNetworksOperandType type;
    type.type = ANEURALNETWORKS_FLOAT32;
    type.scale = 0.f;
    type.zeroPoint = 0;
    type.dimensionCount = 0;
    type.dimensions = nullptr;

    return type;
}

ModelBuilder::Index ModelBuilder::addOperand(uint32_t value) {
    if (uint32OperandMap.find(value) == uint32OperandMap.end()) {
        ANeuralNetworksOperandType type = getInt32OperandType();
        uint32_t index = addNewOperand(&type);
        THROW_ON_ERROR_WITH_NOTE(ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, &value, sizeof(value)), 
                "value: " + std::to_string(value));
        uint32OperandMap[value] = index;
    }
    return uint32OperandMap[value];
}

ModelBuilder::Index ModelBuilder::addOperand(int32_t value) {
    if (int32OperandMap.find(value) == int32OperandMap.end()) {
        ANeuralNetworksOperandType type = getInt32OperandType();
        uint32_t index = addNewOperand(&type);
        THROW_ON_ERROR_WITH_NOTE(ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, &value, sizeof(value)),
                "value: " + std::to_string(value));
        int32OperandMap[value] = index;
    }
    return int32OperandMap[value];
}

ModelBuilder::Index ModelBuilder::addOperand(float value) {
    if (float32OperandMap.find(value) == float32OperandMap.end()) {
        ANeuralNetworksOperandType type = getFloat32OperandType();
        uint32_t index = addNewOperand(&type);
        THROW_ON_ERROR_WITH_NOTE(ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, &value, sizeof(value)),
                "value: " + std::to_string(value));
        float32OperandMap[value] = index;
    }
    return float32OperandMap[value];

}

ModelBuilder::Index ModelBuilder::addFloat32AsTensorOperand(float value) {
    if (float32AsTensorOperandMap.find(value) == float32AsTensorOperandMap.end()) {
        /**
         * The `dims` variable mustn't be destoried before `addNewOperand`,
         * because ANeuralNetworksOperandType is only a struct storing a pointer to dims[0]
         */
        auto dims = Shape{1};
        auto type = getFloat32OperandTypeWithDims(dims);
        uint32_t index = addNewOperand(&type);
        THROW_ON_ERROR_WITH_NOTE(ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, &value, sizeof(value)),
                "value: " + std::to_string(value));
        float32AsTensorOperandMap[value] = index;
    }
    return float32AsTensorOperandMap[value];

}

ModelBuilder::Index ModelBuilder::addInt32NullOperand() {
    if (missingInt32OperandIndex == UINT32_MAX) {
        ANeuralNetworksOperandType type = getInt32OperandType();
        missingInt32OperandIndex = addNewOperand(&type);
        THROW_ON_ERROR(ANeuralNetworksModel_setOperandValue(dnn_model_->model, missingInt32OperandIndex, nullptr, 0));
    }
    return missingInt32OperandIndex;
}

ModelBuilder::Index ModelBuilder::addFloat32NullOperand() {
    if (missingFloat32OperandIndex == UINT32_MAX) {
        ANeuralNetworksOperandType type = getFloat32OperandType();
        missingFloat32OperandIndex = addNewOperand(&type);
        THROW_ON_ERROR(ANeuralNetworksModel_setOperandValue(dnn_model_->model, missingFloat32OperandIndex, nullptr, 0));
    }
    return missingFloat32OperandIndex;
}

ModelBuilder::Index ModelBuilder::addNewOperand(ANeuralNetworksOperandType *type) {
    THROW_ON_ERROR(ANeuralNetworksModel_addOperand(dnn_model_->model, type));
    return nextIndex++;
}

ModelBuilder::Index ModelBuilder::addTensorFromMemory(const string &name, const unsigned char *addr, Shape dimen) {
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);
    THROW_ON_ERROR(ANeuralNetworksModel_setOperandValueFromMemory(
                dnn_model_->model, index, dnn_model_->memory, addr - dnn_model_->data,
                product(dimen) * sizeof(float)));
    shaper.AddShape(name, dimen);
    AppendOperandIndex(name, index);
    return index;
}

ModelBuilder::Index ModelBuilder::addTensorFromBuffer(const string &name, const float *buffer, Shape dimen) {
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);
    THROW_ON_ERROR(ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, buffer, product(dimen) * sizeof(float)));
    shaper.AddShape(name, dimen);
    AppendOperandIndex(name, index);
    return index;
}

ModelBuilder::Index ModelBuilder::addTensorFromBuffer(const string &name, const int32_t *buffer,
                                                      Shape dimen) {
    ANeuralNetworksOperandType type = getInt32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);
    THROW_ON_ERROR(ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, buffer, product(dimen) * sizeof(int32_t)));
    shaper.AddShape(name, dimen);
    AppendOperandIndex(name, index);
    return index;
}

void ModelBuilder::compile(uint32_t preference) {
    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksModel_identifyInputsAndOutputs(
                dnn_model_->model,
                static_cast<uint32_t>(inputIndexVector.size()), &inputIndexVector[0],
                static_cast<uint32_t>(outputIndexVector.size()), &outputIndexVector[0]
                ), 
            "on identifyInputsAndOutputs");

    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksModel_finish(
                dnn_model_->model
                ),
            "on model finish");

    ;
    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksCompilation_create(
                dnn_model_->model, &dnn_model_->compilation
                ),
            "on create");

    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksCompilation_setPreference(
                dnn_model_->compilation, preference
                ),
            "on setPreference");

    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksCompilation_finish(
                dnn_model_->compilation
                ),
            "on compilation finish");
}

void ModelBuilder::registerBufferPointer(std::unique_ptr<float[]> &&pointer) {
    dnn_model_->floatBufPointers.push_back(std::move(pointer));
}

void ModelBuilder::registerBufferPointer(std::unique_ptr<char[]> &&pointer) {
    dnn_model_->charBufPointers.push_back(std::move(pointer));
}

ModelBuilder::IndexSeq ModelBuilder::getInputIndexes() {
    return inputIndexVector;
}

ModelBuilder::IndexSeq ModelBuilder::getOutputIndexes() {
    return outputIndexVector;
}

ModelBuilder::Index ModelBuilder::getBlobIndex(const string &blobName) {
    return operand_indexes.at(blobName);
}

ModelBuilder::Index ModelBuilder::addFloat32NullOperandWithDims(Shape &dims) {
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dims);
    uint32_t index = addNewOperand(&type);
    ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, nullptr, 0);
    return index;
}

ModelBuilder::Index ModelBuilder::addFloat32ZeroOperandWithDims(Shape &dims) {
    auto zeros = std::unique_ptr<float[]>(new float[product(dims)]);
    for (size_t i = 0; i < product(dims); i++) {
        zeros[i] = 0;
    }
    auto idx = addTensorFromBuffer(std::string(), zeros.get(), dims);
    registerBufferPointer(std::move(zeros));
    return idx;
}

ModelBuilder::Shape ModelBuilder::getBlobDim(const string &blobName) {
    return shaper[blobName];
}

ModelBuilder::Shape ModelBuilder::getBlobDim(uint32_t index) {
    for (const auto &p : operand_indexes) {
        LOG(INFO) << p.second;
        if (p.second == index) {
            return shaper[p.first];
        }
    }
    throw std::invalid_argument("Wrong index in getBlobDim");
}

string ModelBuilder::getErrorCause(int errorCode) {
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

template<typename... Shapes>
ModelBuilder::IndexSeq ModelBuilder::addOperation(int op, IndexSeq input_indexes, Shapes... shapes) {
    vector<Shape> shape_vec;
    (shape_vec.push_back(shapes), ...);
    IndexSeq output_indexes;
    for (auto shape : shape_vec) {
        ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(shape);
        auto index = addNewOperand(&type);
        output_indexes.push_back(index);
    }

    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksModel_addOperation(
                dnn_model_->model, op, input_indexes.size(), &input_indexes[0],
                output_indexes.size(), &output_indexes[0]),
            "op = " + std::to_string(op));
    
    return output_indexes;
}

void ModelBuilder::prepare() {
    dnn_model_ = std::make_unique<Model>();
    auto ret = ANeuralNetworksModel_create(&dnn_model_->model);
    if (ret == ANEURALNETWORKS_OUT_OF_MEMORY) {
        throw std::bad_alloc();
    }
}

void ModelBuilder::setMemory(int fd, size_t size, size_t offset) {
    ANeuralNetworksMemory *mem = nullptr;
    THROW_ON_ERROR(ANeuralNetworksMemory_createFromFd(size, PROT_READ, fd, offset, &mem));
    dnn_model_->memory = mem;
}

void ModelBuilder::setBuffer(unsigned char *data) {
    dnn_model_->data = data;
}

std::unique_ptr<Model> ModelBuilder::finish() {
    LOG(INFO) << "Finishing.. Here are operands in the model:";
    for (const auto &name : ordered_operands) {
        LOG(INFO) << name << ": " << shaper[name];
    }
    operand_indexes.clear();
    ordered_operands.clear();
    shaper.clear();
    return std::move(dnn_model_);
}

void ModelBuilder::addOutput(const std::string &name) {
    outputIndexVector.push_back(getBlobIndex(name));
    dnn_model_->addOutput(name, shaper[name]);
}
