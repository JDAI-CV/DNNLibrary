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
#include <operand_helper.h>
#include <ModelBuilder.h>


using std::vector; using std::ifstream; using std::streamsize; using std::string; using std::ios;
using std::stringstream; using std::array;

#define ADD_OPERATION(operation, input_indexes, output_indexes) \
ANeuralNetworksModel_addOperation(dnn_model_->model, ANEURALNETWORKS_##operation, static_cast<uint32_t>(input_indexes.size()),  \
    &input_indexes[0], static_cast<uint32_t>(output_indexes.size()), &output_indexes[0]);

// Simplest model for test
ModelBuilder &ModelBuilder::simplestModel() {
    auto input = addInput("input", 4, 3, 2);
    auto add = addAddScalar("input", 1.5f, "add");
    addIndexIntoOutput(add);
    return *this;
}

ModelBuilder::Index ModelBuilder::addInput(string name, uint32_t height, uint32_t width, uint32_t depth) {
    vector<uint32_t> dimen{1, width, height, depth};
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);

    dimensMap[index] = dimen;
    inputIndexVector.push_back(index);
    operand_indexes[name] = index;
    return index;
}

ModelBuilder::Index ModelBuilder::addSpaceToBatchND(const std::string &input_name, const std::vector<int32_t> &block_sizes,
        const std::vector<int32_t> &pads, const std::string &output_name) {
    auto input = operand_indexes[input_name];

    auto block_sizes_idx = addTensorFromBuffer(output_name + "_bs", &block_sizes[0], Shape{static_cast<uint32_t>(block_sizes.size())});
    auto pads_idx = addTensorFromBuffer(output_name + "_pad", &pads[0], Shape{static_cast<uint32_t>(pads.size()) / 2, 2});

    auto input_dimen = dimensMap[input];
    auto output_dimen = {input_dimen[0] * product(block_sizes), (input_dimen[1] + pads[0] + pads[1]) / block_sizes[0], 
        (input_dimen[2] + pads[2] + pads[3]) / block_sizes[1], input_dimen[3]};
    IndexSeq input_indexes{input, block_sizes_idx, pads_idx};
    auto output_index = addOperation(ANEURALNETWORKS_SPACE_TO_BATCH_ND, input_indexes, output_dimen)[0];
    operand_indexes[output_name] = output_index;
    return output_index;
}

ModelBuilder::Index ModelBuilder::addBatchToSpaceND(const std::string &input_name, const std::vector<int32_t> &block_sizes,
        const std::string &output_name) {
    auto input = operand_indexes[input_name];

    auto block_sizes_idx = addTensorFromBuffer(output_name + "_bs", &block_sizes[0], Shape{static_cast<uint32_t>(block_sizes.size())});

    auto input_dimen = dimensMap[input];
    auto output_dimen = {input_dimen[0] / product(block_sizes), input_dimen[1] * block_sizes[0], 
        input_dimen[2] * block_sizes[1], input_dimen[3]};
    IndexSeq input_indexes{input, block_sizes_idx};
    auto output_index = addOperation(ANEURALNETWORKS_BATCH_TO_SPACE_ND, input_indexes, output_dimen)[0];
    operand_indexes[output_name] = output_index;
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
    if (input >= nextIndex) return WRONG_INPUT;

    Shape weightDimen = dimensMap[weight];     // 1, height, width, num_output
    // NHWC
    Shape inputDimen = dimensMap[input];
    Shape outputDimen{1,
                      (inputDimen[1] - weightDimen[1] + paddingTop + paddingBottom) / strideY + 1,
                      (inputDimen[2] - weightDimen[2] + paddingLeft + paddingRight) / strideX + 1,
                      weightDimen[3]};
    uint32_t biasIndexValue;
    if (!bias_name.has_value()) {
        Shape bias_dims = Shape{weightDimen[0]};
        biasIndexValue = addFloat32ZeroOperandWithDims(bias_dims);
    } else {
        biasIndexValue = operand_indexes[bias_name.value()];
    }
    IndexSeq input_indexes{input, weight, biasIndexValue};
    addOperands(input_indexes, paddingLeft, paddingRight, paddingTop, paddingBottom,
                strideX, strideY, depthMultiplier, activation);
    auto output_index = addOperation(ANEURALNETWORKS_DEPTHWISE_CONV_2D, input_indexes, outputDimen)[0];
    operand_indexes[output_name] = output_index;
    return output_index;
}

ModelBuilder::Index
ModelBuilder::addConv(const string &input_name, int32_t strideX, int32_t strideY, int32_t paddingLeft,
                      int32_t paddingRight,
                      int32_t paddingTop, int32_t paddingBottom, int32_t activation, const string &weight_name,
                      const std::optional<string> &bias_name, const string &output_name) {
    auto input = operand_indexes[input_name];
    auto weight = operand_indexes[weight_name];
    if (input >= nextIndex) return WRONG_INPUT;

    Shape weightDimen = dimensMap[weight];     // num_output, height, width, num_input
    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];
    vector<uint32_t> outputDimen{1,
                                 (inputDimen[1] - weightDimen[1] + paddingTop + paddingBottom) / strideY + 1,
                                 (inputDimen[2] - weightDimen[2] + paddingLeft + paddingRight) / strideX + 1,
                                 weightDimen[0]};
    uint32_t biasIndexValue;
    if (!bias_name.has_value()) {
        Shape bias_dims = Shape{weightDimen[0]};
        biasIndexValue = addFloat32ZeroOperandWithDims(bias_dims);
    } else {
        biasIndexValue = operand_indexes[bias_name.value()];
    }
    IndexSeq input_indexes{input, weight, biasIndexValue};
    addOperands(input_indexes, paddingLeft, paddingRight, paddingTop, paddingBottom, strideX, strideY, activation);
    auto output_indexes = addOperation(ANEURALNETWORKS_CONV_2D, input_indexes, outputDimen);
    operand_indexes[output_name] = output_indexes[0];
    return output_indexes[0];
}

#if __ANDROID_API__ >= __ANDROID_API_P__

ModelBuilder::Index
ModelBuilder::addStridedSlice(const string &input_name, const vector<int32_t> &starts, const vector<int32_t> &ends,
                              const vector<int32_t> &strides, int32_t beginMask, int32_t endMask,
                              int32_t shrinkAxisMask, const string &output_name) {

    auto input = operand_indexes[input_name];
    if (input >= nextIndex) return WRONG_INPUT;

    uint32_t startsIndex = addTensorFromBuffer("", &starts[0], vector<uint32_t>{static_cast<uint32_t>(starts.size())});
    uint32_t endsIndex = addTensorFromBuffer("", &ends[0], vector<uint32_t>{static_cast<uint32_t>(ends.size())});
    uint32_t stridesIndex = addTensorFromBuffer("", &strides[0],
                                                   vector<uint32_t>{static_cast<uint32_t>(strides.size())});
    uint32_t beginMaskOperandIndex = addOperand(beginMask);
    uint32_t endMaskOperandIndex = addOperand(endMask);
    uint32_t shrinkAxisMaskOperandIndex = addOperand(shrinkAxisMask);

    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];
    vector<uint32_t> outputDimen;
    for (size_t i = 0; i < inputDimen.size(); i++) {
        if (shrinkAxisMask & (1 << i)) {
            continue;
        }
        int32_t start = starts[i], end = ends[i];
        if (beginMask & (1 << i)) {
            start = 0;
        }
        if (endMask & (1 << i)) {
            end = inputDimen[i];
        }
        outputDimen.emplace_back(end - start);
    }

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(outputDimen);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);

    dimensMap[outputOperandIndex] = outputDimen;

    array<uint32_t, 7> inputOperandsArr{{input, startsIndex, endsIndex, stridesIndex,
                                                beginMaskOperandIndex, endMaskOperandIndex, shrinkAxisMaskOperandIndex}};

    ANeuralNetworksModel_addOperation(dnn_model_->model, ANEURALNETWORKS_STRIDED_SLICE, 7, &inputOperandsArr[0], 1,
                                      &outputOperandIndex);
    operand_indexes[output_name] = outputOperandIndex;
    return outputOperandIndex;
}

#endif

ModelBuilder::Index ModelBuilder::addPool(const string &input_name, int32_t strideX, int32_t strideY,
                                          int32_t paddingLeft, int32_t paddingRight,
                                          int32_t paddingTop, int32_t paddingBottom, int32_t height, int32_t width,
                                          int32_t activation,
                                          uint32_t poolingType, const string &output_name) {
    auto input = operand_indexes[input_name];
    if (input >= nextIndex) return WRONG_INPUT;

    // NHWC
    auto inputDimen = dimensMap[input];

    Shape outputDimen{1,
                      (inputDimen[1] - height + paddingTop + paddingBottom) / strideY + 1,
                      (inputDimen[2] - width + paddingLeft + paddingRight) / strideX + 1,
                      inputDimen[3]};

    IndexSeq input_indexes{input};
    if (height == -1 && width == -1) {
        LOG(INFO) << "Global pool, input: " << input_name;
        height = inputDimen[1];
        width = inputDimen[2];
        strideX = width;
        strideY = height;
    }
    addOperands(input_indexes, 
            paddingLeft, paddingRight, paddingTop, paddingBottom, 
            strideX, strideY, width, height, activation);

    Index output_index;
    if (poolingType == MAX_POOL) {
        output_index = addOperation(ANEURALNETWORKS_MAX_POOL_2D, input_indexes, outputDimen)[0];
    } else if (poolingType == AVE_POOL) {
        output_index = addOperation(ANEURALNETWORKS_AVERAGE_POOL_2D, input_indexes, outputDimen)[0];
    } else {
        return WRONG_POOLING_TYPE;
    }
    operand_indexes[output_name] = output_index;
    return output_index;
}

ModelBuilder::Index ModelBuilder::addCaffePool(const string &input_name, int32_t strideX, int32_t strideY,
                                               int32_t paddingLeft, int32_t paddingRight,
                                               int32_t paddingTop, int32_t paddingBottom,
                                               int32_t height, int32_t width,
                                               int32_t activation, uint32_t poolingType,
                                               const string &output_name) {
    auto input = operand_indexes[input_name];
    if (input >= nextIndex) return WRONG_INPUT;

    // NHWC
    auto inputDimen = dimensMap[input];

    // https://github.com/BVLC/caffe/pull/473#issuecomment-45386156
    unsigned int extraY = (inputDimen[1] - height + paddingTop + paddingBottom) % strideY;
    if (extraY != 0) {
        paddingBottom += strideY - extraY;
    }
    unsigned int extraX = (inputDimen[2] - width + paddingLeft + paddingRight) % strideX;
    if (extraX != 0) {
        paddingRight += strideX - extraX;
    }

    Shape outputDimen{1,
                      (inputDimen[1] - height + paddingTop + paddingBottom) / strideY + 1,
                      (inputDimen[2] - width + paddingLeft + paddingRight) / strideX + 1,
                      inputDimen[3]};

    IndexSeq input_indexes{input};
    addOperands(input_indexes, 
            paddingLeft, paddingRight, paddingTop, paddingBottom, 
            strideX, strideY, width, height, activation);

    Index output_index;
    if (poolingType == MAX_POOL) {
        output_index = addOperation(ANEURALNETWORKS_MAX_POOL_2D, input_indexes, outputDimen)[0];
    } else if (poolingType == AVE_POOL) {
        output_index = addOperation(ANEURALNETWORKS_AVERAGE_POOL_2D, input_indexes, outputDimen)[0];
    } else {
        return WRONG_POOLING_TYPE;
    }
    operand_indexes[output_name] = output_index;
    return output_index;
}

ModelBuilder::Index ModelBuilder::addSoftMax(const string &input_name, float beta, const string &output_name) {
    auto input = operand_indexes[input_name];
    vector<uint32_t> dimen = dimensMap[input];

    IndexSeq input_indexes{input};
    addOperands(input_indexes, beta);

    auto output_index = addOperation(ANEURALNETWORKS_SOFTMAX, input_indexes, dimen)[0];
    operand_indexes[output_name] = output_index;
    return output_index;
}

ModelBuilder::Index ModelBuilder::addReLU(const string &input_name, const string &output_name) {
    auto input = operand_indexes[input_name];
    auto dimen = dimensMap[input];

    IndexSeq input_indexes{input};

    auto output_index = addOperation(ANEURALNETWORKS_RELU, input_indexes, dimen)[0];
    operand_indexes[output_name] = output_index;
    return output_index;
}

ModelBuilder::Index ModelBuilder::addConcat(const vector<string> &input_names, uint32_t axis, const string &output_name) {
    IndexSeq inputs;
    for (const auto &input_name : input_names) {
        inputs.push_back(operand_indexes[input_name]);
    }
    vector<Shape> dimens;
    for (const auto &input : inputs) {
        auto &dimen = dimensMap[input];
        if (!dimens.empty()) {
            for (size_t i = 0; i < dimens[0].size(); i++) {
                if (i == axis) continue;
                if (dimen[i] != dimens[0][i]) {
                    throw string("Wrong input for concat");
                }
            }
        }
        dimens.push_back(dimensMap[input]);
    }

    auto outputDimen = dimens[0];
    for (size_t i = 1; i < dimens.size(); i++) {
        outputDimen[axis] += dimens[i][axis];
    }

    IndexSeq input_indexes(inputs);
    addOperands(input_indexes, axis);

    auto output_index = addOperation(ANEURALNETWORKS_CONCATENATION, input_indexes, outputDimen)[0];
    operand_indexes[output_name] = output_index;
    return output_index;
}

ModelBuilder::Index ModelBuilder::addLRN(const string &input_name, uint32_t local_size, float bias, float alpha,
                                         float beta,
                                         const string &output_name) {
    auto input = operand_indexes[input_name];
    auto dimen = dimensMap[input];

    IndexSeq input_indexes{input};
    addOperands(input_indexes, local_size, bias, alpha, beta);

    auto output_idx = addOperation(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION, input_indexes, dimen)[0];
    operand_indexes[output_name] = output_idx;
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
        auto ret = ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, &value, sizeof(value));
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            throw std::invalid_argument("Add operand failed, value: " + std::to_string(value) + ", error " + getErrorCause(ret));
        }
        uint32OperandMap[value] = index;
    }
    return uint32OperandMap[value];
}

ModelBuilder::Index ModelBuilder::addOperand(int32_t value) {
    if (int32OperandMap.find(value) == int32OperandMap.end()) {
        ANeuralNetworksOperandType type = getInt32OperandType();
        uint32_t index = addNewOperand(&type);
        auto ret = ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, &value, sizeof(value));
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            throw std::invalid_argument("Add operand failed, value: " + std::to_string(value) + ", error " + getErrorCause(ret));
        }
        int32OperandMap[value] = index;
    }
    return int32OperandMap[value];
}

ModelBuilder::Index ModelBuilder::addOperand(float value) {
    if (float32OperandMap.find(value) == float32OperandMap.end()) {
        ANeuralNetworksOperandType type = getFloat32OperandType();
        uint32_t index = addNewOperand(&type);
        auto ret = ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, &value, sizeof(value));
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            throw std::invalid_argument("Add operand failed, value: " + std::to_string(value) + ", error " + getErrorCause(ret));
        }
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
        auto ret = ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, &value, sizeof(value));
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            throw std::invalid_argument("Add operand failed, value: " + std::to_string(value) + ", error " + getErrorCause(ret));
        }
        float32AsTensorOperandMap[value] = index;
    }
    return float32AsTensorOperandMap[value];

}

ModelBuilder::Index ModelBuilder::addInt32NullOperand() {
    if (missingInt32OperandIndex == UINT32_MAX) {
        ANeuralNetworksOperandType type = getInt32OperandType();
        missingInt32OperandIndex = addNewOperand(&type);
        auto ret = ANeuralNetworksModel_setOperandValue(dnn_model_->model, missingInt32OperandIndex, nullptr, 0);
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            throw std::invalid_argument("Add operand failed, value: null, error " + getErrorCause(ret));
        }
    }
    return missingInt32OperandIndex;
}

ModelBuilder::Index ModelBuilder::addFloat32NullOperand() {
    if (missingFloat32OperandIndex == UINT32_MAX) {
        ANeuralNetworksOperandType type = getFloat32OperandType();
        missingFloat32OperandIndex = addNewOperand(&type);
        auto ret = ANeuralNetworksModel_setOperandValue(dnn_model_->model, missingFloat32OperandIndex, nullptr, 0);
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            throw std::invalid_argument("Add operand failed, value: null, error " + getErrorCause(ret));
        }
    }
    return missingFloat32OperandIndex;
}

ModelBuilder::Index ModelBuilder::addNewOperand(ANeuralNetworksOperandType *type) {
    int ret;
    if ((ret = ANeuralNetworksModel_addOperand(dnn_model_->model, type)) != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Add new operand failed, error " + getErrorCause(ret));
    }
    return nextIndex++;
}

ModelBuilder::Index ModelBuilder::addTensorFromMemory(const string &name, const unsigned char *addr, Shape dimen) {
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);
    auto ret = ANeuralNetworksModel_setOperandValueFromMemory(dnn_model_->model, index, dnn_model_->memory, addr - dnn_model_->data,
                                                   product(dimen) * sizeof(float));
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("addTensorFromBuffer error, ret: " + getErrorCause(ret));
    }
    dimensMap[index] = dimen;
    operand_indexes[name] = index;
    return index;
}

ModelBuilder::Index ModelBuilder::addTensorFromBuffer(const string &name, const float *buffer, Shape dimen) {
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);
    auto ret = ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, buffer, product(dimen) * sizeof(float));
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("addTensorFromBuffer error, ret: " + getErrorCause(ret));
    }
    dimensMap[index] = dimen;
    operand_indexes[name] = index;
    return index;
}

ModelBuilder::Index ModelBuilder::addTensorFromBuffer(const string &name, const int32_t *buffer,
                                                      Shape dimen) {
    ANeuralNetworksOperandType type = getInt32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);
    auto ret = ANeuralNetworksModel_setOperandValue(dnn_model_->model, index, buffer, product(dimen) * sizeof(int32_t));
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("addTensorFromBuffer error, ret: " + getErrorCause(ret));
    }
    dimensMap[index] = dimen;
    operand_indexes[name] = index;
    return index;
}

void ModelBuilder::addIndexIntoOutput(Index index) {
    outputIndexVector.push_back(index);
}

int ModelBuilder::compile(uint32_t preference) {
    int ret;
    if ((ret = ANeuralNetworksModel_identifyInputsAndOutputs(
            dnn_model_->model,
            static_cast<uint32_t>(inputIndexVector.size()), &inputIndexVector[0],
            static_cast<uint32_t>(outputIndexVector.size()), &outputIndexVector[0])) != ANEURALNETWORKS_NO_ERROR) {

        return NN_IDENTIFY_IO | ret;
    }

    ret = ANeuralNetworksModel_finish(dnn_model_->model);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        return NN_MODEL_FINISH | ret;
    }

    ret = ANeuralNetworksCompilation_create(dnn_model_->model, &dnn_model_->compilation);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        return NN_CREATE | ret;
    }

    ret = ANeuralNetworksCompilation_setPreference(dnn_model_->compilation, preference);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        return NN_PREFERENCE | ret;
    }

    ret = ANeuralNetworksCompilation_finish(dnn_model_->compilation);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        return NN_COMP_FINISH | ret;
    }

    return 0;
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

ModelBuilder::Index ModelBuilder::addFC(const string &input_name, int32_t activation,
                                        const string &weight_name, const std::optional<string> &bias_name,
                                        const string &output_name) {
    auto input = operand_indexes[input_name];
    auto weight = operand_indexes[weight_name];
    Shape weightDimen = dimensMap[weight];     // num_units, input_size
    uint32_t biasIndexValue;
    if (!bias_name.has_value()) {
        Shape bias_dims = Shape{weightDimen[0]};
        biasIndexValue = addFloat32ZeroOperandWithDims(bias_dims);
    } else {
        biasIndexValue = operand_indexes[bias_name.value()];
    }
    IndexSeq input_indexes{input, weight, biasIndexValue};
    addOperands(input_indexes, activation);
    Shape outputDimen{1, weightDimen[0]};   // TODO: multiple batch size
    auto output_idx = addOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indexes, outputDimen)[0];
    operand_indexes[output_name] = output_idx;
    return output_idx;
}

ModelBuilder::Index ModelBuilder::getBlobIndex(const string &blobName) {
    return operand_indexes.at(blobName);
}

ModelBuilder::Index ModelBuilder::addAddScalar(const string &input_name, float scalar, string output_name) {
    auto input = operand_indexes[input_name];
    uint32_t scalarIndex = addFloat32AsTensorOperand(scalar);
    array<uint32_t, 3> inputOperands{{input, scalarIndex, addOperand(
            ModelBuilder::ACTIVATION_NONE)}};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(dimensMap[input]);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);
    dimensMap[outputOperandIndex] = dimensMap[input];

    ANeuralNetworksModel_addOperation(dnn_model_->model, ANEURALNETWORKS_ADD, 3, &inputOperands[0], 1, &outputOperandIndex);
    operand_indexes[output_name] = outputOperandIndex;
    return outputOperandIndex;
}

ModelBuilder::Index ModelBuilder::addAddTensor(const string &input1_name, const string &input2_name,
                                               const string &output_name) {
    auto input1 = operand_indexes[input1_name];
    auto input2 = operand_indexes[input2_name];
    IndexSeq input_indexes{input1, input2};
    addOperands(input_indexes, ModelBuilder::ACTIVATION_NONE);
    auto output_idx = addOperation(ANEURALNETWORKS_ADD, input_indexes, dimensMap[input1])[0];
    operand_indexes[output_name] = output_idx;
    return output_idx;
}

ModelBuilder::Index ModelBuilder::addMulScalar(const string &input_name, float scalar, const string &output_name) {
    auto input = operand_indexes[input_name];
    Index scalarIndex = addFloat32AsTensorOperand(scalar);
    array<uint32_t, 3> inputOperands{{input, scalarIndex, addOperand(
            ModelBuilder::ACTIVATION_NONE)}};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(dimensMap[input]);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);
    dimensMap[outputOperandIndex] = dimensMap[input];

    ANeuralNetworksModel_addOperation(dnn_model_->model, ANEURALNETWORKS_MUL, 3, &inputOperands[0], 1, &outputOperandIndex);
    operand_indexes[output_name] = outputOperandIndex;
    return outputOperandIndex;
}

ModelBuilder::Index ModelBuilder::addMulTensor(const string &input1_name, const string &input2_name,
                                               const string &output_name) {
    auto input1 = operand_indexes[input1_name];
    auto input2 = operand_indexes[input2_name];
    IndexSeq input_indexes{input1, input2};
    addOperands(input_indexes, ModelBuilder::ACTIVATION_NONE);
    auto output_idx = addOperation(ANEURALNETWORKS_MUL, input_indexes, dimensMap[input1])[0];
    operand_indexes[output_name] = output_idx;
    return output_idx;
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
    return dimensMap[getBlobIndex(blobName)];
}

ModelBuilder::Shape ModelBuilder::getBlobDim(uint32_t index) {
    return dimensMap[index];
}

string ModelBuilder::getErrorProcedure(int errorCode) {
    errorCode &= NN_PROCEDURE_MASK;
    switch (errorCode) {
        case NN_COMP_FINISH:
            return "compilation finish";
        case NN_PREFERENCE:
            return "set preference";
        case NN_CREATE:
            return "compilation create";
        case NN_MODEL_FINISH:
            return "model finish";
        case NN_IDENTIFY_IO:
            return "identify input and output";
        case 0:
            return "No error";
        default:
            return "Unknown error code";
    }
}

string ModelBuilder::getErrorCause(int errorCode) {
    errorCode &= NN_CAUSE_MASK;

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
        dimensMap[index] = shape;
    }

    auto ret = ANeuralNetworksModel_addOperation(dnn_model_->model, op, input_indexes.size(), &input_indexes[0],
                                      output_indexes.size(), &output_indexes[0]);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Add operation failed, op = " + std::to_string(op) + ", error " + getErrorCause(ret));
    }
    
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
    auto ret = ANeuralNetworksMemory_createFromFd(size, PROT_READ, fd, offset, &mem);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Error in setMemory, ret: " + getErrorCause(ret));
    }
    dnn_model_->memory = mem;
}

void ModelBuilder::setBuffer(unsigned char *data) {
    dnn_model_->data = data;
}

std::unique_ptr<Model> ModelBuilder::finish() {
    LOG(INFO) << "Finishing.. Here are operands in the model:";
    for (const auto &pair : operand_indexes) {
        LOG(INFO) << pair.first << ": " << dimensMap[operand_indexes[pair.first]];
    }
    return std::move(dnn_model_);
}
