//
// Created by daquexian on 2017/11/8.
//

#include <numeric>
#include "ModelBuilder.h"

using namespace std;

#define  LOG_TAG    "NNAPI Demo"

#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG,__VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)

uint32_t ModelBuilder::addInput(uint32_t height, uint32_t width) {
    vector<uint32_t> dimen{1, width, height, 1};
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t index = addOperand(&type);

    dimensMap[index] = dimen;
    inputIndexVector.push_back(index);
    return index;
}

uint32_t
ModelBuilder::addConv(std::string name, uint32_t input, uint32_t strideX, uint32_t strideY,
                      uint32_t paddingW, uint32_t paddingH, uint32_t height, uint32_t width,
                      uint32_t activation, uint32_t outputDepth) {
    if (input >= nextIndex) return WRONG_INPUT;

    uint32_t strideXOperandIndex = addUInt32Operand(strideX);
    uint32_t strideYOperandIndex = addUInt32Operand(strideY);
    uint32_t paddingWOperandIndex = addUInt32Operand(paddingW);
    uint32_t paddingHOperandIndex = addUInt32Operand(paddingH);
    uint32_t activationOperandIndex = addUInt32Operand(activation);

    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];

    uint32_t weightOperandIndex = addConvWeight(name, height, width, inputDimen[3], outputDepth);
    uint32_t biasOperandIndex = addBias(name, outputDepth);

    vector<uint32_t> outputDimen{1,
                                 (inputDimen[1] - height + 2 * paddingH) / strideY + 1,
                                 (inputDimen[2] - width + 2 * paddingW) / strideX + 1,
                                 outputDepth};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(outputDimen);
    uint32_t outputOperandIndex = addOperand(&outputBlobType);

    dimensMap[outputOperandIndex] = outputDimen;

    array<uint32_t, 10> inputOperandsArr{{input, weightOperandIndex, biasOperandIndex,
                                            paddingWOperandIndex, paddingWOperandIndex,
                                            paddingHOperandIndex, paddingHOperandIndex,
                                            strideXOperandIndex, strideYOperandIndex,
                                            activationOperandIndex}};

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 10, &inputOperandsArr[0], 1,
                                      &outputOperandIndex);
    return outputOperandIndex;
}

uint32_t ModelBuilder::addFC(std::string name, uint32_t input, uint32_t outputNum,
                             uint32_t activation) {
    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];

    uint32_t weightOperandIndex = addFcWeight(name, product(inputDimen),
                                              outputNum);
    uint32_t biasOperandIndex = addBias(name, outputNum);

    uint32_t activationOperandIndex = addUInt32Operand(activation);

    vector<uint32_t> outputDimen{1, outputNum};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(outputDimen);
    uint32_t outputOperandIndex = addOperand(&outputBlobType);

    dimensMap[outputOperandIndex] = outputDimen;

    array<uint32_t, 10> inputOperandsArr{{input, weightOperandIndex, biasOperandIndex,
                                                 activationOperandIndex}};

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED,
                                      4, &inputOperandsArr[0], 1, &outputOperandIndex);

    return outputOperandIndex;
}

uint32_t ModelBuilder::addPool(uint32_t input, uint32_t strideX, uint32_t strideY,
                               uint32_t paddingW, uint32_t paddingH,
                               uint32_t height, uint32_t width,
                               uint32_t activation, uint32_t poolingType) {

    if (input >= nextIndex) return WRONG_INPUT;

    uint32_t widthOperandIndex = addUInt32Operand(width);
    uint32_t heightOperandIndex = addUInt32Operand(height);
    uint32_t strideXOperandIndex = addUInt32Operand(strideX);
    uint32_t strideYOperandIndex = addUInt32Operand(strideY);
    uint32_t paddingWOperandIndex = addUInt32Operand(paddingW);
    uint32_t paddingHOperandIndex = addUInt32Operand(paddingH);
    uint32_t activationOperandIndex = addUInt32Operand(activation);

    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];
    vector<uint32_t> outputDimen{1,
                                 (inputDimen[1] - height + 2 * paddingH) / strideY + 1,
                                 (inputDimen[2] - width + 2 * paddingW) / strideX + 1,
                                 inputDimen[3]};

    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(outputDimen);
    uint32_t outputOperandIndex = addOperand(&type);

    dimensMap[outputOperandIndex] = outputDimen;

    array<uint32_t, 10> inputOperandsArr{{input, paddingWOperandIndex, paddingWOperandIndex,
                                                 paddingHOperandIndex, paddingHOperandIndex,
                                                 strideXOperandIndex, strideYOperandIndex,
                                                 widthOperandIndex, heightOperandIndex,
                                                 activationOperandIndex}};
    if (poolingType == MAX_POOL) {
        ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MAX_POOL_2D, 10,
                                          &inputOperandsArr[0], 1, &outputOperandIndex);
    } else if (poolingType == AVE_POOL) {
        ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_AVERAGE_POOL_2D, 10,
                                          &inputOperandsArr[0], 1, &outputOperandIndex);
    } else {
        return WRONG_POOLING_TYPE;
    }
    return outputOperandIndex;
}


ANeuralNetworksOperandType ModelBuilder::getFloat32OperandTypeWithDims(std::vector<uint32_t> &dims) {
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
    type.type = ANEURALNETWORKS_TENSOR_INT32;
    type.scale = 0.f;
    type.zeroPoint = 0;
    type.dimensionCount = 0;
    type.dimensions = nullptr;

    return type;
}

ANeuralNetworksOperandType ModelBuilder::getFloat32OperandType() {
    ANeuralNetworksOperandType type;
    type.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    type.scale = 0.f;
    type.zeroPoint = 0;
    type.dimensionCount = 0;
    type.dimensions = nullptr;

    return type;
}

/**
 * set operand value from file in assets
 * @param model
 * @param mgr A pointer to AAssetManager got from Java's AssetManager
 * @param index The index of operand
 * @param filename The filename of weight or bias
 * @return a pointer to the buffer of weights or bias, according to the doc, the buffer should not
 * be modified until all executions complete, so please delete the buffer after the executions
 * complete.
 */
char* ModelBuilder::setOperandValueFromAssets(ANeuralNetworksModel *model, AAssetManager *mgr,
                                              int32_t index, string filename) {
    AAsset* asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_UNKNOWN);
    size_t size = static_cast<size_t>(AAsset_getLength(asset));
    char* buffer = new char[size];
    AAsset_read(asset, buffer, static_cast<size_t>(size));
    ANeuralNetworksModel_setOperandValue(model, index, buffer, size);
    return buffer;
}

uint32_t ModelBuilder::addUInt32Operand(uint32_t value) {
    if (uint32OperandMap.find(value) == uint32OperandMap.end()) {
        ANeuralNetworksOperandType type = getInt32OperandType();
        uint32_t index = addOperand(&type);
        ANeuralNetworksModel_setOperandValue(model, index, &value, sizeof(value));
        uint32OperandMap[value] = index;
    }
    return uint32OperandMap[value];
}

uint32_t ModelBuilder::addFloat32Operand(float value) {
    if (float32OperandMap.find(value) == float32OperandMap.end()) {
        ANeuralNetworksOperandType type = getFloat32OperandType();
        uint32_t index = addOperand(&type);
        ANeuralNetworksModel_setOperandValue(model, index, &value, sizeof(value));
        float32OperandMap[value] = index;
    }
    return float32OperandMap[value];

}

uint32_t ModelBuilder::addOperand(ANeuralNetworksOperandType *type) {
    int ret;
    if ((ret = ANeuralNetworksModel_addOperand(model, type)) != ANEURALNETWORKS_NO_ERROR) {
        return UINT32_MAX - ret;
    }
    return nextIndex++;
}

uint32_t ModelBuilder::addConvWeight(std::string name, uint32_t height, uint32_t width,
                                     uint32_t inputDepth,
                                     uint32_t outputDepth) {
    vector<uint32_t> weightDimen{outputDepth, height, width, inputDepth};
    return addWeight(name, weightDimen);
}

uint32_t ModelBuilder::addFcWeight(std::string name, uint32_t inputSize, uint32_t outputNum) {
    vector<uint32_t> weightDimen{outputNum, inputSize};
    return addWeight(name, weightDimen);
}

uint32_t ModelBuilder::addBias(std::string name, uint32_t outputDepth) {
    vector<uint32_t> biasDimen{outputDepth};
    ANeuralNetworksOperandType biasType = getFloat32OperandTypeWithDims(biasDimen);
    uint32_t biasIndex = addOperand(&biasType);
    bufferPointers.push_back(setOperandValueFromAssets(
            model, mgr, biasIndex, "weights_and_biases/" + name + "_biases"));
    return biasIndex;
}

uint32_t ModelBuilder::addWeight(std::string name, std::vector<uint32_t> dimen) {
    ANeuralNetworksOperandType weightType = getFloat32OperandTypeWithDims(dimen);
    uint32_t weightIndex = addOperand(&weightType);
    bufferPointers.push_back(setOperandValueFromAssets(
            model, mgr, weightIndex, "weights_and_biases/" + name + "_weights"));

    return weightIndex;
}

int ModelBuilder::init(AAssetManager *mgr) {
    this->mgr = mgr;
    return ANeuralNetworksModel_create(&model);
}

void ModelBuilder::addIndexIntoOutput(uint32_t index) {
    outputIndexVector.push_back(index);
}

int ModelBuilder::compile(uint32_t preference) {
    ANeuralNetworksModel_identifyInputsAndOutputs(
            model,
            static_cast<uint32_t>(inputIndexVector.size()), &inputIndexVector[0],
            static_cast<uint32_t>(outputIndexVector.size()), &outputIndexVector[0]
    );

    int ret = ANeuralNetworksModel_finish(model);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        return ret;
    }

    ret = ANeuralNetworksCompilation_create(model, &compilation);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        return ret;
    }

    ret = ANeuralNetworksCompilation_setPreference(compilation, preference);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        return ret;
    }

    ret = ANeuralNetworksCompilation_finish(compilation);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        return ret;
    }

    return 0;
}



void ModelBuilder::clear() {
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);
    for (auto pointer : bufferPointers) {
        delete[] pointer;
    }
    bufferPointers.clear();
}

ModelBuilder::ModelBuilder() {
}

int ModelBuilder::setInputBuffer(const Model& model, int32_t index, void *buffer, size_t length) {
    for (auto i = 0; i < inputIndexVector.size(); i++) {
        int32_t opIndex = inputIndexVector[i];
        if (opIndex == index) {
            return ANeuralNetworksExecution_setInput(model.execution, i, NULL, buffer, length);
        }
    }

    return WRONG_OPERAND_INDEX;
}


int ModelBuilder::setOutputBuffer(const Model& model, int32_t index, void *buffer, size_t length) {
    for (auto i = 0; i < outputIndexVector.size(); i++) {
        int32_t opIndex = outputIndexVector[i];
        if (opIndex == index) {
            return ANeuralNetworksExecution_setOutput(model.execution, i, NULL, buffer, length);
        }
    }

    return WRONG_OPERAND_INDEX;
}

Model ModelBuilder::prepareForExecution() {
    ANeuralNetworksExecution *execution = nullptr;
    // From document this method only fails when the compilation is invalid, which is already
    // impossible
    ANeuralNetworksExecution_create(compilation, &execution);

    return Model(execution);
}

vector<uint32_t> ModelBuilder::getInputIndexes() {
    return inputIndexVector;
}

vector<uint32_t> ModelBuilder::getOutputIndexes() {
    return outputIndexVector;
}

uint32_t product(const vector<uint32_t> &v) {
    return static_cast<uint32_t> (accumulate(v.begin(), v.end(), 1, multiplies<uint32_t>()));
}

extern "C" int ANeuralNetworksModel_identifyInputsAndOutputs(
        ANeuralNetworksModel* model,
        uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
        const uint32_t* outputs) {

    return ANeuralNetworksModel_setInputsAndOutputs(
            model, inputCount, inputs, outputCount, outputs);
}

