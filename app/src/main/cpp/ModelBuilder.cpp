//
// Created by daquexian on 2017/11/8.
//

#include <sstream>
#include "ModelBuilder.h"

using namespace std;

#define  LOG_TAG    "NNAPI Demo"

#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG,__VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)


ModelBuilder &ModelBuilder::readFromFile(std::string filename) {
    vector<uint32_t> layerToBlob;
    AAsset* asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_UNKNOWN);
    size_t size = static_cast<size_t>(AAsset_getLength(asset));
    char* buffer = new char[size];
    uint32_t *intPt = reinterpret_cast<uint32_t *>(buffer);
    AAsset_read(asset, buffer, static_cast<size_t>(size));


    uint32_t layerType;
    while ((layerType = *intPt++) != MF_LAYER_END) {
        uint32_t index;
        string topName;
        switch (layerType) {
            case MF_INPUT: {
                intPt++;    // skip N
                uint32_t depth = *intPt++;
                uint32_t height = *intPt++;
                uint32_t width = *intPt++;

                layerToBlob.push_back(addInput(height, width, depth));

                while (*intPt++ != MF_TOP_NAME) ;

                break;
            }
            case MF_CONV: {
                uint32_t input = layerToBlob[*intPt++];
                vector<uint32_t> inputDim = dimensMap[input];
                uint32_t paddingLeft = 0, paddingRight = 0, paddingTop = 0, paddingBottom = 0,
                        strideX = 1, strideY = 1, filterHeight, filterWidth, numOutput,
                        activation = ACTIVATION_NONE;
                uint32_t weightIndex, biasIndex = UINT32_MAX;
                uint32_t paramType;
                while ((paramType = *intPt++) != MF_TOP_NAME) {
                    switch (paramType) {
                        case MF_PADDING_LEFT:
                            paddingLeft = *intPt++;
                            break;
                        case MF_PADDING_RIGHT:
                            paddingRight = *intPt++;
                            break;
                        case MF_PADDING_TOP:
                            paddingTop = *intPt++;
                            break;
                        case MF_PADDING_BOTTOM:
                            paddingBottom = *intPt++;
                            break;
                        case MF_STRIDE_X:
                            strideX = *intPt++;
                            break;
                        case MF_STRIDE_Y:
                            strideY = *intPt++;
                            break;
                        case MF_FILTER_HEIGHT:
                            filterHeight = *intPt++;
                            break;
                        case MF_FILTER_WIDTH:
                            filterWidth = *intPt++;
                            break;
                        case MF_NUM_OUTPUT:
                            numOutput = *intPt++;
                            break;
                        case MF_WEIGHT: {
                            vector<uint32_t> weightDim{numOutput, filterHeight, filterWidth,
                                                       inputDim[3]};
                            weightIndex = addWeightOrBiasFromBuffer(intPt, weightDim);
                            intPt += product(weightDim);

                            break;
                        }
                        case MF_BIAS: {
                            biasIndex = addWeightOrBiasFromBuffer(intPt,
                                                                  vector<uint32_t>{numOutput});
                            intPt += numOutput;
                            break;
                        }
                        case MF_ACTIVATION: {
                            uint32_t mfActType = *intPt++;
                            if (mfActType == MF_ACTIVATION_NONE) {
                                activation = ACTIVATION_NONE;
                            } else if (mfActType == MF_ACTIVATION_RELU) {
                                activation = ACTIVATION_RELU;
                            }
                            break;
                        }
                    }
                }

                if (biasIndex == UINT32_MAX) {
                    biasIndex = addFloat32NullOperand();
                }

                index = addConv(input, strideX, strideY, paddingLeft, paddingRight,
                                paddingBottom, paddingTop,
                                filterHeight, filterWidth,
                                activation, numOutput, weightIndex,
                                biasIndex);
                layerToBlob.push_back(index);
                break;
            }
            case MF_MAX_POOL:
            case MF_AVE_POOL: {
                uint32_t input = layerToBlob[*intPt++];
                uint32_t paddingLeft = 0, paddingRight = 0, paddingTop = 0, paddingBottom = 0,
                        strideX = 1, strideY = 1, filterHeight, filterWidth,
                        activation = ACTIVATION_NONE, poolingType;
                vector<uint32_t> inputDim = dimensMap[input];
                uint32_t paramType;
                while ((paramType = *intPt++) != MF_TOP_NAME) {
                    switch (paramType) {
                        case MF_PADDING_LEFT:
                            paddingLeft = *intPt++;
                            break;
                        case MF_PADDING_RIGHT:
                            paddingRight = *intPt++;
                            break;
                        case MF_PADDING_TOP:
                            paddingTop = *intPt++;
                            break;
                        case MF_PADDING_BOTTOM:
                            paddingBottom = *intPt++;
                            break;
                        case MF_STRIDE_X:
                            strideX = *intPt++;
                            break;
                        case MF_STRIDE_Y:
                            strideY = *intPt++;
                            break;
                        case MF_FILTER_HEIGHT:
                            filterHeight = *intPt++;
                            break;
                        case MF_FILTER_WIDTH:
                            filterWidth = *intPt++;
                            break;
                        case MF_ACTIVATION:
                            uint32_t mfActType = *intPt++;
                            if (mfActType == MF_ACTIVATION_NONE) {
                                activation = ACTIVATION_NONE;
                            } else if (mfActType == MF_ACTIVATION_RELU) {
                                activation = ACTIVATION_RELU;
                            }
                            break;
                    }
                }

                poolingType = layerType == MF_MAX_POOL ? MAX_POOL : AVE_POOL;

                index = addPool(input, strideX, strideY,
                                paddingLeft, paddingRight, paddingTop, paddingBottom,
                                filterHeight, filterWidth, activation, poolingType);

                layerToBlob.push_back(index);
                break;
            }

            case MF_FC: {
                uint32_t input = layerToBlob[*intPt++];
                vector<uint32_t> inputDim = dimensMap[input];
                uint32_t numOutput, activation = ACTIVATION_NONE;
                uint32_t weightIndex, biasIndex;
                uint32_t paramType;
                while ((paramType = *intPt++) != MF_TOP_NAME) {
                    switch (paramType) {
                        case MF_NUM_OUTPUT: {
                            numOutput = *intPt++;

                            break;
                        }
                        case MF_WEIGHT: {
                            vector<uint32_t> weightDim{numOutput, product(inputDim)};
                            weightIndex = addWeightOrBiasFromBuffer(intPt, weightDim);
                            intPt += product(weightDim);

                            break;
                        }
                        case MF_BIAS: {
                            biasIndex = addWeightOrBiasFromBuffer(intPt,
                                                                  vector<uint32_t>{numOutput});
                            intPt += numOutput;
                            break;
                        }
                        case MF_ACTIVATION: {
                            uint32_t mfActType = *intPt++;
                            if (mfActType == MF_ACTIVATION_NONE) {
                                activation = ACTIVATION_NONE;
                            } else if (mfActType == MF_ACTIVATION_RELU) {
                                activation = ACTIVATION_RELU;
                            }
                            break;
                        }
                    }
                }

                if (biasIndex == UINT32_MAX) {
                    biasIndex = addFloat32NullOperand();
                }

                index = addFC(input, numOutput, activation, weightIndex, biasIndex);

                layerToBlob.push_back(index);
                break;
            }
            case MF_SOFTMAX: {
                uint32_t input = layerToBlob[*intPt++];
                float beta = 1.f;
                uint32_t paramType;
                while ((paramType = *intPt++) != MF_TOP_NAME) {
                    switch (paramType) {
                        case MF_BETA: {
                            beta = *reinterpret_cast<float *>(intPt++);
                            break;
                        }
                    }
                }
                index = addSoftMax(input, beta);
                layerToBlob.push_back(index);
                break;
            }
            default:
                throw "Unsupport layer";
        }
        stringstream ss;
        char c;
        while ((c = static_cast<char>(*intPt++)) != MF_STRING_END) {
            ss << c;
        }

        topName = ss.str();
        blobNameToIndex[topName] = index;
        while (*intPt++ != MF_PARAM_END) ;
    }


    return *this;
}

uint32_t ModelBuilder::addInput(uint32_t height, uint32_t width, uint32_t depth) {
    vector<uint32_t> dimen{1, width, height, depth};
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);

    dimensMap[index] = dimen;
    inputIndexVector.push_back(index);
    return index;
}

uint32_t
ModelBuilder::addConv(uint32_t input, uint32_t strideX, uint32_t strideY, uint32_t paddingLeft,
                      uint32_t paddingRight, uint32_t paddingBottom, uint32_t paddingTop,
                      uint32_t height, uint32_t width, uint32_t activation, uint32_t outputDepth,
                      uint32_t weightIndex, uint32_t biasIndex) {
    if (input >= nextIndex) return WRONG_INPUT;

    uint32_t strideXOperandIndex = addUInt32Operand(strideX);
    uint32_t strideYOperandIndex = addUInt32Operand(strideY);
    uint32_t paddingLeftOperandIndex = addUInt32Operand(paddingLeft);
    uint32_t paddingRightOperandIndex = addUInt32Operand(paddingRight);
    uint32_t paddingTopOperandIndex = addUInt32Operand(paddingTop);
    uint32_t paddingBottomOperandIndex = addUInt32Operand(paddingBottom);
    uint32_t activationOperandIndex = addUInt32Operand(activation);

    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];

    vector<uint32_t> outputDimen{1,
                                 (inputDimen[1] - height + 2 * paddingTop) / strideY + 1,
                                 (inputDimen[2] - width + 2 * paddingLeft) / strideX + 1,
                                 outputDepth};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(outputDimen);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);

    dimensMap[outputOperandIndex] = outputDimen;

    array<uint32_t, 10> inputOperandsArr{{input, weightIndex, biasIndex,
                                                 paddingLeftOperandIndex, paddingRightOperandIndex,
                                                 paddingTopOperandIndex, paddingBottomOperandIndex,
                                                 strideXOperandIndex, strideYOperandIndex,
                                                 activationOperandIndex}};

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 10, &inputOperandsArr[0], 1,
                                      &outputOperandIndex);
    return outputOperandIndex;
}

uint32_t
ModelBuilder::addConv(std::string name, uint32_t input, uint32_t strideX, uint32_t strideY,
                      uint32_t paddingW, uint32_t paddingH, uint32_t height, uint32_t width,
                      uint32_t activation, uint32_t outputDepth) {
    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];

    uint32_t weightOperandIndex = addConvWeight(name, height, width, inputDimen[3], outputDepth);
    uint32_t biasOperandIndex = addBias(name, outputDepth);

    return addConv(input, strideX, strideY, paddingW, paddingW, paddingH, paddingH, height, width,
                   activation, outputDepth, weightOperandIndex, biasOperandIndex);

}

uint32_t ModelBuilder::addFC(std::string name, uint32_t input, uint32_t outputNum,
                             uint32_t activation) {
    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];

    uint32_t weightOperandIndex = addFcWeight(name, product(inputDimen),
                                              outputNum);
    uint32_t biasOperandIndex = addBias(name, outputNum);

    return addFC(input, outputNum, activation, weightOperandIndex, biasOperandIndex);
}

uint32_t ModelBuilder::addPool(uint32_t input, uint32_t strideX, uint32_t strideY,
                               uint32_t paddingLeft, uint32_t paddingRight,
                               uint32_t paddingTop, uint32_t paddingBottom,
                               uint32_t height, uint32_t width,
                               uint32_t activation, uint32_t poolingType) {

    if (input >= nextIndex) return WRONG_INPUT;

    uint32_t widthOperandIndex = addUInt32Operand(width);
    uint32_t heightOperandIndex = addUInt32Operand(height);
    uint32_t strideXOperandIndex = addUInt32Operand(strideX);
    uint32_t strideYOperandIndex = addUInt32Operand(strideY);
    uint32_t paddingLeftOperandIndex = addUInt32Operand(paddingLeft);
    uint32_t paddingRightOperandIndex = addUInt32Operand(paddingRight);
    uint32_t paddingTopOperandIndex = addUInt32Operand(paddingTop);
    uint32_t paddingBottomOperandIndex = addUInt32Operand(paddingBottom);
    uint32_t activationOperandIndex = addUInt32Operand(activation);

    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];
    vector<uint32_t> outputDimen{1,
                                 (inputDimen[1] - height + 2 * paddingTop) / strideY + 1,
                                 (inputDimen[2] - width + 2 * paddingLeft) / strideX + 1,
                                 inputDimen[3]};

    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(outputDimen);
    uint32_t outputOperandIndex = addNewOperand(&type);

    dimensMap[outputOperandIndex] = outputDimen;

    array<uint32_t, 10> inputOperandsArr{{input, paddingLeftOperandIndex, paddingRightOperandIndex,
                                                 paddingTopOperandIndex, paddingBottomOperandIndex,
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

uint32_t ModelBuilder::addSoftMax(uint32_t input, float beta) {
    vector<uint32_t> inputDimen = dimensMap[input];
    vector<uint32_t> outputDimen = inputDimen;

    uint32_t betaIndex = addFloat32Operand(beta);

    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(outputDimen);
    uint32_t outputOperandIndex = addNewOperand(&type);

    dimensMap[outputOperandIndex] = outputDimen;

    array<uint32_t, 2> inputOperandsArr{{input, betaIndex}};

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_SOFTMAX, 2, &inputOperandsArr[0],
                                      // 1, &input);
                                      1, &outputOperandIndex);

    // return input;
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
        uint32_t index = addNewOperand(&type);
        ANeuralNetworksModel_setOperandValue(model, index, &value, sizeof(value));
        uint32OperandMap[value] = index;
    }
    return uint32OperandMap[value];
}

uint32_t ModelBuilder::addFloat32Operand(float value) {
    if (float32OperandMap.find(value) == float32OperandMap.end()) {
        ANeuralNetworksOperandType type = getFloat32OperandType();
        uint32_t index = addNewOperand(&type);
        ANeuralNetworksModel_setOperandValue(model, index, &value, sizeof(value));
        float32OperandMap[value] = index;
    }
    return float32OperandMap[value];

}

uint32_t ModelBuilder::addInt32NullOperand() {
    if (missingInt32OperandIndex == UINT32_MAX) {
        ANeuralNetworksOperandType type = getInt32OperandType();
        missingInt32OperandIndex = addNewOperand(&type);
        ANeuralNetworksModel_setOperandValue(model, missingInt32OperandIndex, nullptr, 0);
    }
    return missingInt32OperandIndex;
}

uint32_t ModelBuilder::addFloat32NullOperand(){
    if (missingFloat32OperandIndex == UINT32_MAX) {
        ANeuralNetworksOperandType type = getFloat32OperandType();
        missingFloat32OperandIndex = addNewOperand(&type);
        ANeuralNetworksModel_setOperandValue(model, missingFloat32OperandIndex, nullptr, 0);
    }
    return missingFloat32OperandIndex;
}

uint32_t ModelBuilder::addNewOperand(ANeuralNetworksOperandType *type) {
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
    uint32_t biasIndex = addNewOperand(&biasType);
    bufferPointers.push_back(setOperandValueFromAssets(
            model, mgr, biasIndex, "weights_and_biases/" + name + "_biases"));
    return biasIndex;
}

uint32_t ModelBuilder::addWeight(std::string name, std::vector<uint32_t> dimen) {
    ANeuralNetworksOperandType weightType = getFloat32OperandTypeWithDims(dimen);
    uint32_t weightIndex = addNewOperand(&weightType);
    bufferPointers.push_back(setOperandValueFromAssets(
            model, mgr, weightIndex, "weights_and_biases/" + name + "_weights"));

    return weightIndex;
}

uint32_t ModelBuilder::addWeightOrBiasFromBuffer(const void *buffer, std::vector<uint32_t> dimen) {
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);
    ANeuralNetworksModel_setOperandValue(model, index, buffer, product(dimen) * sizeof(uint32_t));
    return index;
}

int ModelBuilder::init(AAssetManager *mgr) {
    this->mgr = mgr;
    return ANeuralNetworksModel_create(&model);
}

void ModelBuilder::addIndexIntoOutput(uint32_t index) {
    outputIndexVector.push_back(index);
}

int ModelBuilder::compile(uint32_t preference) {
    int ret;
    if ((ret = ANeuralNetworksModel_identifyInputsAndOutputs(
            model,
            static_cast<uint32_t>(inputIndexVector.size()), &inputIndexVector[0],
            static_cast<uint32_t>(outputIndexVector.size()), &outputIndexVector[0] )) != ANEURALNETWORKS_NO_ERROR) {

        return ret;
    }

    ret = ANeuralNetworksModel_finish(model);
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
    LOGD("%d", model.execution == nullptr);
    LOGD("%d", buffer == nullptr);
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

void ModelBuilder::prepareForExecution(Model &model) {
    ANeuralNetworksExecution *execution = nullptr;
    // From document this method only fails when the compilation is invalid, which is already
    // impossible
    ANeuralNetworksExecution_create(compilation, &execution);

    model.execution = execution;
}

vector<uint32_t> ModelBuilder::getInputIndexes() {
    return inputIndexVector;
}

vector<uint32_t> ModelBuilder::getOutputIndexes() {
    return outputIndexVector;
}

uint32_t
ModelBuilder::addFC(uint32_t input, uint32_t outputNum, uint32_t activation, uint32_t weightIndex,
                    uint32_t biasIndex) {

    uint32_t activationOperandIndex = addUInt32Operand(activation);

    vector<uint32_t> outputDimen{1, outputNum};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(outputDimen);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);

    dimensMap[outputOperandIndex] = outputDimen;

    array<uint32_t, 10> inputOperandsArr{{input, weightIndex, biasIndex,
                                                 activationOperandIndex}};

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED,
                                      4, &inputOperandsArr[0], 1, &outputOperandIndex);

    return outputOperandIndex;
}

uint32_t ModelBuilder::getBlobIndex(std::string blobName) {
    return blobNameToIndex[blobName];
}

uint32_t ModelBuilder::addMulScalarInplace(uint32_t input, float scalar) {
    uint32_t scalarIndex = addFloat32Operand(scalar);
    array<uint32_t, 3> inputOperands{{input, scalarIndex, addUInt32Operand(ModelBuilder::ACTIVATION_NONE)}};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MUL, 3, &inputOperands[0], 1, &input);
    return input;
}

uint32_t ModelBuilder::addAddScalarInplace(uint32_t input, float scalar) {
    uint32_t scalarIndex = addFloat32Operand(scalar);
    array<uint32_t, 3> inputOperands{{input, scalarIndex, addUInt32Operand(ModelBuilder::ACTIVATION_NONE)}};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 3, &inputOperands[0], 1, &input);
    return input;
}

uint32_t ModelBuilder::addAddScalar(uint32_t input, float scalar) {
    uint32_t scalarIndex = addFloat32Operand(scalar);
    array<uint32_t, 3> inputOperands{{input, scalarIndex, addUInt32Operand(ModelBuilder::ACTIVATION_NONE)}};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(dimensMap[input]);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);
    dimensMap[outputOperandIndex] = dimensMap[input];

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 3, &inputOperands[0], 1, &outputOperandIndex);
    return outputOperandIndex;
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

