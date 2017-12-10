//
// Created by daquexian on 2017/11/8.
//

#include "ModelBuilder.h"

using namespace std;


ModelBuilder &ModelBuilder::readFromFile(std::string filename) {
    vector<uint32_t> layerToBlob;
    AAsset* asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_UNKNOWN);
    size_t size = static_cast<size_t>(AAsset_getLength(asset));
    char* buffer = new char[size];
    bufferPointers.push_back(static_cast<void *>(buffer));

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
                    // biasIndex = addFloat32NullOperand();
                    vector<uint32_t> dim{numOutput};
                    biasIndex = addFloat32ZeroOperandWithDims(dim);
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
                            if (filterHeight == -1) {
                                filterHeight = inputDim[1];
                            }
                            break;
                        case MF_FILTER_WIDTH:
                            filterWidth = *intPt++;
                            if (filterWidth == -1) {
                                filterWidth = inputDim[2];
                            }
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

                index = addCaffePool(input, strideX, strideY,
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
                    vector<uint32_t> dim{numOutput};
                    biasIndex = addFloat32NullOperandWithDims(dim);
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
            case MF_ADD: {
                uint32_t input1 = layerToBlob[*intPt++];
                uint32_t input2Type = *intPt++;
                switch (input2Type) {
                    case MF_TENSOR_OP: {
                        uint32_t input2 = layerToBlob[*intPt++];
                        index = addAddTensor(input1, input2);
                        break;
                    }
                    case MF_SCALAR_OP: {
                        float scalar = *reinterpret_cast<float *>(intPt++);
                        index = addAddScalar(input1, scalar);
                        break;
                    }
                    case MF_ARRAY_OP: {
                        uint32_t arrayLength = *intPt++;
                        uint32_t input2 = addWeightOrBiasFromBuffer(intPt, vector<uint32_t>{arrayLength});
                        intPt += arrayLength;
                        index = addAddTensor(input1, input2);
                        break;
                    }
                }
                while (*intPt++ != MF_TOP_NAME) ;
                layerToBlob.push_back(index);
                break;
            }
            case MF_MUL: {
                uint32_t input1 = layerToBlob[*intPt++];
                uint32_t input2Type = *intPt++;
                switch (input2Type) {
                    case MF_TENSOR_OP: {
                        uint32_t input2 = layerToBlob[*intPt++];
                        index = addMulTensor(input1, input2);
                        break;
                    }
                    case MF_SCALAR_OP: {
                        float scalar = *reinterpret_cast<float *>(intPt++);
                        index = addMulScalar(input1, scalar);
                        break;
                    }
                    case MF_ARRAY_OP: {
                        uint32_t arrayLength = *intPt++;
                        uint32_t input2 = addWeightOrBiasFromBuffer(intPt, vector<uint32_t>{arrayLength});
                        intPt += arrayLength;
                        index = addMulTensor(input1, input2);
                        break;
                    }
                }
                while (*intPt++ != MF_TOP_NAME) ;
                layerToBlob.push_back(index);
                break;
            }
            case MF_RELU: {
                uint32_t input = layerToBlob[*intPt++];
                index = addReLU(input);
                while (*intPt++ != MF_TOP_NAME) ;
                layerToBlob.push_back(index);
                break;
            }
            case MF_CONCAT: {
                uint32_t inputNum = *intPt++;
                vector<uint32_t> inputs;
                for (uint32_t i = 0; i < inputNum; i++) {
                    inputs.push_back(layerToBlob[*intPt++]);
                }
                uint32_t axis = *intPt++;
                uint32_t activation = *intPt++;
                index = addConcat(inputs, axis, activation);
                layerToBlob.push_back(index);
                while (*intPt++ != MF_TOP_NAME) ;
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
ModelBuilder::addDepthWiseConv(uint32_t input, int32_t strideX, int32_t strideY,
                               int32_t paddingLeft,
                               int32_t paddingRight, int32_t paddingBottom, int32_t paddingTop,
                               int32_t height, int32_t width, int32_t activation,
                               uint32_t outputDepth,
                               int32_t depthMultiplier, uint32_t weightIndex, uint32_t biasIndex) {

    if (input >= nextIndex) return WRONG_INPUT;

    uint32_t strideXOperandIndex = addInt32Operand(strideX);
    uint32_t strideYOperandIndex = addInt32Operand(strideY);
    uint32_t paddingLeftOperandIndex = addInt32Operand(paddingLeft);
    uint32_t paddingRightOperandIndex = addInt32Operand(paddingRight);
    uint32_t paddingTopOperandIndex = addInt32Operand(paddingTop);
    uint32_t paddingBottomOperandIndex = addInt32Operand(paddingBottom);
    uint32_t depthMultiplierOperandIndex = addInt32Operand(depthMultiplier);
    uint32_t activationOperandIndex = addInt32Operand(activation);

    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];

    vector<uint32_t> outputDimen{1,
                                 (inputDimen[1] - height + paddingTop + paddingBottom) / strideY + 1,
                                 (inputDimen[2] - width + paddingLeft + paddingRight) / strideX + 1,
                                 outputDepth};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(outputDimen);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);

    dimensMap[outputOperandIndex] = outputDimen;

    array<uint32_t, 11> inputOperandsArr{{input, weightIndex, biasIndex,
                                                 paddingLeftOperandIndex, paddingRightOperandIndex,
                                                 paddingTopOperandIndex, paddingBottomOperandIndex,
                                                 strideXOperandIndex, strideYOperandIndex,
                                                 depthMultiplierOperandIndex,
                                                 activationOperandIndex}};

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_DEPTHWISE_CONV_2D, 11, &inputOperandsArr[0], 1,
                                      &outputOperandIndex);
    return outputOperandIndex;
}

uint32_t
ModelBuilder::addConv(uint32_t input, int32_t strideX, int32_t strideY, int32_t paddingLeft,
                      int32_t paddingRight, int32_t paddingBottom, int32_t paddingTop,
                      int32_t height, int32_t width, int32_t activation, uint32_t outputDepth,
                      uint32_t weightIndex, uint32_t biasIndex) {
    if (input >= nextIndex) return WRONG_INPUT;

    uint32_t strideXOperandIndex = addInt32Operand(strideX);
    uint32_t strideYOperandIndex = addInt32Operand(strideY);
    uint32_t paddingLeftOperandIndex = addInt32Operand(paddingLeft);
    uint32_t paddingRightOperandIndex = addInt32Operand(paddingRight);
    uint32_t paddingTopOperandIndex = addInt32Operand(paddingTop);
    uint32_t paddingBottomOperandIndex = addInt32Operand(paddingBottom);
    uint32_t activationOperandIndex = addInt32Operand(activation);

    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];

    vector<uint32_t> outputDimen{1,
                                 (inputDimen[1] - height + paddingTop + paddingBottom) / strideY + 1,
                                 (inputDimen[2] - width + paddingLeft + paddingRight) / strideX + 1,
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

uint32_t ModelBuilder::addCaffePool(uint32_t input, int32_t strideX, int32_t strideY,
                                    int32_t paddingLeft, int32_t paddingRight,
                                    int32_t paddingTop, int32_t paddingBottom,
                                    int32_t height, int32_t width,
                                    int32_t activation, uint32_t poolingType) {

    if (input >= nextIndex) return WRONG_INPUT;

    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];


    // https://github.com/BVLC/caffe/pull/473#issuecomment-45386156
    unsigned int extraY = (inputDimen[1] - height + paddingTop + paddingBottom) % strideY;
    if (extraY != 0) {
        paddingBottom += strideY - extraY;
    }
    unsigned int extraX = (inputDimen[2] - width + paddingLeft + paddingRight) % strideX;
    if (extraX != 0) {
        paddingRight += strideX - extraX;
    }

    uint32_t widthOperandIndex = addInt32Operand(width);
    uint32_t heightOperandIndex = addInt32Operand(height);
    uint32_t strideXOperandIndex = addInt32Operand(strideX);
    uint32_t strideYOperandIndex = addInt32Operand(strideY);
    uint32_t paddingLeftOperandIndex = addInt32Operand(paddingLeft);
    uint32_t paddingRightOperandIndex = addInt32Operand(paddingRight);
    uint32_t paddingTopOperandIndex = addInt32Operand(paddingTop);
    uint32_t paddingBottomOperandIndex = addInt32Operand(paddingBottom);
    uint32_t activationOperandIndex = addInt32Operand(activation);

    vector<uint32_t> outputDimen{1,
                                 (inputDimen[1] - height + paddingTop + paddingBottom) / strideY + 1,
                                 (inputDimen[2] - width + paddingLeft + paddingRight) / strideX + 1,
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
    vector<uint32_t> dimen = dimensMap[input];

    uint32_t betaIndex = addFloat32Operand(beta);

    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t outputOperandIndex = addNewOperand(&type);

    dimensMap[outputOperandIndex] = dimen;

    array<uint32_t, 2> inputOperandsArr{{input, betaIndex}};

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_SOFTMAX, 2, &inputOperandsArr[0],
                                      1, &outputOperandIndex);

    return outputOperandIndex;
}

uint32_t ModelBuilder::addReLU(uint32_t input) {
    vector<uint32_t> dimen = dimensMap[input];

    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t outputOperandIndex = addNewOperand(&type);

    dimensMap[outputOperandIndex] = dimen;

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_RELU,
                                      1, &input, 1, &outputOperandIndex);
    return outputOperandIndex;
}

uint32_t
ModelBuilder::addConcat(const vector<uint32_t> &inputs, uint32_t axis, uint32_t activation) {
    vector<vector<uint32_t>> dimens;
    for (const auto &input : inputs) {
        vector<uint32_t> &dimen = dimensMap[input];
        if (dimens.size() > 0) {
            for (size_t i = 0; i < dimens[0].size(); i++) {
                if (i == axis) continue;
                if (dimen[i] != dimens[0][i]) {
                    throw "Wrong input for concat";
                }
            }
        }
        dimens.push_back(dimensMap[input]);
    }

    vector<uint32_t> outputDimen = dimens[0];
    for (size_t i = 1; i < dimens.size(); i++) {
        outputDimen[axis] += dimens[i][axis];
    }

    uint32_t axisOperandIndex = addInt32Operand(axis);
    uint32_t activationOperandIndex = addInt32Operand(activation);

    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(outputDimen);
    uint32_t outputOperandIndex = addNewOperand(&type);

    dimensMap[outputOperandIndex] = outputDimen;

    vector<uint32_t> operationInputs = inputs;
    operationInputs.push_back(axisOperandIndex);
    // This undocumented input are is in MR1, it seems not needed anymore in MR2
    // operationInputs.push_back(activationOperandIndex);

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONCATENATION,
                                      operationInputs.size(), &operationInputs[0], 1, &outputOperandIndex);
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
    bufferPointers.push_back(static_cast<void *>(buffer));
    AAsset_read(asset, buffer, static_cast<size_t>(size));
    ANeuralNetworksModel_setOperandValue(model, index, buffer, size);
    return buffer;
}

uint32_t ModelBuilder::addInt32Operand(int32_t value) {
    if (int32OperandMap.find(value) == int32OperandMap.end()) {
        ANeuralNetworksOperandType type = getInt32OperandType();
        uint32_t index = addNewOperand(&type);
        ANeuralNetworksModel_setOperandValue(model, index, &value, sizeof(value));
        int32OperandMap[value] = index;
    }
    return int32OperandMap[value];
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
ModelBuilder::addFC(uint32_t input, uint32_t outputNum, int32_t activation, uint32_t weightIndex,
                    uint32_t biasIndex) {

    uint32_t activationOperandIndex = addInt32Operand(activation);

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

uint32_t ModelBuilder::addAddScalar(uint32_t input, float scalar) {
    uint32_t scalarIndex = addFloat32Operand(scalar);
    array<uint32_t, 3> inputOperands{{input, scalarIndex, addInt32Operand(
            ModelBuilder::ACTIVATION_NONE)}};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(dimensMap[input]);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);
    dimensMap[outputOperandIndex] = dimensMap[input];

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 3, &inputOperands[0], 1, &outputOperandIndex);
    return outputOperandIndex;
}

uint32_t ModelBuilder::addAddTensor(uint32_t input1, uint32_t input2) {
    array<uint32_t, 3> inputOperands{{input1, input2, addInt32Operand(ModelBuilder::ACTIVATION_NONE)}};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(dimensMap[input1]);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);
    dimensMap[outputOperandIndex] = dimensMap[input1];

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 3, &inputOperands[0], 1, &outputOperandIndex);
    return outputOperandIndex;
}

uint32_t ModelBuilder::addMulScalar(uint32_t input, float scalar) {
    uint32_t scalarIndex = addFloat32Operand(scalar);
    array<uint32_t, 3> inputOperands{{input, scalarIndex, addInt32Operand(
            ModelBuilder::ACTIVATION_NONE)}};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(dimensMap[input]);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);
    dimensMap[outputOperandIndex] = dimensMap[input];

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MUL, 3, &inputOperands[0], 1, &outputOperandIndex);
    return outputOperandIndex;
}

uint32_t ModelBuilder::addMulTensor(uint32_t input1, uint32_t input2) {
    array<uint32_t, 3> inputOperands{{input1, input2, addInt32Operand(ModelBuilder::ACTIVATION_NONE)}};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(dimensMap[input1]);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);
    dimensMap[outputOperandIndex] = dimensMap[input1];

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MUL, 3, &inputOperands[0], 1, &outputOperandIndex);
    return outputOperandIndex;
}

uint32_t ModelBuilder::addFloat32NullOperandWithDims(std::vector<uint32_t> &dims) {
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dims);
    uint32_t index = addNewOperand(&type);
    ANeuralNetworksModel_setOperandValue(model, index, nullptr, 0);
    return index;
}

uint32_t ModelBuilder::addFloat32ZeroOperandWithDims(std::vector<uint32_t> &dims) {
    float *zeros = new float[product(dims)];
    bufferPointers.push_back(static_cast<void *>(zeros));
    for (size_t i = 0; i < product(dims); i++) {
        zeros[i] = 0;
    }
    return addWeightOrBiasFromBuffer(zeros, dims);
}

vector<uint32_t> ModelBuilder::getBlobDim(std::string blobName) {
    return dimensMap[getBlobIndex(blobName)];
}

std::vector<uint32_t> ModelBuilder::getBlobDim(uint32_t index) {
    return dimensMap[index];
}

uint32_t product(const vector<uint32_t> &v) {
    return static_cast<uint32_t> (accumulate(v.begin(), v.end(), 1, multiplies<uint32_t>()));
}
