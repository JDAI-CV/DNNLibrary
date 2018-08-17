//
// Created by daquexian on 2017/11/8.
//
#include "ModelBuilder.h"

#include <array>
#include <cmath>
#include <ctime>
#include <tuple>
#include <numeric>
#include <sstream>
#include <sys/mman.h>
#include <fstream>
#include <iostream>

#include "android_log_helper.h"
#include <operand_helper.h>
#include <ModelBuilder.h>


using std::vector; using std::ifstream; using std::streamsize; using std::string; using std::ios;
using std::stringstream; using std::array;

#define ADD_OPERATION(operation, input_indexes, output_indexes) \
ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_##operation, static_cast<uint32_t>(input_indexes.size()),  \
    &input_indexes[0], static_cast<uint32_t>(output_indexes.size()), &output_indexes[0]);

// Simplest model for test
ModelBuilder &ModelBuilder::simplestModel() {
    auto input = addInput(4, 3, 2);
    auto add = addAddScalar(input, 1.5f);
    addIndexIntoOutput(add);
    return *this;
}


/**
 * It is designed to read a regular file. For reading file in assets folder of Android app,
 * read the content into a char array and call readFromBuffer
 *
 * @param filename , like "/data/local/tmp/squeezenet.daq"
 * @return ModelBuilder itself
 */
ModelBuilder &ModelBuilder::readFromFile(const string &filename) {
    return *this;
    /*
    std::ifstream ifs(filename, ios::binary | ios::ate);
    streamsize len = ifs.tellg();
    ifs.seekg(0, ios::beg);
    auto *buffer = new char[len];
    // read whole content of a file into buffer
    if (ifs.read(buffer, len)) {
        registerBufferPointer(buffer);
        OnnxReader onnx_reader;
        onnx_reader.ReadFile(filename, *this);
        return *this;
    } else {
        throw string("Read file error");
    }
    */
}

/*
ModelBuilder &ModelBuilder::readFromBuffer(const char* buffer) {
    vector<uint32_t> layerToBlob;
    const auto *intPt = reinterpret_cast<const uint32_t *>(buffer);
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
                            weightIndex = addTensorFromBuffer(reinterpret_cast<const float*>(intPt), weightDim);
                            intPt += product(weightDim);

                            break;
                        }
                        case MF_BIAS: {
                            biasIndex = addTensorFromBuffer(reinterpret_cast<const float*>(intPt),
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

                index = addConv(input, strideX, strideY, paddingLeft, paddingRight, paddingBottom, paddingTop,
                                activation, weightIndex, biasIndex);
                layerToBlob.push_back(index);
                break;
            }
            case MF_DEPTH_CONV: {
                uint32_t input = layerToBlob[*intPt++];
                vector<uint32_t> inputDim = dimensMap[input];
                uint32_t paddingLeft = 0, paddingRight = 0, paddingTop = 0, paddingBottom = 0,
                        strideX = 1, strideY = 1, filterHeight, filterWidth, numOutput, depthMultiplier,
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
                        case MF_GROUP:
                            depthMultiplier = numOutput / (*intPt++);
                            break;
                        case MF_WEIGHT: {
                            vector<uint32_t> weightDim{1, filterHeight, filterWidth, numOutput};
                            weightIndex = addTensorFromBuffer(intPt, weightDim);
                            intPt += product(weightDim);
                            break;
                        }
                        case MF_BIAS: {
                            biasIndex = addTensorFromBuffer(intPt, vector<uint32_t>{numOutput});
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

                index = addDepthWiseConv(input, strideX, strideY, paddingLeft, paddingRight, paddingBottom, paddingTop,
                                         activation, depthMultiplier, weightIndex, biasIndex);
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
                            weightIndex = addTensorFromBuffer(intPt, weightDim);
                            intPt += product(weightDim);

                            break;
                        }
                        case MF_BIAS: {
                            biasIndex = addTensorFromBuffer(intPt,
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

                index = addFC(input, activation, weightIndex, biasIndex);

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
                            beta = *reinterpret_cast<const float *>(intPt++);
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
                        float scalar = *reinterpret_cast<const float *>(intPt++);
                        index = addAddScalar(input1, scalar);
                        break;
                    }
                    case MF_ARRAY_OP: {
                        uint32_t arrayLength = *intPt++;
                        uint32_t input2 = addTensorFromBuffer(intPt, vector<uint32_t>{arrayLength});
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
                        float scalar = *reinterpret_cast<const float *>(intPt++);
                        index = addMulScalar(input1, scalar);
                        break;
                    }
                    case MF_ARRAY_OP: {
                        uint32_t arrayLength = *intPt++;
                        uint32_t input2 = addTensorFromBuffer(intPt, vector<uint32_t>{arrayLength});
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
                index = addConcat(inputs, axis);
                layerToBlob.push_back(index);
                while (*intPt++ != MF_TOP_NAME) ;
                break;
            }
#if __ANDROID_API__ >= __ANDROID_API_P__
            case MF_STRIDED_SLICE: {
                uint32_t input = layerToBlob[*intPt++];
                vector<uint32_t> inputDim = dimensMap[input];
                vector<int32_t> starts, ends, strides;
                for (int i = 0 ; i < inputDim.size(); i++) {
                    starts.emplace_back(*intPt++);
                }
                for (int i = 0 ; i < inputDim.size(); i++) {
                    ends.emplace_back(*intPt++);
                }
                for (int i = 0 ; i < inputDim.size(); i++) {
                    strides.emplace_back(*intPt++);
                }
                uint32_t beginMask = *intPt++;
                uint32_t endMask = *intPt++;
                uint32_t shrinkMask = *intPt++;
                index = addStridedSlice(input, starts, ends, strides, beginMask, endMask, shrinkMask);

                while (*intPt++ != MF_TOP_NAME);
                layerToBlob.push_back(index);
                break;
            }
#endif
            case MF_LRN: {
                uint32_t input = layerToBlob[*intPt++];
                uint32_t local_size = 5;
                float alpha = 0.0001, beta = 0.75, bias = 1.0;
                uint32_t paramType;
                while ((paramType = *intPt++) != MF_TOP_NAME) {
                    switch (paramType) {
                        case MF_LRN_ALPHA :
                            alpha = *reinterpret_cast<const float *>(intPt++);
                            break;
                        case MF_LRN_BETA :
                            beta = *reinterpret_cast<const float *>(intPt++);
                            break;
                        case MF_LOCAL_SIZE :
                            local_size = *intPt++;
                            break;
                    }
                }
                index = addLRN(input, local_size, bias, alpha, beta);
                layerToBlob.push_back(index);
                break;
            }
            default:
                throw string("Unsupport layer");
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
 */

ModelBuilder::Index ModelBuilder::addInput(uint32_t height, uint32_t width, uint32_t depth) {
    vector<uint32_t> dimen{1, width, height, depth};
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);

    dimensMap[index] = dimen;
    inputIndexVector.push_back(index);
    return index;
}

ModelBuilder::Index ModelBuilder::addDepthWiseConv(Index input, int32_t strideX, int32_t strideY, int32_t paddingLeft,
                                                   int32_t paddingRight,
                                                   int32_t paddingBottom, int32_t paddingTop, int32_t activation,
                                                   int32_t depthMultiplier,
                                                   uint32_t weightIndex, std::optional<uint32_t> biasIndex) {

    if (input >= nextIndex) return WRONG_INPUT;

    Shape weightDimen = dimensMap[weightIndex];     // num_output, height, width, num_input
    // NHWC
    Shape inputDimen = dimensMap[input];
    Shape outputDimen{1,
                      (inputDimen[1] - weightDimen[1] + paddingTop + paddingBottom) / strideY + 1,
                      (inputDimen[2] - weightDimen[2] + paddingLeft + paddingRight) / strideX + 1,
                      weightDimen[0]};
    uint32_t biasIndexValue;
    if (!biasIndex.has_value()) {
        Shape bias_dims = Shape{weightDimen[0]};
        biasIndexValue = addFloat32ZeroOperandWithDims(bias_dims);
    } else {
        biasIndexValue = biasIndex.value();
    }
    IndexSeq input_indexes{input, weightIndex, biasIndexValue};
    addOperands(input_indexes, paddingLeft, paddingRight, paddingTop, paddingBottom,
                strideX, strideY, depthMultiplier, activation);
    auto output_index = addOperation(ANEURALNETWORKS_DEPTHWISE_CONV_2D, input_indexes, outputDimen)[0];
    return output_index;
}

ModelBuilder::Index
ModelBuilder::addConv(Index input, int32_t strideX, int32_t strideY, int32_t paddingLeft, int32_t paddingRight,
                      int32_t paddingBottom,
                      int32_t paddingTop, int32_t activation, uint32_t weightIndex, std::optional<uint32_t> biasIndex) {
    if (input >= nextIndex) return WRONG_INPUT;

    Shape weightDimen = dimensMap[weightIndex];     // num_output, height, width, num_input
    // NHWC
    vector<uint32_t> inputDimen = dimensMap[input];
    std::cout << inputDimen[1] << ", " << weightDimen[1] << ", " << paddingTop << ", " << paddingBottom << ", " << strideY << std::endl;
    vector<uint32_t> outputDimen{1,
                                 (inputDimen[1] - weightDimen[1] + paddingTop + paddingBottom) / strideY + 1,
                                 (inputDimen[2] - weightDimen[2] + paddingLeft + paddingRight) / strideX + 1,
                                 weightDimen[0]};
    uint32_t biasIndexValue;
    if (!biasIndex.has_value()) {
        Shape bias_dims = Shape{weightDimen[0]};
        biasIndexValue = addFloat32ZeroOperandWithDims(bias_dims);
    } else {
        biasIndexValue = biasIndex.value();
    }
    IndexSeq input_indexes{input, weightIndex, biasIndexValue};
    addOperands(input_indexes, paddingLeft, paddingRight, paddingTop, paddingBottom, strideX, strideY, activation);
    auto output_indexes = addOperation(ANEURALNETWORKS_CONV_2D, input_indexes, outputDimen);
    return output_indexes[0];
}

#if __ANDROID_API__ >= __ANDROID_API_P__

ModelBuilder::Index
ModelBuilder::addStridedSlice(Index input, const vector<int32_t> &starts, const vector<int32_t> &ends,
                              const vector<int32_t> &strides, int32_t beginMask, int32_t endMask,
                              int32_t shrinkAxisMask) {

    if (input >= nextIndex) return WRONG_INPUT;

    uint32_t startsIndex = addTensorFromBuffer(&starts[0], vector<uint32_t>{static_cast<uint32_t>(starts.size())});
    uint32_t endsIndex = addTensorFromBuffer(&ends[0], vector<uint32_t>{static_cast<uint32_t>(ends.size())});
    uint32_t stridesIndex = addTensorFromBuffer(&strides[0],
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

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_STRIDED_SLICE, 7, &inputOperandsArr[0], 1,
                                      &outputOperandIndex);
    return outputOperandIndex;
}

#endif

ModelBuilder::Index ModelBuilder::addPool(Index input, int32_t strideX, int32_t strideY,
                                          int32_t paddingLeft, int32_t paddingRight,
                                          int32_t paddingTop, int32_t paddingBottom,
                                          int32_t height, int32_t width,
                                          int32_t activation, uint32_t poolingType) {

    if (input >= nextIndex) return WRONG_INPUT;

    // NHWC
    auto inputDimen = dimensMap[input];

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
    return output_index;
}

ModelBuilder::Index ModelBuilder::addCaffePool(Index input, int32_t strideX, int32_t strideY,
                                               int32_t paddingLeft, int32_t paddingRight,
                                               int32_t paddingTop, int32_t paddingBottom,
                                               int32_t height, int32_t width,
                                               int32_t activation, uint32_t poolingType) {

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
    return output_index;
}

ModelBuilder::Index ModelBuilder::addSoftMax(Index input, float beta) {
    vector<uint32_t> dimen = dimensMap[input];

    IndexSeq input_indexes{input};
    addOperands(input_indexes, beta);

    auto output_index = addOperation(ANEURALNETWORKS_SOFTMAX, input_indexes, dimen)[0];

    return output_index;
}

ModelBuilder::Index ModelBuilder::addReLU(Index input) {
    auto dimen = dimensMap[input];

    IndexSeq input_indexes{input};

    auto output_index = addOperation(ANEURALNETWORKS_RELU, input_indexes, dimen)[0];

    return output_index;
}

ModelBuilder::Index ModelBuilder::addConcat(const IndexSeq &inputs, uint32_t axis) {
    vector<Shape> dimens;
    for (const auto &input : inputs) {
        auto &dimen = dimensMap[input];
        if (dimens.size() > 0) {
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

    return output_index;
}

ModelBuilder::Index ModelBuilder::addLRN(uint32_t input, uint32_t local_size, float bias, float alpha, float beta) {
    auto dimen = dimensMap[input];

    IndexSeq input_indexes{input};
    addOperands(input_indexes, local_size, bias, alpha, beta);

    auto output_idx = addOperation(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION, input_indexes, dimen)[0];
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
/*
char* ModelBuilder::setOperandValueFromAssets(ANeuralNetworksModel *model, AAssetManager *mgr,
                                             int32_t index, string filename) {
   AAsset* asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_UNKNOWN);
   size_t size = static_cast<size_t>(AAsset_getLength(asset));
   char* buffer = new char[size];
   bufferPointers.push_back(static_cast<void *>(buffer));
   AAsset_read(asset, buffer, static_cast<size_t>(size));
   ANeuralNetworksModel_setOperandValue(model, index, buffer, size);
   return buffer;
}*/

ModelBuilder::Index ModelBuilder::addOperand(uint32_t value) {
    if (uint32OperandMap.find(value) == uint32OperandMap.end()) {
        ANeuralNetworksOperandType type = getInt32OperandType();
        uint32_t index = addNewOperand(&type);
        auto ret = ANeuralNetworksModel_setOperandValue(model, index, &value, sizeof(value));
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
        auto ret = ANeuralNetworksModel_setOperandValue(model, index, &value, sizeof(value));
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
        auto ret = ANeuralNetworksModel_setOperandValue(model, index, &value, sizeof(value));
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
        auto ret = ANeuralNetworksModel_setOperandValue(model, index, &value, sizeof(value));
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
        auto ret = ANeuralNetworksModel_setOperandValue(model, missingInt32OperandIndex, nullptr, 0);
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
        auto ret = ANeuralNetworksModel_setOperandValue(model, missingFloat32OperandIndex, nullptr, 0);
        if (ret != ANEURALNETWORKS_NO_ERROR) {
            throw std::invalid_argument("Add operand failed, value: null, error " + getErrorCause(ret));
        }
    }
    return missingFloat32OperandIndex;
}

ModelBuilder::Index ModelBuilder::addNewOperand(ANeuralNetworksOperandType *type) {
    int ret;
    if ((ret = ANeuralNetworksModel_addOperand(model, type)) != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Add new operand failed, error " + getErrorCause(ret));
    }
    return nextIndex++;
}

ModelBuilder::Index ModelBuilder::addTensorFromMemory(const unsigned char *addr, Shape dimen) {
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);
    auto ret = ANeuralNetworksModel_setOperandValueFromMemory(model, index, dnn_model_->memory, addr - dnn_model_->data,
                                                   product(dimen) * sizeof(float));
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("addTensorFromBuffer error, ret: " + getErrorCause(ret));
    }
    dimensMap[index] = dimen;
    return index;
}

ModelBuilder::Index ModelBuilder::addTensorFromBuffer(const float *buffer, Shape dimen) {
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);
    auto ret = ANeuralNetworksModel_setOperandValue(model, index, buffer, product(dimen) * sizeof(float));
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("addTensorFromBuffer error, ret: " + getErrorCause(ret));
    }
    dimensMap[index] = dimen;
    return index;
}

ModelBuilder::Index ModelBuilder::addTensorFromBuffer(const int32_t *buffer, Shape dimen) {
    ANeuralNetworksOperandType type = getInt32OperandTypeWithDims(dimen);
    uint32_t index = addNewOperand(&type);
    auto ret = ANeuralNetworksModel_setOperandValue(model, index, buffer, product(dimen) * sizeof(int32_t));
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("addTensorFromBuffer error, ret: " + getErrorCause(ret));
    }
    dimensMap[index] = dimen;
    return index;
}

void ModelBuilder::addIndexIntoOutput(Index index) {
    outputIndexVector.push_back(index);
}

int ModelBuilder::compile(uint32_t preference) {
    int ret;
    if ((ret = ANeuralNetworksModel_identifyInputsAndOutputs(
            model,
            static_cast<uint32_t>(inputIndexVector.size()), &inputIndexVector[0],
            static_cast<uint32_t>(outputIndexVector.size()), &outputIndexVector[0])) != ANEURALNETWORKS_NO_ERROR) {

        return NN_IDENTIFY_IO | ret;
    }

    ret = ANeuralNetworksModel_finish(model);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        return NN_MODEL_FINISH | ret;
    }

    ret = ANeuralNetworksCompilation_create(model, &compilation);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        return NN_CREATE | ret;
    }

    ret = ANeuralNetworksCompilation_setPreference(compilation, preference);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        return NN_PREFERENCE | ret;
    }

    ret = ANeuralNetworksCompilation_finish(compilation);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        return NN_COMP_FINISH | ret;
    }

    return 0;
}

void ModelBuilder::registerBufferPointer(float *pointer) {
    floatBufPointers.push_back(pointer);
}

void ModelBuilder::registerBufferPointer(char *pointer) {
    charBufPointers.push_back(pointer);
}

void ModelBuilder::prepareForExecution() {
    ANeuralNetworksExecution *execution = nullptr;
    auto ret = ANeuralNetworksExecution_create(compilation, &execution);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Error in prepareForExecution, ret: " + getErrorCause(ret));
    }

    dnn_model_->execution = execution;
}

ModelBuilder::IndexSeq ModelBuilder::getInputIndexes() {
    return inputIndexVector;
}

ModelBuilder::IndexSeq ModelBuilder::getOutputIndexes() {
    return outputIndexVector;
}

ModelBuilder::Index ModelBuilder::addFC(Index input, int32_t activation, uint32_t weightIndex, uint32_t biasIndex) {

    IndexSeq input_indexes{input, weightIndex, biasIndex};
    addOperands(input_indexes, activation);
    Shape weightDimen = dimensMap[weightIndex];     // num_output, num_input
    Shape outputDimen{1, weightDimen[0]};
    auto output_idx = addOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indexes, outputDimen)[0];
    return output_idx;
}

ModelBuilder::Index ModelBuilder::getBlobIndex(std::string blobName) {
    return blobNameToIndex.at(blobName);
}

ModelBuilder::Index ModelBuilder::addAddScalar(uint32_t input, float scalar) {
    uint32_t scalarIndex = addFloat32AsTensorOperand(scalar);
    array<uint32_t, 3> inputOperands{{input, scalarIndex, addOperand(
            ModelBuilder::ACTIVATION_NONE)}};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(dimensMap[input]);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);
    dimensMap[outputOperandIndex] = dimensMap[input];

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 3, &inputOperands[0], 1, &outputOperandIndex);
    return outputOperandIndex;
}

ModelBuilder::Index ModelBuilder::addAddTensor(uint32_t input1, uint32_t input2) {
    IndexSeq input_indexes{input1, input2};
    addOperands(input_indexes, ModelBuilder::ACTIVATION_NONE);
    auto output_idx = addOperation(ANEURALNETWORKS_ADD, input_indexes, dimensMap[input1])[0];
    return output_idx;
}

ModelBuilder::Index ModelBuilder::addMulScalar(uint32_t input, float scalar) {
    Index scalarIndex = addFloat32AsTensorOperand(scalar);
    array<uint32_t, 3> inputOperands{{input, scalarIndex, addOperand(
            ModelBuilder::ACTIVATION_NONE)}};

    ANeuralNetworksOperandType outputBlobType = getFloat32OperandTypeWithDims(dimensMap[input]);
    uint32_t outputOperandIndex = addNewOperand(&outputBlobType);
    dimensMap[outputOperandIndex] = dimensMap[input];

    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MUL, 3, &inputOperands[0], 1, &outputOperandIndex);
    return outputOperandIndex;
}

ModelBuilder::Index ModelBuilder::addMulTensor(uint32_t input1, uint32_t input2) {
    IndexSeq input_indexes{input1, input2};
    addOperands(input_indexes, ModelBuilder::ACTIVATION_NONE);
    auto output_idx = addOperation(ANEURALNETWORKS_MUL, input_indexes, dimensMap[input1])[0];
    return output_idx;
}

ModelBuilder::Index ModelBuilder::addFloat32NullOperandWithDims(Shape &dims) {
    ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(dims);
    uint32_t index = addNewOperand(&type);
    ANeuralNetworksModel_setOperandValue(model, index, nullptr, 0);
    return index;
}

ModelBuilder::Index ModelBuilder::addFloat32ZeroOperandWithDims(Shape &dims) {
    auto *zeros = new float[product(dims)];
    registerBufferPointer(zeros);
    for (size_t i = 0; i < product(dims); i++) {
        zeros[i] = 0;
    }
    return addTensorFromBuffer(zeros, dims);
}

ModelBuilder::Shape ModelBuilder::getBlobDim(std::string blobName) {
    return dimensMap[getBlobIndex(std::move(blobName))];
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
    std::cout << "Shape: " << shape_vec[0] << std::endl;
    IndexSeq output_indexes;
    for (auto shape : shape_vec) {
        ANeuralNetworksOperandType type = getFloat32OperandTypeWithDims(shape);
        auto index = addNewOperand(&type);
        output_indexes.push_back(index);
        dimensMap[index] = shape;
    }

    auto ret = ANeuralNetworksModel_addOperation(model, op, input_indexes.size(), &input_indexes[0],
                                      output_indexes.size(), &output_indexes[0]);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Add operation failed, op = " + std::to_string(op) + ", error " + getErrorCause(ret));
    }
    return output_indexes;
}

ModelBuilder::~ModelBuilder() {
    ANeuralNetworksCompilation_free(compilation);   // FIXME: there will be an error when compilation is not created
    ANeuralNetworksModel_free(model);
    for (auto pointer : charBufPointers) {
        delete[] pointer;
    }
    for (auto pointer : floatBufPointers) {
        delete[] pointer;
    }
}

ModelBuilder::ModelBuilder() {
    auto ret = ANeuralNetworksModel_create(&model);
    if (ret == ANEURALNETWORKS_OUT_OF_MEMORY) {
        throw std::bad_alloc();
    }
}

void ModelBuilder::prepare() {
    dnn_model_ = std::make_unique<Model>();
}

void ModelBuilder::setMemory(int fd, size_t size, size_t offset) {
    std::cout << fd << ", " << size << ", " << offset << std::endl;
    ANeuralNetworksMemory *mem = nullptr;
    auto ret = ANeuralNetworksMemory_createFromFd(size, PROT_READ, fd, offset, &mem);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Error in setMemory, ret: " + getErrorCause(ret));
    }
    dnn_model_->memory = mem;
}

void ModelBuilder::setBuffer(unsigned char *data, size_t data_size) {
    dnn_model_->data = data;
}

std::unique_ptr<Model> ModelBuilder::finish() {
    return std::move(dnn_model_);
}

uint32_t product(const vector<uint32_t> &v) {
    return static_cast<uint32_t> (accumulate(v.begin(), v.end(), 1, std::multiplies<uint32_t>()));
}
