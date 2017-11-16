//
// Created by daquexian on 2017/11/8.
//

#ifndef NNAPIEXAMPLE_MODELBUILDER_H
#define NNAPIEXAMPLE_MODELBUILDER_H

#include <android/asset_manager.h>
#include <android/NeuralNetworks.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <linux/stat.h>
#include <numeric>
#include <cassert>
#include <sstream>
#include <cmath>
#include <jni.h>
#include "Model.h"

class ModelBuilder {
private:
    ANeuralNetworksModel* model = nullptr;
    AAssetManager *mgr;
    std::vector<char*> bufferPointers;
    // NHWC
    std::map<uint32_t, std::vector<uint32_t>> dimensMap;
    std::vector<uint32_t> inputIndexVector;
    std::vector<uint32_t> outputIndexVector;
    std::map<int32_t , uint32_t> int32OperandMap;
    std::map<float, uint32_t> float32OperandMap;

    std::map<std::string, uint32_t> blobNameToIndex;

    uint32_t missingInt32OperandIndex = UINT32_MAX;
    uint32_t missingFloat32OperandIndex = UINT32_MAX;

    uint32_t nextIndex = 0;

    static const uint32_t WRONG_INPUT = UINT32_MAX -1;
    static const uint32_t WRONG_POOLING_TYPE = UINT32_MAX -2;
    static const int WRONG_OPERAND_INDEX = -10;

    uint32_t addNewOperand(ANeuralNetworksOperandType *type);

    uint32_t addInt32Operand(int32_t value);
    uint32_t addFloat32Operand(float value);
    uint32_t addInt32NullOperand();
    uint32_t addFloat32NullOperand();
    uint32_t addFloat32NullOperandWithDims(std::vector<uint32_t> &dims);
    uint32_t addFloat32ZeroOperandWithDims(std::vector<uint32_t> &dims);

    uint32_t addConvWeight(std::string name, uint32_t height, uint32_t width, uint32_t inputDepth,
                           uint32_t outputDepth);

    uint32_t addFcWeight(std::string name, uint32_t inputSize, uint32_t outputNum);

    uint32_t addWeight(std::string name, std::vector<uint32_t> dimen);

    uint32_t addBias(std::string name, uint32_t outputDepth);

    ANeuralNetworksOperandType getFloat32OperandTypeWithDims(std::vector<uint32_t> &dims);

    ANeuralNetworksOperandType getInt32OperandType();
    ANeuralNetworksOperandType getFloat32OperandType();

    char* setOperandValueFromAssets(ANeuralNetworksModel *model, AAssetManager *mgr, int32_t index,
                                    std::string filename);

public:
    static const int MAX_POOL = 0;
    static const int AVE_POOL = 1;

    static const int32_t ACTIVATION_NONE = ANEURALNETWORKS_FUSED_NONE;
    static const int32_t ACTIVATION_RELU = ANEURALNETWORKS_FUSED_RELU;

    static const uint32_t PREFERENCE_FAST_SINGLE_ANSWER = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER;
    static const uint32_t PREFERENCE_SUSTAINED_SPEED = ANEURALNETWORKS_PREFER_SUSTAINED_SPEED;
    static const uint32_t PREFERENCE_LOW_POWER = ANEURALNETWORKS_PREFER_LOW_POWER;

    static const uint32_t MF_LAYER_END = 0;
    static const uint32_t MF_CONV = 1;
    static const uint32_t MF_MAX_POOL = 2;
    static const uint32_t MF_AVE_POOL = 3;
    static const uint32_t MF_FC = 4;
    static const uint32_t MF_SOFTMAX = 5;
    static const uint32_t MF_INPUT = 6;
    static const uint32_t MF_MUL = 7;
    static const uint32_t MF_ADD = 8;
    static const uint32_t MF_RELU = 9;
    static const uint32_t MF_CONCAT = 10;

    static const uint32_t MF_ACTIVATION_NONE = 0;
    static const uint32_t MF_ACTIVATION_RELU = 1;

    static const uint32_t MF_TENSOR_OP = 0;
    static const uint32_t MF_SCALAR_OP = 1;
    static const uint32_t MF_ARRAY_OP = 2;

    static const uint32_t MF_STRING_END = 0;

    static const uint32_t MF_PARAM_END = 0;
    static const uint32_t MF_PADDING_LEFT = 1;
    static const uint32_t MF_PADDING_RIGHT = 2;
    static const uint32_t MF_PADDING_TOP = 3;
    static const uint32_t MF_PADDING_BOTTOM = 4;
    static const uint32_t MF_STRIDE_X = 5;
    static const uint32_t MF_STRIDE_Y = 6;
    static const uint32_t MF_FILTER_HEIGHT = 7;
    static const uint32_t MF_FILTER_WIDTH = 8;
    static const uint32_t MF_NUM_OUTPUT = 9;
    static const uint32_t MF_WEIGHT = 10;
    static const uint32_t MF_BIAS = 11;
    static const uint32_t MF_ACTIVATION = 12;
    static const uint32_t MF_TOP_NAME = 13;
    static const uint32_t MF_BETA = 14;

    ANeuralNetworksCompilation* compilation = nullptr;

    int init(AAssetManager *mgr);
    ModelBuilder& readFromFile(std::string filename);
    uint32_t getBlobIndex(std::string blobName);
    std::vector<uint32_t> getBlobDim(std::string blobName);
    std::vector<uint32_t> getBlobDim(uint32_t index);
    uint32_t addInput(uint32_t height, uint32_t width, uint32_t depth);
    uint32_t addDepthWiseConv(uint32_t input, int32_t strideX, int32_t strideY, int32_t paddingLeft,
                              int32_t paddingRight, int32_t paddingBottom, int32_t paddingTop,
                              int32_t height, int32_t width, int32_t activation,
                              uint32_t outputDepth,
                              int32_t depthMultiplier, uint32_t weightIndex, uint32_t biasIndex);
    uint32_t addConv(uint32_t input, int32_t strideX, int32_t strideY, int32_t paddingLeft,
                     int32_t paddingRight, int32_t paddingBottom, int32_t paddingTop,
                     int32_t height, int32_t width, int32_t activation, uint32_t outputDepth,
                     uint32_t weightIndex, uint32_t biasIndex);
    uint32_t addWeightOrBiasFromBuffer(const void *buffer, std::vector<uint32_t> dimen);
    uint32_t addFC(uint32_t input, uint32_t outputNum, int32_t activation,
                   uint32_t weightIndex, uint32_t biasIndex);
    uint32_t addCaffePool(uint32_t input, int32_t strideX, int32_t strideY, int32_t paddingLeft,
                          int32_t paddingRight, int32_t paddingTop, int32_t paddingBottom,
                          int32_t height, int32_t width, int32_t activation,
                          uint32_t poolingType);
    uint32_t addSoftMax(uint32_t input, float beta);
    uint32_t addAddScalar(uint32_t input, float scalar);
    uint32_t addAddTensor(uint32_t input1, uint32_t input2);
    uint32_t addMulScalar(uint32_t input, float scalar);
    uint32_t addMulTensor(uint32_t input1, uint32_t input2);
    uint32_t addReLU(uint32_t input);
    uint32_t addConcat(const std::vector<uint32_t> &inputs, uint32_t axis);
    void addIndexIntoOutput(uint32_t index);
    int compile(uint32_t preference);
    void prepareForExecution(Model &model);
    std::vector<uint32_t> getInputIndexes();
    std::vector<uint32_t> getOutputIndexes();
    int setInputBuffer(const Model& model, int32_t index, void *buffer, size_t length);
    int setOutputBuffer(const Model& model, int32_t index, void *buffer, size_t length);
    void clear();

    ModelBuilder();
};




// TODO: Remove when O MR1 Beta 2 is available.
__attribute__((weak))
extern "C" int ANeuralNetworksModel_setInputsAndOutputs(
        ANeuralNetworksModel* model,
        uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
        const uint32_t* outputs);

extern "C" int ANeuralNetworksModel_identifyInputsAndOutputs(
        ANeuralNetworksModel* model,
        uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
        const uint32_t* outputs);

uint32_t product(const std::vector<uint32_t> &v);
#endif //NNAPIEXAMPLE_MODELBUILDER_H
