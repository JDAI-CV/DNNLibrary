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
    std::map<uint32_t, uint32_t> uint32OperandMap;
    std::map<float, uint32_t> float32OperandMap;

    uint32_t nextIndex = 0;

    static const uint32_t WRONG_INPUT = UINT32_MAX -1;
    static const uint32_t WRONG_POOLING_TYPE = UINT32_MAX -2;
    static const int WRONG_OPERAND_INDEX = -10;

    uint32_t addOperand(ANeuralNetworksOperandType *type);

    uint32_t addUInt32Operand(uint32_t value);
    uint32_t addFloat32Operand(float value);

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

    static const uint32_t ACTIVATION_NONE = ANEURALNETWORKS_FUSED_NONE;
    static const uint32_t ACTIVATION_RELU = ANEURALNETWORKS_FUSED_RELU;

    static const uint32_t PREFERENCE_FAST_SINGLE_ANSWER = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER;
    static const uint32_t PREFERENCE_SUSTAINED_SPEED = ANEURALNETWORKS_PREFER_SUSTAINED_SPEED;
    static const uint32_t PREFERENCE_LOW_POWER = ANEURALNETWORKS_PREFER_LOW_POWER;

    ANeuralNetworksCompilation* compilation = nullptr;

    int init(AAssetManager *mgr);
    uint32_t addInput(uint32_t height, uint32_t width);
    uint32_t addConv(std::string name, uint32_t input, uint32_t strideX, uint32_t strideY,
                     uint32_t paddingW, uint32_t paddingH, uint32_t height, uint32_t width,
                     uint32_t activation, uint32_t outputDepth);
    uint32_t addPool(uint32_t input, uint32_t strideX, uint32_t strideY,
                     uint32_t paddingW, uint32_t paddingH,
                     uint32_t height,uint32_t width,
                     uint32_t activation, uint32_t poolingType);
    uint32_t addFC(std::string name, uint32_t input, uint32_t outputNum, uint32_t activation);
    uint32_t addSoftMax(uint32_t input);
    void addIndexIntoOutput(uint32_t index);
    int compile(uint32_t preference);
    Model prepareForExecution();
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
