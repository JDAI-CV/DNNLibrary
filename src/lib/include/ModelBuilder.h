//
// Created by daquexian on 2017/11/8.
//

#ifndef NNAPIEXAMPLE_MODELBUILDER_H
#define NNAPIEXAMPLE_MODELBUILDER_H

#include <android/NeuralNetworks.h>
#include <string>
#include <vector>
#include <numeric>
#include <map>
#include <memory>
#include <optional>

#include <common/StrKeyMap.h>
#include <common/Shaper.h>
#include "Model.h"

class ModelBuilder {
public:
    using Index = uint32_t;
    using IndexSeq = std::vector<Index>;
    using Shape = Shaper::Shape;

private:
    std::unique_ptr<Model> dnn_model_;
    std::vector<std::string> ordered_operands;  // operands in insertion order, for printing in finish()
    StrKeyMap<Index> operand_indexes;
    Shaper shaper;
    IndexSeq inputIndexVector;
    IndexSeq outputIndexVector;
    std::map<uint32_t , Index> uint32OperandMap;
    std::map<int32_t , Index> int32OperandMap;
    std::map<float, Index> float32OperandMap;
    std::map<float, Index> float32AsTensorOperandMap;

    uint32_t missingInt32OperandIndex = UINT32_MAX;
    uint32_t missingFloat32OperandIndex = UINT32_MAX;

    uint32_t nextIndex = 0;

    void AppendOperandIndex(const std::string &name, Index index);
    uint32_t addNewOperand(ANeuralNetworksOperandType *type);

    // IndexSeq addOperation(int op, IndexSeq input_indexes, Shape... shapes);
    template <typename... Shapes>
    IndexSeq addOperation(int op, IndexSeq input_indexes, Shapes... shapes);

    Index addOperand(int32_t value);
    Index addOperand(float value);
    Index addOperand(uint32_t value);
    Index addFloat32AsTensorOperand(float value);
    Index addInt32NullOperand();
    Index addFloat32NullOperand();
    Index addFloat32NullOperandWithDims(Shape &dims);
    Index addFloat32ZeroOperandWithDims(Shape &dims);

    ANeuralNetworksOperandType getFloat32OperandTypeWithDims(Shape &dims);
    ANeuralNetworksOperandType getInt32OperandTypeWithDims(Shape &dims);

    ANeuralNetworksOperandType getInt32OperandType();
    ANeuralNetworksOperandType getFloat32OperandType();

public:
    static const int MAX_POOL = 0;
    static const int AVE_POOL = 1;

    static const uint32_t ACTIVATION_NONE = ANEURALNETWORKS_FUSED_NONE;
    static const uint32_t ACTIVATION_RELU = ANEURALNETWORKS_FUSED_RELU;

    static const uint32_t PREFERENCE_FAST_SINGLE_ANSWER = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER;
    static const uint32_t PREFERENCE_SUSTAINED_SPEED = ANEURALNETWORKS_PREFER_SUSTAINED_SPEED;
    static const uint32_t PREFERENCE_LOW_POWER = ANEURALNETWORKS_PREFER_LOW_POWER;

    static std::string getErrorCause(int errorCode);

    Index getBlobIndex(const std::string &blobName);
    Shape getBlobDim(const std::string &blobName);
    Shape getBlobDim(Index index);
    Index addInput(std::string name, uint32_t height, uint32_t width, uint32_t depth);
    Index addDepthWiseConv(const std::string &input_name, int32_t strideX, int32_t strideY,
                                         int32_t paddingLeft,
                                         int32_t paddingRight, int32_t paddingBottom, int32_t paddingTop,
                                         int32_t activation,
                                         int32_t depthMultiplier, const std::string &weight_name,
                                         const std::optional<std::string> &bias_name,
                                         const std::string &output_name);
    Index addConv(const std::string &input_name, int32_t strideX, int32_t strideY, int32_t paddingLeft,
                                int32_t paddingRight, int32_t paddingTop, int32_t paddingBottom,
                                int32_t activation, const std::string &weight_name,
                                const std::optional<std::string> &bias_name, const std::string &output_name);
    Index addTensorFromBuffer(const std::string &name, const float *buffer, Shape dimen);
    Index addTensorFromBuffer(const std::string &name, const int32_t *buffer, Shape dimen);
    Index addTensorFromMemory(const std::string &name, const uint8_t *addr, Shape dimen);
    Index addFC(const std::string &input_name, int32_t activation, const std::string &weight_name,
                const std::optional<std::string> &bias_name, const std::string &output_name);
    Index
    addPool(const std::string &input_name, int32_t strideX, int32_t strideY, int32_t paddingLeft, int32_t paddingRight,
            int32_t paddingTop, int32_t paddingBottom, int32_t height, int32_t width, int32_t activation,
            uint32_t poolingType, const std::string &output_name);
    Index addSoftMax(const std::string &input_name, float beta, const std::string &output_name);
    Index addAddScalar(const std::string &input_name, float scalar, std::string output_name);
    Index addAddTensor(const std::string &input1_name, const std::string &input2_name, const std::string &output_name);
    Index addMulScalar(const std::string &input_name, float scalar, const std::string &output_name);
    Index addMulTensor(const std::string &input1_name, const std::string &input2_name, const std::string &output_name);
    Index addReLU(const std::string &input_name, const std::string &output_name);
    Index addConcat(const std::vector<std::string> &input_names, uint32_t axis, const std::string &output_name);
    Index addLRN(const std::string &input_name, uint32_t local_size, float bias, float alpha, float beta,
                 const std::string &output_name);
#if __ANDROID_API__ >= __ANDROID_API_P__
    Index addStridedSlice(const std::string &input_name, const std::vector<int32_t> &starts,
                          const std::vector<int32_t> &ends,
                          const std::vector<int32_t> &strides, int32_t beginMask, int32_t endMask,
                          int32_t shrinkAxisMask, const std::string &output_name);
    Index addSpaceToBatchND(const std::string &input_name, const std::vector<int32_t> &block_sizes,
            const std::vector<int32_t> &pads, const std::string &output_name);
    Index addBatchToSpaceND(const std::string &input_name, const std::vector<int32_t> &block_sizes,
            const std::string &output_name);
#endif
    void addOutput(const std::string &name);
    std::unique_ptr<Model> compile(uint32_t preference);
    IndexSeq getInputIndexes();
    IndexSeq getOutputIndexes();
    void registerBufferPointer(std::unique_ptr<int8_t[]> &&pointer);
    void registerBufferPointer(std::unique_ptr<float[]> &&pointer);
    void registerBufferPointer(std::unique_ptr<uint8_t[]> &&pointer);

    void prepare();
    void setMemory(int fd, size_t size, size_t offset);
    void setBasePtr(uint8_t *data);
    template <typename... Args>
    void addOperands(IndexSeq &indexes, Args... args) {
        (indexes.push_back(addOperand(args)), ...);
    }
};
#endif //NNAPIEXAMPLE_MODELBUILDER_H
