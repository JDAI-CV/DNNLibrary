//
// Created by daquexian on 2017/11/8.
//

#ifndef NNAPIEXAMPLE_MODELBUILDER_H
#define NNAPIEXAMPLE_MODELBUILDER_H

#include <string>
#include <vector>
#include <numeric>
#include <map>
#include <memory>
#include <optional>

#include <common/StrKeyMap.h>
#include <common/Shaper.h>
#include "Model.h"
#include <NeuralNetworksWrapper.h>

class ModelBuilder {
public:
    using Index = uint32_t;
    using IndexSeq = std::vector<Index>;
    using Shape = Shaper::Shape;

private:
    std::unique_ptr<Model> dnn_model_;
    std::vector<std::string> ordered_operands_;  // operands in insertion order, for printing in finish()
    StrKeyMap<Index> operand_indexes_;
    Shaper shaper_;
    IndexSeq input_index_vec_;
    IndexSeq output_index_vec_;
    std::map<uint32_t , Index> uint32_operand_map_;
    std::map<int32_t , Index> int32_operand_map_;
    std::map<float, Index> float32_operand_map_;
    std::map<float, Index> float32_as_tensor_operand_map_;

    uint32_t int32_missing_index = UINT32_MAX;
    uint32_t float32_missing_index = UINT32_MAX;

    uint32_t next_index_ = 0;

    void AppendOperandIndex(const std::string &name, Index index);
    uint32_t AddNewOperand(const android::nn::wrapper::OperandType &type);

    // IndexSeq addOperation(int op, IndexSeq input_indexes, Shape... shapes);
    template <typename... Shapes>
    IndexSeq AddOperation(int op, IndexSeq input_indexes, Shapes... shapes);

    Index AddOperand(int32_t value);
    Index AddOperand(float value);
    Index AddOperand(uint32_t value);
    Index AddFloat32AsTensorOperand(float value);
    Index AddInt32NullOperand();
    Index AddFloat32NullOperand();
    Index AddFloat32NullOperandWithDims(Shape &dims);
    Index AddFloat32ZeroOperandWithDims(Shape &dims);

    android::nn::wrapper::OperandType GetOperandType(const android::nn::wrapper::Type &type);
    android::nn::wrapper::OperandType GetOperandType(const android::nn::wrapper::Type &type, const Shape &dims);

public:
    static const int MAX_POOL = 0;
    static const int AVE_POOL = 1;

    static const int32_t ACTIVATION_NONE = ANEURALNETWORKS_FUSED_NONE;
    static const int32_t ACTIVATION_RELU = ANEURALNETWORKS_FUSED_RELU;

    static const uint32_t PREFERENCE_FAST_SINGLE_ANSWER = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER;
    static const uint32_t PREFERENCE_SUSTAINED_SPEED = ANEURALNETWORKS_PREFER_SUSTAINED_SPEED;
    static const uint32_t PREFERENCE_LOW_POWER = ANEURALNETWORKS_PREFER_LOW_POWER;

    static std::string GetErrorCause(int errorCode);

    Index GetBlobIndex(const std::string &blobName);
    Shape GetBlobDim(const std::string &blobName);
    Shape GetBlobDim(Index index);
    Index AddInput(std::string name, uint32_t height, uint32_t width, uint32_t depth);
    Index AddDepthWiseConv(const std::string &input_name, int32_t strideX, int32_t strideY,
                                         int32_t paddingLeft,
                                         int32_t paddingRight, int32_t paddingBottom, int32_t paddingTop,
                                         int32_t activation,
                                         int32_t depthMultiplier, const std::string &weight_name,
                                         const std::optional<std::string> &bias_name,
                                         const std::string &output_name);
    Index AddConv(const std::string &input_name, int32_t strideX, int32_t strideY, int32_t paddingLeft,
                                int32_t paddingRight, int32_t paddingTop, int32_t paddingBottom,
                                int32_t activation, const std::string &weight_name,
                                const std::optional<std::string> &bias_name, const std::string &output_name);
    Index AddTensorFromBuffer(const std::string &name, const float *buffer, Shape dimen);
    Index AddTensorFromBuffer(const std::string &name, const int32_t *buffer, Shape dimen);
    Index AddTensorFromMemory(const std::string &name, const uint8_t *addr, Shape dimen);
    Index AddFC(const std::string &input_name, int32_t activation, const std::string &weight_name,
                const std::optional<std::string> &bias_name, const std::string &output_name);
    Index
    AddPool(const std::string &input_name, int32_t strideX, int32_t strideY, int32_t paddingLeft, int32_t paddingRight,
            int32_t paddingTop, int32_t paddingBottom, int32_t height, int32_t width, int32_t activation,
            uint32_t poolingType, const std::string &output_name);
    Index AddSoftMax(const std::string &input_name, float beta, const std::string &output_name);
    Index AddAddScalar(const std::string &input_name, float scalar, std::string output_name);
    Index AddAddTensor(const std::string &input1_name, const std::string &input2_name, const std::string &output_name);
    Index AddMulScalar(const std::string &input_name, float scalar, const std::string &output_name);
    Index AddMulTensor(const std::string &input1_name, const std::string &input2_name, const std::string &output_name);
    Index AddReLU(const std::string &input_name, const std::string &output_name);
    Index AddConcat(const std::vector<std::string> &input_names, uint32_t axis, const std::string &output_name);
    Index AddLRN(const std::string &input_name, uint32_t local_size, float bias, float alpha, float beta,
                 const std::string &output_name);
#if __ANDROID_API__ >= __ANDROID_API_P__
    Index AddStridedSlice(const std::string &input_name, const std::vector<int32_t> &starts,
                          const std::vector<int32_t> &ends,
                          const std::vector<int32_t> &strides, int32_t beginMask, int32_t endMask,
                          int32_t shrinkAxisMask, const std::string &output_name);
    Index AddSpaceToBatchND(const std::string &input_name, const std::vector<int32_t> &block_sizes,
            const std::vector<int32_t> &pads, const std::string &output_name);
    Index AddBatchToSpaceND(const std::string &input_name, const std::vector<int32_t> &block_sizes,
            const std::string &output_name);
    ModelBuilder &AllowFp16(const bool allowed);
#endif
    ModelBuilder &AddOutput(const std::string &name);
    std::unique_ptr<Model> Compile(uint32_t preference);
    IndexSeq GetInputIndexes();
    IndexSeq GetOutputIndexes();
    void RegisterBufferPointer(std::unique_ptr<int8_t[]> &&pointer);
    void RegisterBufferPointer(std::unique_ptr<float[]> &&pointer);
    void RegisterBufferPointer(std::unique_ptr<uint8_t[]> &&pointer);

    void Prepare();
    void SetMemory(int fd, size_t size, size_t offset);
    void SetBasePtr(uint8_t *data);
    // Add scalar operands, aka ANEURALNETWORKS_FLOAT32, ANEURALNETWORKS_INT32, ANEURALNETWORKS_UINT32. It should not be used to append tensor operand indexes to a IndexSeq
    template <typename... Args>
    void AddScalarOperands(IndexSeq &indexes, Args... args) {
        (indexes.push_back(AddOperand(args)), ...);
    }
};
#endif //NNAPIEXAMPLE_MODELBUILDER_H
