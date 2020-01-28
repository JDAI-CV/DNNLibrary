//
// Created by daquexian on 2017/11/8.
//

#ifndef NNAPIEXAMPLE_MODELBUILDER_H
#define NNAPIEXAMPLE_MODELBUILDER_H

#include <common/Shaper.h>
#include <common/StrKeyMap.h>
#include <common/data_types.h>
#include <dnnlibrary/Device.h>
#include <dnnlibrary/Model.h>
#include <dnnlibrary/NeuralNetworksWrapper.h>

#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace dnn {
class ModelBuilder {
   public:
    using Index = uint32_t;
    using IndexSeq = std::vector<Index>;
    using Shape = Shaper::Shape;

    struct QuantInfo {
        android::nn::wrapper::Type type_;
        std::vector<float> scales_;
        dnn::optional<int> zero_point_;
    };

   private:
    std::unique_ptr<Model> dnn_model_;
    std::vector<std::string> ordered_operands_;  // operands in insertion order,
                                                 // for printing in finish()
    StrKeyMap<Index> operand_indexes_;
    Shaper shaper_;
    IndexSeq input_index_vec_;
    IndexSeq output_index_vec_;
    std::map<bool, Index> bool_operand_map_;
    std::map<uint32_t, Index> uint32_operand_map_;
    std::map<int32_t, Index> int32_operand_map_;
    std::map<float, Index> float32_operand_map_;
    std::map<float, Index> float32_as_tensor_operand_map_;
    StrKeyMap<android::nn::wrapper::OperandType> operand_types_;
    // imm_blob_inputs_ and imm_blob_outputs_ is to automatically determine the
    // output of the model
    std::set<std::string> imm_blob_inputs_;
    std::set<std::string> imm_blob_outputs_;

    uint32_t int32_missing_index = UINT32_MAX;
    uint32_t float32_missing_index = UINT32_MAX;

    uint32_t next_index_ = 0;

    void RegisterOperand(const std::string &name, Index index,
                         const android::nn::wrapper::OperandType &operand_type);
    uint32_t AddNewOperand(const android::nn::wrapper::OperandType &type);

    template <typename... OperandTypes>
    IndexSeq AddOperation(int op, IndexSeq input_indexes,
                          OperandTypes... output_types);

    Index OperandFromScalar(bool value);
    Index OperandFromScalar(int32_t value);
    Index OperandFromScalar(float value);
    Index OperandFromScalar(uint32_t value);
    Index OperandFromScalar(FuseCode value);
    Index AddMissingOperand(
        const android::nn::wrapper::OperandType &operand_type);
    Index FillOperand(const std::string &name,
                      const android::nn::wrapper::OperandType &operand_type,
                      const float val);
    Index FillOperand(const std::string &name,
                      const android::nn::wrapper::OperandType &operand_type,
                      const int32_t val);
    Index FillOperand(const std::string &name,
                      const android::nn::wrapper::OperandType &operand_type,
                      const uint32_t val);

    android::nn::wrapper::OperandType GetOperandType(
        const android::nn::wrapper::Type &type);
    android::nn::wrapper::OperandType GetOperandType(
        const android::nn::wrapper::Type &type, const Shape &dims,
        const dnn::optional<QuantInfo> &quant_info = dnn::nullopt);
    android::nn::wrapper::OperandType GetOperandType(
        const QuantInfo &quant_info, const Shape &dims);

    int32_t android_api_level() const;

    const NnApi *nnapi_ = nullptr;

   public:
    ModelBuilder();
    enum class PoolingType { MAX_POOL, AVE_POOL };

    static const int32_t ACTIVATION_NONE = ANEURALNETWORKS_FUSED_NONE;
    static const int32_t ACTIVATION_RELU = ANEURALNETWORKS_FUSED_RELU;

    static const uint32_t PREFERENCE_FAST_SINGLE_ANSWER =
        ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER;
    static const uint32_t PREFERENCE_SUSTAINED_SPEED =
        ANEURALNETWORKS_PREFER_SUSTAINED_SPEED;
    static const uint32_t PREFERENCE_LOW_POWER =
        ANEURALNETWORKS_PREFER_LOW_POWER;

    Index GetBlobIndex(const std::string &blobName);
    Shape GetBlobDim(const std::string &blobName);
    Shape GetBlobDim(Index index);
    Index AddInput(std::string name, const uint32_t batch,
                   const uint32_t height, const uint32_t width,
                   const uint32_t depth);
    Index AddInput(std::string name,
                   const android::nn::wrapper::OperandType &operand_type);
    // ModelBuilder auto generated methods start
    expected<Unit, std::string> AddLayer_CONV_2D(
        const std::string &input, const std::string &weight,
        const dnn::optional<std::string> &bias, int32_t padding_left,
        int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
        int32_t stride_x, int32_t stride_y, FuseCode fuse_code, bool nchw,
        int32_t dilation_x, int32_t dilation_y, const std::string &output,
        const dnn::optional<QuantInfo> &output_quant_info = dnn::nullopt);

   private:
    expected<Unit, std::string> AddLayer_CONV_2D_Impl(
        const std::string &input, const std::string &weight,
        const dnn::optional<std::string> &bias, int32_t padding_left,
        int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
        int32_t stride_x, int32_t stride_y, FuseCode fuse_code, bool nchw,
        int32_t dilation_x, int32_t dilation_y, const std::string &output,
        const dnn::optional<QuantInfo> &output_quant_info = dnn::nullopt);

   public:
    expected<Unit, std::string> AddLayer_AVERAGE_POOL_2D(
        const std::string &input, int32_t padding_left, int32_t padding_right,
        int32_t padding_top, int32_t padding_bottom, int32_t stride_x,
        int32_t stride_y, int32_t kernel_width, int32_t kernel_height,
        FuseCode fuse_code, const std::string &output,
        const dnn::optional<QuantInfo> &output_quant_info = dnn::nullopt);
    expected<Unit, std::string> AddLayer_MAX_POOL_2D(
        const std::string &input, int32_t padding_left, int32_t padding_right,
        int32_t padding_top, int32_t padding_bottom, int32_t stride_x,
        int32_t stride_y, int32_t kernel_width, int32_t kernel_height,
        FuseCode fuse_code, const std::string &output,
        const dnn::optional<QuantInfo> &output_quant_info = dnn::nullopt);
    expected<Unit, std::string> AddLayer_RELU(const std::string &input,
                                              const std::string &output);
    expected<Unit, std::string> AddLayer_SOFTMAX(const std::string &input,
                                                 float beta,
                                                 const std::string &output);
    expected<Unit, std::string> AddLayer_FULLY_CONNECTED(
        const std::string &input, const std::string &weight,
        const dnn::optional<std::string> &bias, FuseCode fuse_code,
        const std::string &output,
        const dnn::optional<QuantInfo> &output_quant_info = dnn::nullopt);
    expected<Unit, std::string> AddLayer_ADD(
        const std::string &input1, const std::string &input2,
        FuseCode fuse_code, const std::string &output,
        const dnn::optional<QuantInfo> &output_quant_info = dnn::nullopt);
    expected<Unit, std::string> AddLayer_CONCATENATION(
        const std::vector<std::string> &inputs, int32_t axis,
        const std::string &output);
    expected<Unit, std::string> AddLayer_DEPTHWISE_CONV_2D(
        const std::string &input, const std::string &weight,
        const dnn::optional<std::string> &bias, int32_t padding_left,
        int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
        int32_t stride_x, int32_t stride_y, int32_t depth_multiplier,
        FuseCode fuse_code, const std::string &output,
        const dnn::optional<QuantInfo> &output_quant_info = dnn::nullopt);
    expected<Unit, std::string> AddLayer_BATCH_TO_SPACE_ND(
        const std::string &input, const std::vector<int32_t> &block_sizes,
        const std::string &output);
    expected<Unit, std::string> AddLayer_SPACE_TO_BATCH_ND(
        const std::string &input, const std::vector<int32_t> &block_sizes,
        const std::vector<int32_t> &pads, const std::string &output);
    expected<Unit, std::string> AddLayer_STRIDED_SLICE(
        const std::string &input, const std::vector<int32_t> &starts,
        const std::vector<int32_t> &ends, const std::vector<int32_t> &strides,
        int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask,
        const std::string &output);
    expected<Unit, std::string> AddLayer_MUL(
        const std::string &input1, const std::string &input2,
        FuseCode fuse_code, const std::string &output,
        const dnn::optional<QuantInfo> &output_quant_info = dnn::nullopt);
    expected<Unit, std::string> AddLayer_DEQUANTIZE(const std::string &input,
                                                    const std::string &output);
    expected<Unit, std::string> AddLayer_LOCAL_RESPONSE_NORMALIZATION(
        const std::string &input, int32_t radius, float bias, float alpha,
        float beta, const std::string &output);
    expected<Unit, std::string> AddLayer_TANH(const std::string &input,
                                              const std::string &output);
    expected<Unit, std::string> AddLayer_FLOOR(const std::string &input,
                                               const std::string &output);
    expected<Unit, std::string> AddLayer_LOGISTIC(const std::string &input,
                                                  const std::string &output);
    expected<Unit, std::string> AddLayer_PRELU(const std::string &input,
                                               const std::string &alpha,
                                               const std::string &output);

   private:
    expected<Unit, std::string> AddLayer_PRELU_Impl(const std::string &input,
                                                    const std::string &alpha,
                                                    const std::string &output);

   public:
    expected<Unit, std::string> AddLayer_POW(const std::string &input,
                                             const std::string &exp,
                                             const std::string &output);
    expected<Unit, std::string> AddLayer_NEG(const std::string &input,
                                             const std::string &output);
    expected<Unit, std::string> AddLayer_MINIMUM(const std::string &input1,
                                                 const std::string &input2,
                                                 const std::string &output);
    expected<Unit, std::string> AddLayer_MAXIMUM(const std::string &input1,
                                                 const std::string &input2,
                                                 const std::string &output);
    expected<Unit, std::string> AddLayer_LOG(const std::string &input,
                                             const std::string &output);
    expected<Unit, std::string> AddLayer_ABS(const std::string &input,
                                             const std::string &output);
    expected<Unit, std::string> AddLayer_EXP(const std::string &input,
                                             const std::string &output);
    expected<Unit, std::string> AddLayer_SUB(const std::string &input1,
                                             const std::string &input2,
                                             FuseCode fuse_code,
                                             const std::string &output);

   private:
    expected<Unit, std::string> AddLayer_SUB_Impl(const std::string &input1,
                                                  const std::string &input2,
                                                  FuseCode fuse_code,
                                                  const std::string &output);

   public:
    // ModelBuilder auto generated methods end
    Index AddTensorFromPersistentBuffer(
        const std::string &name, const void *buffer,
        const android::nn::wrapper::OperandType &operand_type);
    Index AddTensorFromBuffer(
        const std::string &name, const void *buffer,
        const android::nn::wrapper::OperandType &operand_type);
    Index AddTensorFromMemory(const std::string &name, const uint8_t *addr,
                              Shape dimen);
    Index AddMaxPool(
        const std::string &input_name, int32_t strideX, int32_t strideY,
        int32_t paddingLeft, int32_t paddingRight, int32_t paddingTop,
        int32_t paddingBottom, int32_t height, int32_t width,
        int32_t activation, const std::string &output_name,
        const dnn::optional<QuantInfo> &output_quant_info = dnn::nullopt);
    Index AddAvePool(
        const std::string &input_name, int32_t strideX, int32_t strideY,
        int32_t paddingLeft, int32_t paddingRight, int32_t paddingTop,
        int32_t paddingBottom, int32_t height, int32_t width,
        int32_t activation, const std::string &output_name,
        const dnn::optional<QuantInfo> &output_quant_info = dnn::nullopt);
    ModelBuilder &AllowFp16(const bool allowed);
    ModelBuilder &AddOutput(const std::string &name);
    std::unique_ptr<Model> Compile(uint32_t preference);
    IndexSeq GetInputIndexes();
    IndexSeq GetOutputIndexes();
    void RegisterBufferPointer(std::unique_ptr<int8_t[]> &&pointer);
    void RegisterBufferPointer(std::unique_ptr<float[]> &&pointer);
    void RegisterBufferPointer(std::unique_ptr<uint8_t[]> &&pointer);
    void RegisterBufferPointer(std::unique_ptr<int32_t[]> &&pointer);

    void Prepare();
    void SetMemory(int fd, size_t size, size_t offset);
    void SetBasePtr(uint8_t *data);
    // Add scalar operands, aka ANEURALNETWORKS_FLOAT32, ANEURALNETWORKS_INT32,
    // ANEURALNETWORKS_UINT32. It should not be used to append tensor operand
    // indexes to a IndexSeq
    template <typename... Args>
    void AddScalarOperands(IndexSeq &indexes, Args... args) {
        (indexes.push_back(OperandFromScalar(args)), ...);
    }

    dnn::optional<std::vector<Device>> GetDevices();
};
}  // namespace dnn
#endif  // NNAPIEXAMPLE_MODELBUILDER_H
