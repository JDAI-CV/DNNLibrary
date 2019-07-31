//
// Created by daquexian on 2017/11/8.
//
#include <dnnlibrary/ModelBuilder.h>

#include <sys/mman.h>
#include <sys/system_properties.h>
#include <algorithm>
#include <array>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>

#include <android_log_helper.h>
#include <common/data_types.h>
#include <common/helper.h>
#include <glog/logging.h>
#include <nnapi_helper.h>
#include <operand_helper.h>

namespace dnn {
using std::array;
using std::ifstream;
using std::ios;
using std::streamsize;
using std::string;
using std::stringstream;
using std::vector;
using namespace android::nn::wrapper;

void ModelBuilder::RegisterOperand(const std::string &name,
                                   ModelBuilder::Index index,
                                   const OperandType &operand_type) {
    operand_indexes_[name] = index;
    ordered_operands_.push_back(name);
    operand_types_.insert({name, operand_type});
}

ModelBuilder::Index ModelBuilder::AddInput(string name, const uint32_t batch,
                                           const uint32_t height,
                                           const uint32_t width,
                                           const uint32_t depth) {
    const vector<uint32_t> dimen{batch, height, width, depth};
    return AddInput(name, {Type::TENSOR_FLOAT32, dimen});
}

ModelBuilder::Index ModelBuilder::AddInput(std::string name,
                                           const OperandType &operand_type) {
    const auto index = AddNewOperand(operand_type);

    shaper_.AddShape(name, operand_type.dimensions);
    input_index_vec_.push_back(index);
    dnn_model_->AddInput(name, shaper_[name]);
    RegisterOperand(name, index, operand_type);
    return index;
}

// ModelBuilder auto generated methods start
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddConv(
    const std::string &input, const std::string &weight,
    const dnn::optional<std::string> &bias, int32_t padding_left,
    int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
    int32_t stride_x, int32_t stride_y, int32_t fuse_code,
    const std::string &output,
    const dnn::optional<QuantInfo> &output_quant_info) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    imm_blob_inputs_.insert(weight);
    const auto weight_idx = operand_indexes_.at(weight);
    input_indexes.push_back(weight_idx);
    uint32_t bias_idx_val;
    css bias_val = bias.value_or(weight + "_b");
    if (!bias.has_value()) {
        const auto weight_dimen = shaper_[weight];
        const Shape bias_dimen{weight_dimen[0]};
        const auto &weight_type = operand_types_.at(weight).type;
        if (weight_type == Type::TENSOR_FLOAT32) {
            bias_idx_val =
                FillOperand(bias_val, {Type::TENSOR_FLOAT32, bias_dimen}, 0.f);
        } else if (weight_type == Type::TENSOR_QUANT8_ASYMM) {
            const auto input_scale = operand_types_.at(input).operandType.scale;
            const auto weight_scale =
                operand_types_.at(weight).operandType.scale;
            bias_idx_val = FillOperand(
                bias_val,
                {Type::TENSOR_INT32, bias_dimen, input_scale * weight_scale},
                0);
        } else {
            throw std::invalid_argument("Unknown type " +
                                        typeToStr(weight_type));
        }
    } else {
        bias_idx_val = operand_indexes_.at(bias.value());
    }
    input_indexes.push_back(bias_idx_val);
    AddScalarOperands(input_indexes, padding_left, padding_right, padding_top,
                      padding_bottom, stride_x, stride_y, fuse_code);
    shaper_.Conv(input, weight, padding_left, padding_right, padding_top,
                 padding_bottom, stride_x, stride_y, output);
    const OperandType operand_type = GetOperandType(
        operand_types_.at(input).type, shaper_[output], output_quant_info);
    const auto output_idx =
        AddOperation(ANEURALNETWORKS_CONV_2D, input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddAvePool(
    const std::string &input, int32_t padding_left, int32_t padding_right,
    int32_t padding_top, int32_t padding_bottom, int32_t stride_x,
    int32_t stride_y, int32_t kernel_width, int32_t kernel_height,
    int32_t fuse_code, const std::string &output,
    const dnn::optional<QuantInfo> &output_quant_info) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    AddScalarOperands(input_indexes, padding_left, padding_right, padding_top,
                      padding_bottom, stride_x, stride_y, kernel_width,
                      kernel_height, fuse_code);
    shaper_.Pool(input, padding_left, padding_right, padding_top,
                 padding_bottom, stride_x, stride_y, kernel_width,
                 kernel_height, output);
    const OperandType operand_type = GetOperandType(
        operand_types_.at(input).type, shaper_[output], output_quant_info);
    const auto output_idx = AddOperation(ANEURALNETWORKS_AVERAGE_POOL_2D,
                                         input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddMaxPool(
    const std::string &input, int32_t padding_left, int32_t padding_right,
    int32_t padding_top, int32_t padding_bottom, int32_t stride_x,
    int32_t stride_y, int32_t kernel_width, int32_t kernel_height,
    int32_t fuse_code, const std::string &output,
    const dnn::optional<QuantInfo> &output_quant_info) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    AddScalarOperands(input_indexes, padding_left, padding_right, padding_top,
                      padding_bottom, stride_x, stride_y, kernel_width,
                      kernel_height, fuse_code);
    shaper_.Pool(input, padding_left, padding_right, padding_top,
                 padding_bottom, stride_x, stride_y, kernel_width,
                 kernel_height, output);
    const OperandType operand_type = GetOperandType(
        operand_types_.at(input).type, shaper_[output], output_quant_info);
    const auto output_idx = AddOperation(ANEURALNETWORKS_MAX_POOL_2D,
                                         input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddReLU(const std::string &input,
                                          const std::string &output) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    shaper_.Relu(input, output);
    const OperandType operand_type =
        GetOperandType(operand_types_.at(input).type, shaper_[output]);
    const auto output_idx =
        AddOperation(ANEURALNETWORKS_RELU, input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddSoftmax(const std::string &input,
                                             float beta,
                                             const std::string &output) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    AddScalarOperands(input_indexes, beta);
    shaper_.Softmax(input, output);
    const OperandType operand_type =
        GetOperandType(operand_types_.at(input).type, shaper_[output]);
    const auto output_idx =
        AddOperation(ANEURALNETWORKS_SOFTMAX, input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddFC(
    const std::string &input, const std::string &weight,
    const dnn::optional<std::string> &bias, int32_t fuse_code,
    const std::string &output,
    const dnn::optional<QuantInfo> &output_quant_info) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    imm_blob_inputs_.insert(weight);
    const auto weight_idx = operand_indexes_.at(weight);
    input_indexes.push_back(weight_idx);
    uint32_t bias_idx_val;
    css bias_val = bias.value_or(weight + "_b");
    if (!bias.has_value()) {
        const auto weight_dimen = shaper_[weight];
        const Shape bias_dimen{weight_dimen[0]};
        const auto &weight_type = operand_types_.at(weight).type;
        if (weight_type == Type::TENSOR_FLOAT32) {
            bias_idx_val =
                FillOperand(bias_val, {Type::TENSOR_FLOAT32, bias_dimen}, 0.f);
        } else if (weight_type == Type::TENSOR_QUANT8_ASYMM) {
            const auto input_scale = operand_types_.at(input).operandType.scale;
            const auto weight_scale =
                operand_types_.at(weight).operandType.scale;
            bias_idx_val = FillOperand(
                bias_val,
                {Type::TENSOR_INT32, bias_dimen, input_scale * weight_scale},
                0);
        } else {
            throw std::invalid_argument("Unknown type " +
                                        typeToStr(weight_type));
        }
    } else {
        bias_idx_val = operand_indexes_.at(bias.value());
    }
    input_indexes.push_back(bias_idx_val);
    AddScalarOperands(input_indexes, fuse_code);
    shaper_.FC(input, weight, output);
    const OperandType operand_type = GetOperandType(
        operand_types_.at(input).type, shaper_[output], output_quant_info);
    const auto output_idx = AddOperation(ANEURALNETWORKS_FULLY_CONNECTED,
                                         input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddAdd(
    const std::string &input1, const std::string &input2, int32_t fuse_code,
    const std::string &output,
    const dnn::optional<QuantInfo> &output_quant_info) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input1);
    const auto input1_idx = operand_indexes_.at(input1);
    input_indexes.push_back(input1_idx);
    imm_blob_inputs_.insert(input2);
    const auto input2_idx = operand_indexes_.at(input2);
    input_indexes.push_back(input2_idx);
    AddScalarOperands(input_indexes, fuse_code);
    shaper_.Eltwise(input1, input2, output);
    const OperandType operand_type = GetOperandType(
        operand_types_.at(input1).type, shaper_[output], output_quant_info);
    const auto output_idx =
        AddOperation(ANEURALNETWORKS_ADD, input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddConcat(
    const std::vector<std::string> &inputs, int32_t axis,
    const std::string &output) {
    IndexSeq input_indexes;
    for (const auto &x : inputs) {
        imm_blob_inputs_.insert(x);
        input_indexes.push_back(operand_indexes_.at(x));
    }
    AddScalarOperands(input_indexes, axis);
    shaper_.Concat(inputs, axis, output);
    const OperandType operand_type =
        GetOperandType(operand_types_.at(inputs[0]).type, shaper_[output]);
    const auto output_idx = AddOperation(ANEURALNETWORKS_CONCATENATION,
                                         input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddDepthwiseConv(
    const std::string &input, const std::string &weight,
    const dnn::optional<std::string> &bias, int32_t padding_left,
    int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
    int32_t stride_x, int32_t stride_y, int32_t depth_multiplier,
    int32_t fuse_code, const std::string &output,
    const dnn::optional<QuantInfo> &output_quant_info) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    imm_blob_inputs_.insert(weight);
    const auto weight_idx = operand_indexes_.at(weight);
    input_indexes.push_back(weight_idx);
    uint32_t bias_idx_val;
    css bias_val = bias.value_or(weight + "_b");
    if (!bias.has_value()) {
        const auto weight_dimen = shaper_[weight];
        const Shape bias_dimen{weight_dimen[0]};
        const auto &weight_type = operand_types_.at(weight).type;
        if (weight_type == Type::TENSOR_FLOAT32) {
            bias_idx_val =
                FillOperand(bias_val, {Type::TENSOR_FLOAT32, bias_dimen}, 0.f);
        } else if (weight_type == Type::TENSOR_QUANT8_ASYMM) {
            const auto input_scale = operand_types_.at(input).operandType.scale;
            const auto weight_scale =
                operand_types_.at(weight).operandType.scale;
            bias_idx_val = FillOperand(
                bias_val,
                {Type::TENSOR_INT32, bias_dimen, input_scale * weight_scale},
                0);
        } else {
            throw std::invalid_argument("Unknown type " +
                                        typeToStr(weight_type));
        }
    } else {
        bias_idx_val = operand_indexes_.at(bias.value());
    }
    input_indexes.push_back(bias_idx_val);
    AddScalarOperands(input_indexes, padding_left, padding_right, padding_top,
                      padding_bottom, stride_x, stride_y, depth_multiplier,
                      fuse_code);
    shaper_.DepthwiseConv(input, weight, padding_left, padding_right,
                          padding_top, padding_bottom, stride_x, stride_y,
                          output);
    const OperandType operand_type = GetOperandType(
        operand_types_.at(input).type, shaper_[output], output_quant_info);
    const auto output_idx = AddOperation(ANEURALNETWORKS_DEPTHWISE_CONV_2D,
                                         input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 28
ModelBuilder::Index ModelBuilder::AddBatchToSpaceND(
    const std::string &input, const std::vector<int32_t> &block_sizes,
    const std::string &output) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    const auto block_sizes_idx = AddTensorFromBuffer(
        "input_block_sizes_of_" + output, &block_sizes[0],
        {Type::TENSOR_INT32, Shape{static_cast<uint32_t>(block_sizes.size())}});
    input_indexes.push_back(block_sizes_idx);
    shaper_.BatchToSpace(input, block_sizes, output);
    const OperandType operand_type =
        GetOperandType(operand_types_.at(input).type, shaper_[output]);
    const auto output_idx = AddOperation(ANEURALNETWORKS_BATCH_TO_SPACE_ND,
                                         input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 28
#if __ANDROID_API__ >= 28
ModelBuilder::Index ModelBuilder::AddSpaceToBatchND(
    const std::string &input, const std::vector<int32_t> &block_sizes,
    const std::vector<int32_t> &pads, const std::string &output) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    const auto block_sizes_idx = AddTensorFromBuffer(
        "input_block_sizes_of_" + output, &block_sizes[0],
        {Type::TENSOR_INT32, Shape{static_cast<uint32_t>(block_sizes.size())}});
    input_indexes.push_back(block_sizes_idx);
    const auto pads_idx = AddTensorFromBuffer(
        "input_pads_of_" + output, &pads[0],
        {Type::TENSOR_INT32, Shape{static_cast<uint32_t>(pads.size())}});
    input_indexes.push_back(pads_idx);
    shaper_.SpaceToBatch(input, block_sizes, pads, output);
    const OperandType operand_type =
        GetOperandType(operand_types_.at(input).type, shaper_[output]);
    const auto output_idx = AddOperation(ANEURALNETWORKS_SPACE_TO_BATCH_ND,
                                         input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 28
#if __ANDROID_API__ >= 28
ModelBuilder::Index ModelBuilder::AddStridedSlice(
    const std::string &input, const std::vector<int32_t> &starts,
    const std::vector<int32_t> &ends, const std::vector<int32_t> &strides,
    int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask,
    const std::string &output) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    const auto starts_idx = AddTensorFromBuffer(
        "input_starts_of_" + output, &starts[0],
        {Type::TENSOR_INT32, Shape{static_cast<uint32_t>(starts.size())}});
    input_indexes.push_back(starts_idx);
    const auto ends_idx = AddTensorFromBuffer(
        "input_ends_of_" + output, &ends[0],
        {Type::TENSOR_INT32, Shape{static_cast<uint32_t>(ends.size())}});
    input_indexes.push_back(ends_idx);
    const auto strides_idx = AddTensorFromBuffer(
        "input_strides_of_" + output, &strides[0],
        {Type::TENSOR_INT32, Shape{static_cast<uint32_t>(strides.size())}});
    input_indexes.push_back(strides_idx);
    AddScalarOperands(input_indexes, begin_mask, end_mask, shrink_axis_mask);
    shaper_.StridedSlice(input, starts, ends, strides, begin_mask, end_mask,
                         shrink_axis_mask, output);
    const OperandType operand_type =
        GetOperandType(operand_types_.at(input).type, shaper_[output]);
    const auto output_idx = AddOperation(ANEURALNETWORKS_STRIDED_SLICE,
                                         input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 28
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddMul(
    const std::string &input1, const std::string &input2, int32_t fuse_code,
    const std::string &output,
    const dnn::optional<QuantInfo> &output_quant_info) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input1);
    const auto input1_idx = operand_indexes_.at(input1);
    input_indexes.push_back(input1_idx);
    imm_blob_inputs_.insert(input2);
    const auto input2_idx = operand_indexes_.at(input2);
    input_indexes.push_back(input2_idx);
    AddScalarOperands(input_indexes, fuse_code);
    shaper_.Eltwise(input1, input2, output);
    const OperandType operand_type = GetOperandType(
        operand_types_.at(input1).type, shaper_[output], output_quant_info);
    const auto output_idx =
        AddOperation(ANEURALNETWORKS_MUL, input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddAdd(const std::string &input, float scalar,
                                         int32_t fuse_code,
                                         const std::string &output) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    const auto scalar_idx = FillOperand("input_scalar_of_" + output,
                                        {Type::TENSOR_FLOAT32, {1}}, scalar);
    input_indexes.push_back(scalar_idx);
    AddScalarOperands(input_indexes, fuse_code);
    shaper_.Eltwise(input, output);
    const OperandType operand_type =
        GetOperandType(operand_types_.at(input).type, shaper_[output]);
    const auto output_idx =
        AddOperation(ANEURALNETWORKS_ADD, input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddMul(const std::string &input, float scalar,
                                         int32_t fuse_code,
                                         const std::string &output) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    const auto scalar_idx = FillOperand("input_scalar_of_" + output,
                                        {Type::TENSOR_FLOAT32, {1}}, scalar);
    input_indexes.push_back(scalar_idx);
    AddScalarOperands(input_indexes, fuse_code);
    shaper_.Eltwise(input, output);
    const OperandType operand_type =
        GetOperandType(operand_types_.at(input).type, shaper_[output]);
    const auto output_idx =
        AddOperation(ANEURALNETWORKS_MUL, input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddDequantize(const std::string &input,
                                                const std::string &output) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    shaper_.Identity(input, output);
    const OperandType operand_type =
        GetOperandType(Type::TENSOR_FLOAT32, shaper_[output]);
    const auto output_idx = AddOperation(ANEURALNETWORKS_DEQUANTIZE,
                                         input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
#if __ANDROID_API__ >= 27
ModelBuilder::Index ModelBuilder::AddLRN(const std::string &input,
                                         int32_t radius, float bias,
                                         float alpha, float beta,
                                         const std::string &output) {
    IndexSeq input_indexes;
    imm_blob_inputs_.insert(input);
    const auto input_idx = operand_indexes_.at(input);
    input_indexes.push_back(input_idx);
    AddScalarOperands(input_indexes, radius, bias, alpha, beta);
    shaper_.Identity(input, output);
    const OperandType operand_type =
        GetOperandType(operand_types_.at(input).type, shaper_[output]);
    const auto output_idx =
        AddOperation(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION,
                     input_indexes, operand_type)[0];
    RegisterOperand(output, output_idx, operand_type);
    imm_blob_outputs_.insert(output);
    return output_idx;
}
#endif  // __ANDROID_API__ >= 27
// ModelBuilder auto generated methods end

ModelBuilder::Index ModelBuilder::AddDepthWiseConv(
    const string &input_name, int32_t strideX, int32_t strideY,
    int32_t paddingLeft, int32_t paddingRight, int32_t paddingBottom,
    int32_t paddingTop, int32_t activation, int32_t depthMultiplier,
    const string &weight_name, const dnn::optional<string> &bias_name,
    const string &output_name,
    const dnn::optional<QuantInfo> &output_quant_info) {
    return AddDepthwiseConv(input_name, weight_name, bias_name, paddingLeft,
                            paddingRight, paddingTop, paddingBottom, strideX,
                            strideY, depthMultiplier, activation, output_name,
                            output_quant_info);
}

ModelBuilder::Index ModelBuilder::AddConv(
    const string &input_name, int32_t strideX, int32_t strideY,
    int32_t paddingLeft, int32_t paddingRight, int32_t paddingTop,
    int32_t paddingBottom, int32_t activation, const string &weight_name,
    const dnn::optional<string> &bias_name, const string &output_name,
    const dnn::optional<QuantInfo> &output_quant_info) {
    return AddConv(input_name, weight_name, bias_name, paddingLeft,
                   paddingRight, paddingTop, paddingBottom, strideX, strideY,
                   activation, output_name, output_quant_info);
}

ModelBuilder::Index ModelBuilder::AddPool(
    const string &input_name, int32_t strideX, int32_t strideY,
    int32_t paddingLeft, int32_t paddingRight, int32_t paddingTop,
    int32_t paddingBottom, int32_t height, int32_t width, int32_t activation,
    PoolingType poolingType, const string &output_name,
    const dnn::optional<QuantInfo> &output_quant_info) {
    if (height == -1 && width == -1) {
        VLOG(5) << "Global pool, input: " << input_name;
        const auto inputDimen = shaper_[input_name];
        height = inputDimen[1];
        width = inputDimen[2];
        strideX = width;
        strideY = height;
    }
    switch (poolingType) {
        case PoolingType::AVE_POOL:
            return AddAvePool(input_name, paddingLeft, paddingRight, paddingTop,
                              paddingBottom, strideX, strideY, width, height,
                              activation, output_name, output_quant_info);
            break;
        case PoolingType::MAX_POOL:
            return AddMaxPool(input_name, paddingLeft, paddingRight, paddingTop,
                              paddingBottom, strideX, strideY, width, height,
                              activation, output_name, output_quant_info);
            break;
    }
}

ModelBuilder::Index ModelBuilder::AddSoftMax(const string &input_name,
                                             float beta,
                                             const string &output_name) {
    return AddSoftmax(input_name, beta, output_name);
}

ModelBuilder::Index ModelBuilder::AddFC(
    const string &input_name, int32_t activation, const string &weight_name,
    const dnn::optional<string> &bias_name, const string &output_name,
    const dnn::optional<QuantInfo> &output_quant_info) {
    return AddFC(input_name, weight_name, bias_name, activation, output_name,
                 output_quant_info);
}

ModelBuilder::Index ModelBuilder::AddOperationAdd(const string &input_name,
                                                  float scalar,
                                                  string output_name) {
    return AddAdd(input_name, scalar, ANEURALNETWORKS_FUSED_NONE, output_name);
}

ModelBuilder::Index ModelBuilder::AddOperationAdd(
    const string &input1_name, const string &input2_name,
    const string &output_name,
    const dnn::optional<QuantInfo> &output_quant_info) {
    return AddAdd(input1_name, input2_name, ANEURALNETWORKS_FUSED_NONE,
                  output_name, output_quant_info);
}

ModelBuilder::Index ModelBuilder::AddMul(const string &input_name, float scalar,
                                         const string &output_name) {
    return AddMul(input_name, scalar, ANEURALNETWORKS_FUSED_NONE, output_name);
}

ModelBuilder::Index ModelBuilder::AddMul(
    const string &input1_name, const string &input2_name,
    const string &output_name,
    const dnn::optional<QuantInfo> &output_quant_info) {
    return AddMul(input1_name, input2_name, ANEURALNETWORKS_FUSED_NONE,
                  output_name, output_quant_info);
}
//--------------------------------------------------------------------------------------------------//

OperandType ModelBuilder::GetOperandType(const Type &type) {
    return {type, {}};
}

OperandType ModelBuilder::GetOperandType(
    const Type &type, const Shape &dims,
    const dnn::optional<QuantInfo> &quant_info) {
    if (quant_info.has_value()) {
        const auto &quant_info_val = quant_info.value();
        return GetOperandType(quant_info_val, dims);
    }
    return {type, dims};
}

OperandType ModelBuilder::GetOperandType(const QuantInfo &quant_info,
                                         const Shape &dims) {
    if (quant_info.type_ == Type::TENSOR_QUANT8_SYMM_PER_CHANNEL) {
        // FIXME: implement it
        throw std::invalid_argument("");
    } else {
        DNN_ASSERT(quant_info.scales_.size() == 1, "");
        return {quant_info.type_, dims, quant_info.scales_[0],
                quant_info.zero_point_.value_or(0)};
    }
}

#define DEFINE_OPERAND_FROM_SCALAR(scalar_type, map_type, op_type)           \
    ModelBuilder::Index ModelBuilder::OperandFromScalar(scalar_type value) { \
        if (map_type##_operand_map_.find(value) ==                           \
            map_type##_operand_map_.end()) {                                 \
            const auto index = AddNewOperand({Type::op_type});               \
            THROW_ON_ERROR_WITH_NOTE(                                        \
                nnapi_->ANeuralNetworksModel_setOperandValue(                \
                    dnn_model_->model_, index, &value, sizeof(value)),       \
                "value: " + std::to_string(value));                          \
            map_type##_operand_map_[value] = index;                          \
        }                                                                    \
        return map_type##_operand_map_[value];                               \
    }

DEFINE_OPERAND_FROM_SCALAR(uint32_t, uint32, UINT32);
DEFINE_OPERAND_FROM_SCALAR(int32_t, int32, INT32);
DEFINE_OPERAND_FROM_SCALAR(float, float32, FLOAT32);

#undef DEFINE_OPERAND_FROM_SCALAR

ModelBuilder::Index ModelBuilder::AddMissingOperand(
    const OperandType &operand_type) {
    const auto index = AddNewOperand(operand_type);
    THROW_ON_ERROR(nnapi_->ANeuralNetworksModel_setOperandValue(
        dnn_model_->model_, index, nullptr, 0));
    return index;
}

ModelBuilder::Index ModelBuilder::AddNewOperand(
    const OperandType &operand_type) {
    THROW_ON_ERROR(nnapi_->ANeuralNetworksModel_addOperand(
        dnn_model_->model_, &operand_type.operandType));
    return next_index_++;
}

// TODO: combine it and AddTensorFromBuffer
ModelBuilder::Index ModelBuilder::AddTensorFromMemory(const string &name,
                                                      const uint8_t *addr,
                                                      Shape dimen) {
    throw std::invalid_argument("");
    DNN_ASSERT(!dimen.empty(), "");
    const auto index = AddNewOperand({Type::TENSOR_FLOAT32, dimen});
    THROW_ON_ERROR(nnapi_->ANeuralNetworksModel_setOperandValueFromMemory(
        dnn_model_->model_, index, dnn_model_->memory_,
        addr - dnn_model_->data_, Product(dimen) * sizeof(float)));
    shaper_.AddShape(name, dimen);
    // RegisterOperand(name, index);
    return index;
}

ModelBuilder::Index ModelBuilder::AddTensorFromBuffer(
    const string &name, const void *buffer, const OperandType &operand_type) {
    DNN_ASSERT(!operand_type.dimensions.empty(), "");
    DNN_ASSERT(!isScalarType(operand_type.type), "");
    size_t element_size;
    switch (operand_type.type) {
        case Type::TENSOR_BOOL8:
            element_size = 1;
            break;
        case Type::TENSOR_FLOAT16:
            element_size = 2;
            break;
        case Type::TENSOR_FLOAT32:
            element_size = 4;
            break;
        case Type::TENSOR_INT32:
            element_size = 4;
            break;
        case Type::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            element_size = 1;
            break;
        case Type::TENSOR_QUANT8_ASYMM:
            element_size = 1;
            break;
        case Type::TENSOR_QUANT16_SYMM:
            element_size = 2;
            break;
        case Type::TENSOR_QUANT16_ASYMM:
            element_size = 2;
            break;
        default:
            throw std::invalid_argument("Wrong type: " +
                                        typeToStr(operand_type.type));
    }
    uint32_t index = AddNewOperand(operand_type);
    THROW_ON_ERROR(nnapi_->ANeuralNetworksModel_setOperandValue(
        dnn_model_->model_, index, buffer,
        Product(operand_type.dimensions) * element_size));
    shaper_.AddShape(name, operand_type.dimensions);
    RegisterOperand(name, index, operand_type);
    return index;
}

std::unique_ptr<Model> ModelBuilder::Compile(uint32_t preference) {
    if (output_index_vec_.empty()) {
        std::set<std::string> outputs;
        std::set_difference(imm_blob_outputs_.begin(), imm_blob_outputs_.end(),
                            imm_blob_inputs_.begin(), imm_blob_inputs_.end(),
                            std::inserter(outputs, outputs.end()));
        for (const auto &output : outputs) {
            VLOG(3) << "No blob is set explicitly as the output, automatically "
                       "set " +
                           output;
            AddOutput(output);
        }
    }
    THROW_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworksModel_identifyInputsAndOutputs(
            dnn_model_->model_, static_cast<uint32_t>(input_index_vec_.size()),
            &input_index_vec_[0],
            static_cast<uint32_t>(output_index_vec_.size()),
            &output_index_vec_[0]),
        "on identifyInputsAndOutputs");

    THROW_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworksModel_finish(dnn_model_->model_),
        "on model finish");

    ;
    THROW_ON_ERROR_WITH_NOTE(nnapi_->ANeuralNetworksCompilation_create(
                                 dnn_model_->model_, &dnn_model_->compilation_),
                             "on create");

    THROW_ON_ERROR_WITH_NOTE(nnapi_->ANeuralNetworksCompilation_setPreference(
                                 dnn_model_->compilation_, preference),
                             "on setPreference");

    THROW_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworksCompilation_finish(dnn_model_->compilation_),
        "on compilation finish");

    VLOG(5) << "Finishing.. Here are operands in the model:";
    for (const auto &name : ordered_operands_) {
        VLOG(5) << name << ": " << shaper_[name];
    }
    operand_indexes_.clear();
    ordered_operands_.clear();
    shaper_.Clear();
    return std::move(dnn_model_);
}

void ModelBuilder::RegisterBufferPointer(std::unique_ptr<uint8_t[]> &&pointer) {
    dnn_model_->uint8_buf_pointers_.push_back(std::move(pointer));
}

void ModelBuilder::RegisterBufferPointer(std::unique_ptr<float[]> &&pointer) {
    dnn_model_->float_buf_pointers_.push_back(std::move(pointer));
}

void ModelBuilder::RegisterBufferPointer(std::unique_ptr<int8_t[]> &&pointer) {
    dnn_model_->int8_buf_pointers_.push_back(std::move(pointer));
}

void ModelBuilder::RegisterBufferPointer(std::unique_ptr<int32_t[]> &&pointer) {
    dnn_model_->int32_buf_pointers_.push_back(std::move(pointer));
}

ModelBuilder::IndexSeq ModelBuilder::GetInputIndexes() {
    return input_index_vec_;
}

ModelBuilder::IndexSeq ModelBuilder::GetOutputIndexes() {
    return output_index_vec_;
}

ModelBuilder::Index ModelBuilder::GetBlobIndex(const string &blobName) {
    return operand_indexes_.at(blobName);
}

#define DEFINE_FILL_OPERAND(val_type, op_type)                            \
    ModelBuilder::Index ModelBuilder::FillOperand(                        \
        css &name, const OperandType &operand_type, const val_type val) { \
        DNN_ASSERT(operand_type.type == Type::TENSOR_##op_type, "");      \
        auto buf = std::unique_ptr<val_type[]>(                           \
            new val_type[Product(operand_type.dimensions)]);              \
        for (size_t i = 0; i < Product(operand_type.dimensions); i++) {   \
            buf[i] = val;                                                 \
        }                                                                 \
        auto idx = AddTensorFromBuffer(name, buf.get(), operand_type);    \
        RegisterBufferPointer(std::move(buf));                            \
        return idx;                                                       \
    }

DEFINE_FILL_OPERAND(float, FLOAT32);
DEFINE_FILL_OPERAND(int32_t, INT32);

#undef DEFINE_FILL_OPERAND

ModelBuilder::Shape ModelBuilder::GetBlobDim(const string &blobName) {
    return shaper_[blobName];
}

ModelBuilder::Shape ModelBuilder::GetBlobDim(uint32_t index) {
    for (const auto &p : operand_indexes_) {
        VLOG(5) << p.second;
        if (p.second == index) {
            return shaper_[p.first];
        }
    }
    throw std::invalid_argument("Wrong index in GetBlobDim");
}

template <typename... OperandTypes>
ModelBuilder::IndexSeq ModelBuilder::AddOperation(
    int op, IndexSeq input_indexes, OperandTypes... operand_types) {
    using android::nn::wrapper::OperandType;
    vector<OperandType> types;
    (types.push_back(operand_types), ...);
    IndexSeq output_indexes;
    for (const auto &type : types) {
        auto index = AddNewOperand(type);
        output_indexes.push_back(index);
    }

    THROW_ON_ERROR_WITH_NOTE(
        nnapi_->ANeuralNetworksModel_addOperation(
            dnn_model_->model_, op, input_indexes.size(), &input_indexes[0],
            output_indexes.size(), &output_indexes[0]),
        "op = " + std::to_string(op));

    return output_indexes;
}

void ModelBuilder::Prepare() {
    dnn_model_ = std::unique_ptr<Model>(new Model());
    const auto ret = nnapi_->ANeuralNetworksModel_create(&dnn_model_->model_);
    if (ret == ANEURALNETWORKS_OUT_OF_MEMORY) {
        throw std::bad_alloc();
    }
}

void ModelBuilder::SetMemory(int fd, size_t size, size_t offset) {
    ANeuralNetworksMemory *mem = nullptr;
    THROW_ON_ERROR(nnapi_->ANeuralNetworksMemory_createFromFd(
        size, PROT_READ, fd, offset, &mem));
    dnn_model_->memory_ = mem;
}

void ModelBuilder::SetBasePtr(uint8_t *data) {
    dnn_model_->data_ = data;
}

ModelBuilder &ModelBuilder::AddOutput(const std::string &name) {
    output_index_vec_.push_back(GetBlobIndex(name));
    dnn_model_->AddOutput(name, shaper_[name]);
    return *this;
}

ModelBuilder &ModelBuilder::AllowFp16(const bool allowed) {
    if (nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16 !=
        nullptr) {
        nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16(
            dnn_model_->model_, allowed);
    }
    return *this;
}

ModelBuilder::ModelBuilder() : nnapi_(NnApiImplementation()) {
}

dnn::optional<std::vector<Device>> ModelBuilder::getDevices() {
    if (nnapi_->android_sdk_version >= __ANDROID_API_Q__) {
        uint32_t device_count;
        THROW_ON_ERROR(nnapi_->ANeuralNetworks_getDeviceCount(&device_count));
        std::vector<Device> devices;
        FORZ(i, device_count) {
            ANeuralNetworksDevice *nn_device;
            nnapi_->ANeuralNetworks_getDevice(i, &nn_device);
            const char *nn_name_ptr;
            nnapi_->ANeuralNetworksDevice_getName(nn_device, &nn_name_ptr);
            const std::string device_name(nn_name_ptr);
            int64_t feature_level;
            nnapi_->ANeuralNetworksDevice_getFeatureLevel(nn_device, &feature_level);
            int type;
            nnapi_->ANeuralNetworksDevice_getType(nn_device, &type);
            const char *nn_version_ptr;
            nnapi_->ANeuralNetworksDevice_getVersion(nn_device, &nn_version_ptr);
            const std::string version(nn_version_ptr);
            Device device{device_name, feature_level, type, version};
            devices.push_back(device);
        }
        return devices;
    } else {
        return dnn::nullopt;
    }
}
}  // namespace dnn
