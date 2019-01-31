//
// Created by daquexian on 2017/11/8.
//
#include "ModelBuilder.h"

#include <array>
#include <ctime>
#include <tuple>
#include <sstream>
#include <sys/mman.h>
#include <fstream>
#include <iostream>

#include <glog/logging.h>
#include "android_log_helper.h"
#include <common/helper.h>
#include <operand_helper.h>

#define THROW_ON_ERROR(val) \
    if ((val) != ANEURALNETWORKS_NO_ERROR) {  \
        throw std::invalid_argument(std::string("Error in ") + __FILE__ + std::string(":") + \
                std::to_string(__LINE__) + std::string(", function name: ") + \
                std::string(__func__) + "error, ret: " + GetErrorCause(val));   \
    }

#define THROW_ON_ERROR_WITH_NOTE(val, note) \
    if ((val) != ANEURALNETWORKS_NO_ERROR) {  \
        throw std::invalid_argument(std::string("Error in ") + __FILE__ + std::string(":") + \
                std::to_string(__LINE__) + std::string(", function name: ") + \
                std::string(__func__) + "error, ret: " + GetErrorCause(val) + std::string(", ") + (note));   \
    }

using std::vector; using std::ifstream; using std::streamsize; using std::string; using std::ios;
using std::stringstream; using std::array;
using namespace android::nn::wrapper;

void ModelBuilder::AppendOperandIndex(const std::string &name, ModelBuilder::Index index) {
    operand_indexes_[name] = index;
    ordered_operands_.push_back(name);
}

ModelBuilder::Index ModelBuilder::AddInput(string name, uint32_t height, uint32_t width, uint32_t depth) {
    vector<uint32_t> dimen{1, width, height, depth};
    uint32_t index = AddNewOperand({Type::TENSOR_FLOAT32, dimen});

    shaper_.AddShape(name, dimen);
    input_index_vec_.push_back(index);
    dnn_model_->AddInput(name, shaper_[name]);
    AppendOperandIndex(name, index);
    return index;
}

ModelBuilder::Index ModelBuilder::AddDepthWiseConv(const string &input_name, int32_t strideX, int32_t strideY,
                                                   int32_t paddingLeft,
                                                   int32_t paddingRight, int32_t paddingBottom, int32_t paddingTop,
                                                   int32_t activation,
                                                   int32_t depthMultiplier, const string &weight_name,
                                                   const std::optional<string> &bias_name,
                                                   const string &output_name) {
    auto input = operand_indexes_[input_name];
    auto weight = operand_indexes_[weight_name];

    uint32_t biasIndexValue;
    if (!bias_name.has_value()) {
        Shape weightDimen = shaper_[weight_name];     // 1, height, width, num_output
        Shape bias_dims = Shape{weightDimen[3]};
        biasIndexValue = FillOperand(weight_name + "_b", {Type::TENSOR_FLOAT32, bias_dims}, 0.f);
    } else {
        biasIndexValue = operand_indexes_[bias_name.value()];
    }
    shaper_.DepthwiseConv(input_name, strideX, strideY, 1, 1, paddingLeft, paddingRight, paddingTop, paddingBottom, weight_name, output_name);
    IndexSeq input_indexes{input, weight, biasIndexValue};
    AddScalarOperands(input_indexes, paddingLeft, paddingRight, paddingTop, paddingBottom,
                strideX, strideY, depthMultiplier, activation);
    auto output_index = AddOperation(ANEURALNETWORKS_DEPTHWISE_CONV_2D, input_indexes, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index
ModelBuilder::AddConv(const string &input_name, int32_t strideX, int32_t strideY, int32_t paddingLeft,
                      int32_t paddingRight,
                      int32_t paddingTop, int32_t paddingBottom, int32_t activation, const string &weight_name,
                      const std::optional<string> &bias_name, const string &output_name) {
    auto input = operand_indexes_[input_name];
    auto weight = operand_indexes_[weight_name];

    uint32_t biasIndexValue;
    if (!bias_name.has_value()) {
        Shape weightDimen = shaper_[weight_name];     // num_output, height, width, num_input
        Shape bias_dims = Shape{weightDimen[0]};
        biasIndexValue = FillOperand(weight_name + "_b", {Type::TENSOR_FLOAT32, bias_dims}, 0.f);
    } else {
        biasIndexValue = operand_indexes_[bias_name.value()];
    }
    shaper_.Conv(input_name, strideX, strideY, 1, 1, paddingLeft, paddingRight, paddingTop, paddingBottom, weight_name, output_name);
    IndexSeq input_indexes{input, weight, biasIndexValue};
    AddScalarOperands(input_indexes, paddingLeft, paddingRight, paddingTop, paddingBottom, strideX, strideY, activation);
    auto output_index = AddOperation(ANEURALNETWORKS_CONV_2D, input_indexes, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

#if __ANDROID_API__ >= __ANDROID_API_P__

ModelBuilder::Index
ModelBuilder::AddStridedSlice(const string &input_name, const vector<int32_t> &starts, const vector<int32_t> &ends,
                              const vector<int32_t> &strides, int32_t beginMask, int32_t endMask,
                              int32_t shrinkAxisMask, const string &output_name) {

    auto input = operand_indexes_[input_name];

    uint32_t startsIndex = AddTensorFromBuffer(output_name + "_starts", &starts[0], vector<uint32_t>{static_cast<uint32_t>(starts.size())});
    uint32_t endsIndex = AddTensorFromBuffer(output_name + "_ends", &ends[0], vector<uint32_t>{static_cast<uint32_t>(ends.size())});
    uint32_t stridesIndex = AddTensorFromBuffer(output_name + "_strides", &strides[0],
                                                   vector<uint32_t>{static_cast<uint32_t>(strides.size())});

    shaper_.StridedSlice(input_name, starts, ends, strides, beginMask, endMask, shrinkAxisMask, output_name);
    IndexSeq input_indexes{input, startsIndex, endsIndex, stridesIndex};
    AddScalarOperands(input_indexes, beginMask, endMask, shrinkAxisMask);

    auto output_index = AddOperation(ANEURALNETWORKS_STRIDED_SLICE, input_indexes, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddSpaceToBatchND(const std::string &input_name, const std::vector<int32_t> &block_sizes,
        const std::vector<int32_t> &pads, const std::string &output_name) {
    auto input = operand_indexes_[input_name];

    auto block_sizes_idx = AddTensorFromBuffer(output_name + "_bs", &block_sizes[0], Shape{static_cast<uint32_t>(block_sizes.size())});
    auto pads_idx = AddTensorFromBuffer(output_name + "_pad", &pads[0], Shape{static_cast<uint32_t>(pads.size()) / 2, 2});

    shaper_.SpaceToBatch(input_name, block_sizes, pads, output_name);
    IndexSeq input_indexes{input, block_sizes_idx, pads_idx};
    auto output_index = AddOperation(ANEURALNETWORKS_SPACE_TO_BATCH_ND, input_indexes, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddBatchToSpaceND(const std::string &input_name, const std::vector<int32_t> &block_sizes,
        const std::string &output_name) {
    auto input = operand_indexes_[input_name];

    auto block_sizes_idx = AddTensorFromBuffer(output_name + "_bs", &block_sizes[0], Shape{static_cast<uint32_t>(block_sizes.size())});

    shaper_.BatchToSpace(input_name, block_sizes, output_name);
    IndexSeq input_indexes{input, block_sizes_idx};
    auto output_index = AddOperation(ANEURALNETWORKS_BATCH_TO_SPACE_ND, input_indexes, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

#endif

ModelBuilder::Index ModelBuilder::AddPool(const string &input_name, int32_t strideX, int32_t strideY,
                                          int32_t paddingLeft, int32_t paddingRight,
                                          int32_t paddingTop, int32_t paddingBottom, int32_t height, int32_t width,
                                          int32_t activation,
                                          uint32_t poolingType, const string &output_name) {
    auto input = operand_indexes_[input_name];

    if (height == -1 && width == -1) {
        VLOG(5) << "Global pool, input: " << input_name;
        auto inputDimen = shaper_[input_name];
        height = inputDimen[1];
        width = inputDimen[2];
        strideX = width;
        strideY = height;
    }
    shaper_.Pool(input_name, strideX, strideY, paddingLeft, paddingRight, paddingTop, paddingBottom,
            height, width, output_name);
    IndexSeq input_indexes{input};
    AddScalarOperands(input_indexes, 
            paddingLeft, paddingRight, paddingTop, paddingBottom, 
            strideX, strideY, width, height, activation);

    Index output_index;
    if (poolingType == MAX_POOL) {  // TODO: use strong typed enum here
        output_index = AddOperation(ANEURALNETWORKS_MAX_POOL_2D, input_indexes, shaper_[output_name])[0];
    } else if (poolingType == AVE_POOL) {
        output_index = AddOperation(ANEURALNETWORKS_AVERAGE_POOL_2D, input_indexes, shaper_[output_name])[0];
    }
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddSoftMax(const string &input_name, float beta, const string &output_name) {
    auto input = operand_indexes_[input_name];

    shaper_.Softmax(input_name, output_name);
    IndexSeq input_indexes{input};
    AddScalarOperands(input_indexes, beta);

    auto output_index = AddOperation(ANEURALNETWORKS_SOFTMAX, input_indexes, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddReLU(const string &input_name, const string &output_name) {
    auto input = operand_indexes_[input_name];

    shaper_.Relu(input_name, output_name);
    IndexSeq input_indexes{input};

    auto output_index = AddOperation(ANEURALNETWORKS_RELU, input_indexes, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddConcat(const vector<string> &input_names, int32_t axis, const string &output_name) {
    IndexSeq inputs;
    for (const auto &input_name : input_names) {
        inputs.push_back(operand_indexes_[input_name]);
    }

    shaper_.Concat(input_names, axis, output_name);
    IndexSeq input_indexes(inputs);
    AddScalarOperands(input_indexes, axis);

    auto output_index = AddOperation(ANEURALNETWORKS_CONCATENATION, input_indexes, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddLRN(const string &input_name, int32_t local_size, float bias, float alpha,
                                         float beta,
                                         const string &output_name) {
    auto input = operand_indexes_[input_name];

    shaper_.LRN(input_name, output_name);
    IndexSeq input_indexes{input};
    AddScalarOperands(input_indexes, local_size, bias, alpha, beta);

    auto output_idx = AddOperation(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION, input_indexes, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_idx);
    return output_idx;
}

ModelBuilder::Index ModelBuilder::AddFC(const string &input_name, int32_t activation,
                                        const string &weight_name, const std::optional<string> &bias_name,
                                        const string &output_name) {
    auto input = operand_indexes_[input_name];
    auto weight = operand_indexes_[weight_name];
    uint32_t biasIndexValue;
    if (!bias_name.has_value()) {
        auto weightDimen = shaper_[weight_name];
        Shape bias_dims = Shape{weightDimen[0]};
        biasIndexValue = FillOperand(weight_name + "_b", {Type::TENSOR_FLOAT32, bias_dims}, 0.f);
    } else {
        biasIndexValue = operand_indexes_[bias_name.value()];
    }
    shaper_.FC(input_name, weight_name, output_name);
    IndexSeq input_indexes{input, weight, biasIndexValue};
    AddScalarOperands(input_indexes, activation);
    auto output_idx = AddOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indexes, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_idx);
    return output_idx;
}

ModelBuilder::Index ModelBuilder::AddOperationAdd(const string &input_name, float scalar, string output_name) {
    auto input = operand_indexes_[input_name];
    Index scalarIndex = FillOperand(output_name + "_add", {Type::TENSOR_FLOAT32, {1}}, scalar);
    IndexSeq inputOperands{input, scalarIndex, OperandFromScalar(
            ModelBuilder::ACTIVATION_NONE)};
    shaper_.Eltwise(input_name, output_name);
    auto output_index = AddOperation(ANEURALNETWORKS_ADD, inputOperands, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddOperationAdd(const string &input1_name, const string &input2_name,
                                               const string &output_name) {
    auto input1 = operand_indexes_[input1_name];
    auto input2 = operand_indexes_[input2_name];
    shaper_.Eltwise(input1_name, input2_name, output_name);
    IndexSeq input_indexes{input1, input2};
    AddScalarOperands(input_indexes, ModelBuilder::ACTIVATION_NONE);
    auto output_idx = AddOperation(ANEURALNETWORKS_ADD, input_indexes, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_idx);
    return output_idx;
}

ModelBuilder::Index ModelBuilder::AddMul(const string &input_name, float scalar, const string &output_name) {
    auto input = operand_indexes_[input_name];
    Index scalarIndex = FillOperand(output_name + "_mul", {Type::TENSOR_FLOAT32, {1}}, scalar);
    IndexSeq inputOperands{input, scalarIndex, OperandFromScalar(
            ModelBuilder::ACTIVATION_NONE)};

    shaper_.Eltwise(input_name, output_name);
    auto output_index = AddOperation(ANEURALNETWORKS_MUL, inputOperands, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_index);
    return output_index;
}

ModelBuilder::Index ModelBuilder::AddMul(const string &input1_name, const string &input2_name,
                                               const string &output_name) {
    auto input1 = operand_indexes_[input1_name];
    auto input2 = operand_indexes_[input2_name];
    IndexSeq input_indexes{input1, input2};
    shaper_.Eltwise(input1_name, output_name);
    AddScalarOperands(input_indexes, ModelBuilder::ACTIVATION_NONE);
    auto output_idx = AddOperation(ANEURALNETWORKS_MUL, input_indexes, shaper_[output_name])[0];
    AppendOperandIndex(output_name, output_idx);
    return output_idx;
}
//--------------------------------------------------------------------------------------------------//

OperandType ModelBuilder::GetOperandType(const Type &type) {
    return {type, {}};
}

OperandType ModelBuilder::GetOperandType(const Type &type, const Shape &dims) {
    return {type, dims};
}

#define DEFINE_OPERAND_FROM_SCALAR(scalar_type, map_type, op_type)  \
ModelBuilder::Index ModelBuilder::OperandFromScalar(scalar_type value) {   \
    if (map_type##_operand_map_.find(value) == map_type##_operand_map_.end()) { \
        uint32_t index = AddNewOperand({Type::op_type}); \
        THROW_ON_ERROR_WITH_NOTE(ANeuralNetworksModel_setOperandValue(dnn_model_->model_, index, &value, sizeof(value)),    \
                "value: " + std::to_string(value)); \
        map_type##_operand_map_[value] = index; \
    }   \
    return map_type##_operand_map_[value];  \
}

DEFINE_OPERAND_FROM_SCALAR(uint32_t, uint32, UINT32);
DEFINE_OPERAND_FROM_SCALAR(int32_t, int32, INT32);
DEFINE_OPERAND_FROM_SCALAR(float, float32, FLOAT32);

#undef DEFINE_OPERAND_FROM_SCALAR

ModelBuilder::Index ModelBuilder::AddMissingOperand(const OperandType &operand_type) {
    const auto index = AddNewOperand(operand_type);
    THROW_ON_ERROR(ANeuralNetworksModel_setOperandValue(dnn_model_->model_, index, nullptr, 0));
    return index;
}

ModelBuilder::Index ModelBuilder::AddNewOperand(const OperandType &operand_type) {
    THROW_ON_ERROR(ANeuralNetworksModel_addOperand(dnn_model_->model_, &operand_type.operandType));
    return next_index_++;
}

ModelBuilder::Index ModelBuilder::AddTensorFromMemory(const string &name, const uint8_t *addr, Shape dimen) {
    DNN_ASSERT(!dimen.empty(), "");
    uint32_t index = AddNewOperand({Type::TENSOR_FLOAT32, dimen});
    THROW_ON_ERROR(ANeuralNetworksModel_setOperandValueFromMemory(
                dnn_model_->model_, index, dnn_model_->memory_, addr - dnn_model_->data_,
                Product(dimen) * sizeof(float)));
    shaper_.AddShape(name, dimen);
    AppendOperandIndex(name, index);
    return index;
}

ModelBuilder::Index ModelBuilder::AddTensorFromBuffer(const string &name, const float *buffer, Shape dimen) {
    DNN_ASSERT(!dimen.empty(), "");
    uint32_t index = AddNewOperand({Type::TENSOR_FLOAT32, dimen});
    THROW_ON_ERROR(ANeuralNetworksModel_setOperandValue(dnn_model_->model_, index, buffer, Product(dimen) * sizeof(float)));
    shaper_.AddShape(name, dimen);
    AppendOperandIndex(name, index);
    return index;
}

ModelBuilder::Index ModelBuilder::AddTensorFromBuffer(const string &name, const int32_t *buffer,
                                                      Shape dimen) {
    DNN_ASSERT(!dimen.empty(), "");
    uint32_t index = AddNewOperand({Type::TENSOR_INT32, dimen});
    THROW_ON_ERROR(ANeuralNetworksModel_setOperandValue(dnn_model_->model_, index, buffer, Product(dimen) * sizeof(int32_t)));
    shaper_.AddShape(name, dimen);
    AppendOperandIndex(name, index);
    return index;
}

std::unique_ptr<Model> ModelBuilder::Compile(uint32_t preference) {
    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksModel_identifyInputsAndOutputs(
                dnn_model_->model_,
                static_cast<uint32_t>(input_index_vec_.size()), &input_index_vec_[0],
                static_cast<uint32_t>(output_index_vec_.size()), &output_index_vec_[0]
                ), 
            "on identifyInputsAndOutputs");

    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksModel_finish(
                dnn_model_->model_
                ),
            "on model finish");

    ;
    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksCompilation_create(
                dnn_model_->model_, &dnn_model_->compilation_
                ),
            "on create");

    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksCompilation_setPreference(
                dnn_model_->compilation_, preference
                ),
            "on setPreference");

    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksCompilation_finish(
                dnn_model_->compilation_
                ),
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

ModelBuilder::IndexSeq ModelBuilder::GetInputIndexes() {
    return input_index_vec_;
}

ModelBuilder::IndexSeq ModelBuilder::GetOutputIndexes() {
    return output_index_vec_;
}

ModelBuilder::Index ModelBuilder::GetBlobIndex(const string &blobName) {
    return operand_indexes_.at(blobName);
}

#define DEFINE_FILL_OPERAND(val_type, op_type)  \
ModelBuilder::Index ModelBuilder::FillOperand(css &name, const OperandType &operand_type, const val_type val) {   \
    DNN_ASSERT(operand_type.type == Type::TENSOR_##op_type, "");  \
    auto buf = std::unique_ptr<val_type[]>(new val_type[Product(operand_type.dimensions)]);   \
    for (size_t i = 0; i < Product(operand_type.dimensions); i++) { \
        buf[i] = val;   \
    }   \
    auto idx = AddTensorFromBuffer(name, buf.get(), operand_type.dimensions);  \
    RegisterBufferPointer(std::move(buf));  \
    return idx; \
}

DEFINE_FILL_OPERAND(float, FLOAT32);

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

string ModelBuilder::GetErrorCause(int errorCode) {
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
ModelBuilder::IndexSeq ModelBuilder::AddOperation(int op, IndexSeq input_indexes, Shapes... shapes) {
    vector<Shape> shape_vec;
    (shape_vec.push_back(shapes), ...);
    IndexSeq output_indexes;
    for (auto shape : shape_vec) {
        auto index = AddNewOperand({Type::TENSOR_FLOAT32, shape});
        output_indexes.push_back(index);
    }

    THROW_ON_ERROR_WITH_NOTE(
            ANeuralNetworksModel_addOperation(
                dnn_model_->model_, op, input_indexes.size(), &input_indexes[0],
                output_indexes.size(), &output_indexes[0]),
            "op = " + std::to_string(op));
    
    return output_indexes;
}

void ModelBuilder::Prepare() {
    dnn_model_ = std::unique_ptr<Model>(new Model());
    auto ret = ANeuralNetworksModel_create(&dnn_model_->model_);
    if (ret == ANEURALNETWORKS_OUT_OF_MEMORY) {
        throw std::bad_alloc();
    }
}

void ModelBuilder::SetMemory(int fd, size_t size, size_t offset) {
    ANeuralNetworksMemory *mem = nullptr;
    THROW_ON_ERROR(ANeuralNetworksMemory_createFromFd(size, PROT_READ, fd, offset, &mem));
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

#if __ANDROID_API__ >= __ANDROID_API_P__
ModelBuilder &ModelBuilder::AllowFp16(const bool allowed) {
    ANeuralNetworksModel_relaxComputationFloat32toFloat16(dnn_model_->model_, allowed);
    return *this;
}
#endif
