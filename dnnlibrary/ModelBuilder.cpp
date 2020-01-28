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

#include <common/data_types.h>
#include <common/helper.h>
#include <dnnlibrary/android_log_helper.h>
#include <dnnlibrary/nnapi_helper.h>
#include <glog/logging.h>

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

DEFINE_OPERAND_FROM_SCALAR(bool, bool, BOOL);
DEFINE_OPERAND_FROM_SCALAR(uint32_t, uint32, UINT32);
DEFINE_OPERAND_FROM_SCALAR(int32_t, int32, INT32);
DEFINE_OPERAND_FROM_SCALAR(float, float32, FLOAT32);

#undef DEFINE_OPERAND_FROM_SCALAR

template <typename E>
constexpr typename std::underlying_type<E>::type to_underlying(E e) noexcept {
    return static_cast<typename std::underlying_type<E>::type>(e);
}

ModelBuilder::Index ModelBuilder::OperandFromScalar(dnn::FuseCode value) {
    return OperandFromScalar(to_underlying(value));
}


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

size_t GetBytesNumFromOperandType(const OperandType &operand_type) {
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
    return Product(operand_type.dimensions) * element_size;
}

/**
 * @brief Add NNAPI operand from `buffer`, the memory pointed
 * by `buffer` should be persistent until the execution finished.
 * No copying.
 *
 * @param name The name of operand
 * @param buffer The address of the buffer
 * @param operand_type The OperandType of the operand
 *
 * @return The index of the added operand
 */
ModelBuilder::Index ModelBuilder::AddTensorFromPersistentBuffer(
    const string &name, const void *buffer, const OperandType &operand_type) {
    DNN_ASSERT(!operand_type.dimensions.empty(), "");
    DNN_ASSERT(!isScalarType(operand_type.type), "");
    uint32_t index = AddNewOperand(operand_type);
    THROW_ON_ERROR(nnapi_->ANeuralNetworksModel_setOperandValue(
        dnn_model_->model_, index, buffer, GetBytesNumFromOperandType(operand_type)
        ));
    shaper_.AddShape(name, operand_type.dimensions);
    RegisterOperand(name, index, operand_type);
    return index;
}

/**
 * @brief It is the same as `AddTensorFromPersistentBuffer` except
 * the memory pointed by `buffer` will be copied so that `buffer`
 * does not need to be persistent
 *
 * @param name
 * @param buffer
 * @param operand_type
 *
 * @return
 */
ModelBuilder::Index ModelBuilder::AddTensorFromBuffer(
    const string &name, const void *buffer, const OperandType &operand_type) {
    const auto bytes = GetBytesNumFromOperandType(operand_type);
    auto persistent_buf = std::unique_ptr<uint8_t[]>(
        new uint8_t[bytes]);
    memmove(persistent_buf.get(), buffer, bytes);

    auto idx =
        AddTensorFromPersistentBuffer(name, persistent_buf.get(), operand_type);
    RegisterBufferPointer(std::move(persistent_buf));

    return idx;
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
        auto idx =                                                        \
            AddTensorFromPersistentBuffer(name, buf.get(), operand_type); \
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

dnn::optional<std::vector<Device>> ModelBuilder::GetDevices() {
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
            nnapi_->ANeuralNetworksDevice_getFeatureLevel(nn_device,
                                                          &feature_level);
            int type;
            nnapi_->ANeuralNetworksDevice_getType(nn_device, &type);
            const char *nn_version_ptr;
            nnapi_->ANeuralNetworksDevice_getVersion(nn_device,
                                                     &nn_version_ptr);
            const std::string version(nn_version_ptr);
            Device device{device_name, feature_level, type, version};
            devices.push_back(device);
        }
        return devices;
    } else {
        return dnn::nullopt;
    }
}


int32_t ModelBuilder::android_api_level() const {
    return nnapi_->android_sdk_version;
}
}  // namespace dnn
