//
// Created by daquexian on 8/13/18.
//

#include "DaqReader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <iostream>

#include <android_log_helper.h>
#include <flatbuffers_helper.h>
#include <glog/logging.h>

void ReadDaqImpl(const uint8_t *buf, ModelBuilder &builder);

std::string layer_type_to_str(DNN::LayerType type) {
    switch (type) {
        case DNN::LayerType::FC:
            return "fc";
        case DNN::LayerType::Add:
            return "Add";
        case DNN::LayerType::Relu:
            return "relu";
        case DNN::LayerType::Conv2D:
            return "conv";
        case DNN::LayerType::Concat:
            return "concat";
        case DNN::LayerType::MaxPool:
            return "maxpool";
        case DNN::LayerType::AvePool:
            return "avepool";
        case DNN::LayerType::Softmax:
            return "softmax";
        case DNN::LayerType::DepthwiseConv2D:
            return "depthwsie";
        case DNN::LayerType::BatchToSpace:
            return "batch2space";
        case DNN::LayerType::SpaceToBatch:
            return "space2batch";
        case DNN::LayerType::StridedSlice:
            return "stridedslice";
        case DNN::LayerType::Mul:
            return "mul";
        case DNN::LayerType::MulScalar:
            return "mulscalar";
        case DNN::LayerType::AddScalar:
            return "addscalar";
        case DNN::LayerType::Dequantize:
            return "dequantize";
    }
}

int convert_fuse_code_to_nnapi(const DNN::FuseCode fuse_code) {
    switch (fuse_code) {
        case DNN::FuseCode::None:
            return FuseCode::ANEURALNETWORKS_FUSED_NONE;
        case DNN::FuseCode::Relu:
            return FuseCode::ANEURALNETWORKS_FUSED_RELU;
        case DNN::FuseCode::Relu1:
            return FuseCode::ANEURALNETWORKS_FUSED_RELU1;
        case DNN::FuseCode::Relu6:
            return FuseCode::ANEURALNETWORKS_FUSED_RELU6;
    }
    throw std::invalid_argument("Invalid fuse_code");
}

const DNN::QuantInfo *GetQuantInfo(const DNN::Model &model, css &name) {
    if (model.quant_infos() == nullptr) {
        return nullptr;
    }
    FORZ(i, model.quant_infos()->size()) {
        const auto &quant_info = model.quant_infos()->Get(i);
        if (quant_info->name()->str() == name) {
            return quant_info;
            break;
        }
    }
    return nullptr;
}

std::optional<ModelBuilder::QuantInfo> DaqQuantInfoToModelBuilderQuantInfo(
    const DNN::QuantInfo *daq_quant_info) {
    if (daq_quant_info == nullptr) {
        return std::nullopt;
    }
    using android::nn::wrapper::Type;
    ModelBuilder::QuantInfo quant_info;
    std::map<DNN::DataType, Type> type_mapping = {
        {DNN::DataType::Float32, Type::TENSOR_FLOAT32},
        {DNN::DataType::Int32, Type::TENSOR_INT32},
        {DNN::DataType::QUANT8_ASYMM, Type::TENSOR_QUANT8_ASYMM},
        {DNN::DataType::QUANT8_SYMM_PER_CHANNEL,
         Type::TENSOR_QUANT8_SYMM_PER_CHANNEL}};
    quant_info.type_ = type_mapping[daq_quant_info->data_type()];
    quant_info.scales_ = unpack_fbs(daq_quant_info->scales());
    quant_info.zero_point_ = daq_quant_info->zero_point();

    return quant_info;
}

void AddInitializersFromBuffer(const DNN::Model &model, ModelBuilder &builder) {
    using namespace android::nn::wrapper;

    for (const auto &tensor : *model.initializers()) {
        ModelBuilder::Shape shape(tensor->shape()->begin(),
                                  tensor->shape()->end());
        if (tensor->data_type() == DNN::DataType::Float32) {
            builder.AddTensorFromBuffer(tensor->name()->str(),
                                        tensor->float32_data()->data(),
                                        {Type::TENSOR_FLOAT32, shape});
        } else if (tensor->data_type() == DNN::DataType::QUANT8_ASYMM) {
            const auto *quant_info = GetQuantInfo(model, tensor->name()->str());
            if (quant_info == nullptr) {
                throw std::invalid_argument("No quant info for " +
                                            tensor->name()->str());
            }
            float scale = quant_info->scales()->Get(0);
            int32_t zero_point = quant_info->zero_point();

            builder.AddTensorFromBuffer(
                tensor->name()->str(), tensor->int8_data()->data(),
                {Type::TENSOR_QUANT8_ASYMM, shape, scale, zero_point});
        } else if (tensor->data_type() == DNN::DataType::Int32) {
            const auto *quant_info = GetQuantInfo(model, tensor->name()->str());
            if (quant_info == nullptr) {
                throw std::invalid_argument("No quant info for " +
                                            tensor->name()->str());
            }
            float scale = quant_info->scales()->Get(0);

            builder.AddTensorFromBuffer(tensor->name()->str(),
                                        tensor->int32_data()->data(),
                                        {Type::TENSOR_INT32, shape, scale});
        } else {
            throw std::invalid_argument("Unknown data type");
        }
    }
}

// TODO: combine it and AddInitializersFromBuffer
void AddInitializersFromMmap(const DNN::Model &model, ModelBuilder &builder) {
    for (const auto &tensor : *model.initializers()) {
        if (tensor->data_type() == DNN::DataType::Float32) {
            ModelBuilder::Shape shape(tensor->shape()->begin(),
                                      tensor->shape()->end());
            builder.AddTensorFromMemory(tensor->name()->str(),
                                        tensor->float32_data()->Data(), shape);
        }
    }
}

void AddInputs(const DNN::Model &model, ModelBuilder &builder) {
    using namespace android::nn::wrapper;
    for (const auto &input : *model.inputs()) {
        css input_name = input->name()->str();
        ModelBuilder::Shape shape(input->shape()->begin(),
                                  input->shape()->end());
        const auto *daq_quant_info = GetQuantInfo(model, input_name);
        if (daq_quant_info != nullptr) {
            const auto quant_info =
                DaqQuantInfoToModelBuilderQuantInfo(daq_quant_info).value();
            DNN_ASSERT(quant_info.type_ == Type::TENSOR_QUANT8_ASYMM, "");
            OperandType operand_type(quant_info.type_, shape,
                                     quant_info.scales_[0],
                                     quant_info.zero_point_.value_or(0));
            builder.AddInput(input_name, operand_type);
        } else {
            builder.AddInput(input_name, shape[1], shape[2], shape[3]);
        }
    }
}

void AddLayers(const DNN::Model &model, ModelBuilder &builder) {
    for (const auto layer : *model.layers()) {
        switch (layer->type()) {
            case DNN::LayerType::Conv2D: {
                UNPACK_LAYER_QUANT(conv2d, strides, pads, fuse, input, weight,
                                   bias, output);
                builder.AddConv(
                    input, strides[1], strides[0], pads[2], pads[3], pads[0],
                    pads[1], fuse, weight,
                    (bias != "" ? std::make_optional(bias) : std::nullopt),
                    output, quant_info);
                break;
            }
            case DNN::LayerType::DepthwiseConv2D: {
                UNPACK_LAYER_QUANT(depthwise_conv2d, strides, pads, multiplier,
                                   fuse, input, weight, bias, output);
                builder.AddDepthWiseConv(
                    input, strides[1], strides[0], pads[2], pads[3], pads[1],
                    pads[0], fuse, multiplier, weight,
                    (bias != "" ? std::make_optional(bias) : std::nullopt),
                    output, quant_info);
                break;
            }
            case DNN::LayerType::AvePool: {
                UNPACK_LAYER_QUANT(avepool, strides, pads, kernel_shape, fuse,
                                   input, output);
                builder.AddPool(
                    input, strides[1], strides[0], pads[2], pads[3], pads[0],
                    pads[1], kernel_shape[0], kernel_shape[1], fuse,
                    ModelBuilder::PoolingType::AVE_POOL, output, quant_info);
                break;
            }
            case DNN::LayerType::MaxPool: {
                UNPACK_LAYER_QUANT(maxpool, strides, pads, kernel_shape, fuse,
                                   input, output);
                builder.AddPool(
                    input, strides[1], strides[0], pads[2], pads[3], pads[0],
                    pads[1], kernel_shape[0], kernel_shape[1], fuse,
                    ModelBuilder::PoolingType::MAX_POOL, output, quant_info);
                break;
            }
            case DNN::LayerType::Relu: {
                ADD_LAYER(relu, ReLU, input, output);
                break;
            }
            case DNN::LayerType::Add: {
                ADD_LAYER_QUANT(add, OperationAdd, input1, input2, output);
                break;
            }
            case DNN::LayerType::AddScalar: {
                ADD_LAYER(add, OperationAdd, input1, input2, output);
                break;
            }
            case DNN::LayerType::Mul: {
                ADD_LAYER_QUANT(mul, Mul, input1, input2, output);
                break;
            }
            case DNN::LayerType::MulScalar: {
                ADD_LAYER(mul, Mul, input1, input2, output);
                break;
            }
            case DNN::LayerType::FC: {
                UNPACK_LAYER_QUANT(fc, input, weight, bias, fuse, output);
                builder.AddFC(
                    input, fuse, weight,
                    (bias != "" ? std::make_optional(bias) : std::nullopt),
                    output, quant_info);
                break;
            }
            case DNN::LayerType::Softmax: {
                UNPACK_LAYER(softmax, input, output);
                builder.AddSoftMax(input, 1.f, output);
                break;
            }
            case DNN::LayerType::Concat: {
                ADD_LAYER(concat, Concat, inputs, axis, output)
                break;
            }
            case DNN::LayerType::Dequantize: {
                ADD_LAYER(dequantize, Dequantize, input, output);
                break;
            }
            case DNN::LayerType::BatchToSpace: {
#if __ANDROID_API__ >= __ANDROID_API_P__
                ADD_LAYER(batch_to_space, BatchToSpaceND, input, block_sizes,
                          output);
                break;
#endif
            }
            case DNN::LayerType::SpaceToBatch: {
#if __ANDROID_API__ >= __ANDROID_API_P__
                ADD_LAYER(space_to_batch, SpaceToBatchND, input, block_sizes,
                          pads, output);
                break;
#endif
            }
            case DNN::LayerType::StridedSlice: {
#if __ANDROID_API__ >= __ANDROID_API_P__
                ADD_LAYER(strided_slice, StridedSlice, input, starts, ends,
                          strides, begin_mask, end_mask, shrink_axis_mask,
                          output);
#else
                throw std::invalid_argument("Unsupported layer " +
                                            layer_type_to_str(layer->type()) +
                                            " in API 28");
#endif
                break;
            }
            default: {
                throw std::invalid_argument("Unsupported layer " +
                                            layer_type_to_str(layer->type()));
            }
        }
    }
}

/**
 * It is designed to read a regular file. For reading file in assets folder of
 * Android app, read the content into a char array and call readFromBuffer
 *
 * It will throw an exception when opening file failed
 *
 * @param filepath , like "/data/local/tmp/squeezenet.daq"
 * @param builder a ModelBuilder object
 */
void DaqReader::ReadDaq(const std::string &filepath, ModelBuilder &builder,
                        const bool use_mmap) {
    if (use_mmap) {
        const auto fd = open(filepath.c_str(), O_RDONLY);
        ReadDaq(fd, builder);
    } else {
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::unique_ptr<uint8_t[]> buf(new uint8_t[size]);
        if (!file.read(reinterpret_cast<char *>(buf.get()), size)) {
            throw std::invalid_argument("Read file error");
        }
        ReadDaq(std::move(buf), builder);
    }
}

void DaqReader::ReadDaq(const int &fd, ModelBuilder &builder,
                        const off_t offset, size_t fsize) {
    if (fd == -1) {
        throw std::invalid_argument("Open file error " + std::to_string(errno));
    }
    if (fsize == 0) {
        fsize = static_cast<size_t>(lseek(fd, offset, SEEK_END));
    }
    auto data = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, offset);
    if (data == MAP_FAILED) {
        throw std::invalid_argument("mmap failed, errno = " +
                                    std::to_string(errno));
    }
    builder.SetMemory(fd, fsize, offset);
    builder.SetBasePtr(static_cast<unsigned char *>(data));
    auto ret = close(fd);
    if (ret == -1) {
        throw std::runtime_error("close file error, errno = " +
                                 std::to_string(errno));
    }
    VLOG(4) << "Read daq from mmap";
    ReadDaqImpl(static_cast<const uint8_t *>(data), builder);
}

void DaqReader::ReadDaq(std::unique_ptr<uint8_t[]> buf, ModelBuilder &builder) {
    ReadDaq(buf.get(), builder);
    builder.RegisterBufferPointer(std::move(buf));
}

void DaqReader::ReadDaq(const uint8_t *buf, ModelBuilder &builder) {
    VLOG(4) << "Read daq from buffer";
    ReadDaqImpl(buf, builder);
}

void ReadDaqImpl(const uint8_t *buf, ModelBuilder &builder) {
    builder.Prepare();  // a daq file should be a full model, so prepare here
    auto model = DNN::GetModel(buf);
    AddInitializersFromBuffer(*model, builder);
    AddInputs(*model, builder);
    AddLayers(*model, builder);
}
