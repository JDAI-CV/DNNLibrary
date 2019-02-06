//
// Created by daquexian on 8/13/18.
//

#include "DaqReader.h"

#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <unistd.h>

#include <glog/logging.h>
#include <android_log_helper.h>
#include <flatbuffers_helper.h>

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
    FORZ(i, model.quant_infos()->size()) {
        const auto &quant_info = model.quant_infos()->Get(i);
        if (quant_info->name()->str() == name) {
            return quant_info;
            break;
        }
    }
    return nullptr;
}

std::optional<ModelBuilder::QuantInfo> DaqQuantInfoToModelBuilderQuantInfo(const DNN::QuantInfo *daq_quant_info) {
    if (daq_quant_info == nullptr) {
        return std::nullopt;
    }
    using android::nn::wrapper::Type;
    ModelBuilder::QuantInfo quant_info;
    std::map<DNN::DataType, Type> type_mapping =    
    {
        {DNN::DataType::Float32, Type::TENSOR_FLOAT32},
        {DNN::DataType::Int32, Type::TENSOR_INT32},
        {DNN::DataType::QUANT8_ASYMM, Type::TENSOR_QUANT8_ASYMM},
        {DNN::DataType::QUANT8_SYMM_PER_CHANNEL, Type::TENSOR_QUANT8_SYMM_PER_CHANNEL}
    };
    quant_info.type_ = type_mapping[daq_quant_info->data_type()];
    quant_info.scales_ = unpack_fbs(daq_quant_info->scales());
    quant_info.zero_point_ = daq_quant_info->zero_point();
    
    return quant_info;
}


void AddInitializersFromBuffer(const DNN::Model &model, ModelBuilder &builder) {
    using namespace android::nn::wrapper;

    for (const auto &tensor : *model.initializers()) {
        ModelBuilder::Shape shape(tensor->shape()->begin(), tensor->shape()->end());
        if (tensor->data_type() == DNN::DataType::Float32) {
            builder.AddTensorFromBuffer(tensor->name()->str(),
                                        tensor->float32_data()->data(),
                                        {Type::TENSOR_FLOAT32, shape});
        } else if (tensor->data_type() == DNN::DataType::QUANT8_ASYMM) {
            const auto *quant_info = GetQuantInfo(model, tensor->name()->str());
            if (quant_info == nullptr) {
                throw std::invalid_argument("No quant info for " + tensor->name()->str());
            }
            float scale = quant_info->scales()->Get(0);
            int32_t zero_point = quant_info->zero_point();

            builder.AddTensorFromBuffer(tensor->name()->str(),
                                        tensor->int8_data()->data(),
                                        {Type::TENSOR_QUANT8_ASYMM, shape, scale, zero_point});
        } else if (tensor->data_type() == DNN::DataType::Int32) {
            const auto *quant_info = GetQuantInfo(model, tensor->name()->str());
            if (quant_info == nullptr) {
                FORZ(i, model.quant_infos()->size()) {
                    LOG(INFO) << "quant info of: " << model.quant_infos()->Get(i)->name()->c_str();
                }
                throw std::invalid_argument("No quant info for " + tensor->name()->str());
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
            ModelBuilder::Shape shape(tensor->shape()->begin(), tensor->shape()->end());
            builder.AddTensorFromMemory(tensor->name()->str(),
                                        tensor->float32_data()->Data(),
                                        shape);
        }
    }
}

void AddInputs(const DNN::Model &model, ModelBuilder &builder) {
    using namespace android::nn::wrapper;
    for (const auto &input : *model.inputs()) {
        css input_name = input->name()->str();
        ModelBuilder::Shape shape(input->shape()->begin(), input->shape()->end());
        const auto *daq_quant_info = GetQuantInfo(model, input_name);
        if (daq_quant_info != nullptr) {
            const auto quant_info = DaqQuantInfoToModelBuilderQuantInfo(daq_quant_info).value();
            DNN_ASSERT(quant_info.type_ == Type::TENSOR_QUANT8_ASYMM, "");
            OperandType operand_type(quant_info.type_, shape, quant_info.scales_[0], quant_info.zero_point_.value_or(0));
            builder.AddInput(input_name, operand_type);
        } else {
            builder.AddInput(input_name, shape[1], shape[2], shape[3]);
        }
    }

}

void AddLayers(const DNN::Model &model, ModelBuilder &builder) {
    for (auto layer : *model.layers()) {
        switch (layer->type()) {
            case DNN::LayerType::Conv2D: {
                auto param = layer->conv2d_param();
                auto strides = param->strides();
                auto pads = param->pads();
                auto fuse = param->fuse();
                auto input_name = param->input()->str();
                auto weight_name = param->weight()->str();
                auto bias = param->bias();
                auto output_name = param->output()->str();
                const auto *daq_quant_info = GetQuantInfo(model, output_name);
                const auto quant_info = DaqQuantInfoToModelBuilderQuantInfo(daq_quant_info);
                VLOG(5) << "Conv, input: " << input_name << ", weight: " << weight_name << ", output: " << output_name;
                builder.AddConv(input_name, strides->Get(1), strides->Get(0),
                                pads->Get(2), pads->Get(3), pads->Get(0), pads->Get(1),
                                convert_fuse_code_to_nnapi(fuse), weight_name,
                                (bias ? std::make_optional(bias->str()) : std::nullopt),
                                output_name, quant_info);
                break;
            }
            case DNN::LayerType::DepthwiseConv2D: {
                auto param = layer->depthwise_conv2d_param();
                auto strides = param->strides();
                auto pads = param->pads();
                auto multiplier = param->multiplier();
                auto fuse = param->fuse();
                auto input_name = param->input()->str();
                auto weight_name = param->weight()->str();
                auto bias = param->bias();
                auto output_name = param->output()->str();
                const auto *daq_quant_info = GetQuantInfo(model, output_name);
                const auto quant_info = DaqQuantInfoToModelBuilderQuantInfo(daq_quant_info);
                VLOG(5) << "Depthwise Conv, input: " << input_name << ", weight: " << weight_name << ", output: " << output_name;
                builder.AddDepthWiseConv(input_name, strides->Get(1), strides->Get(0),
                                         pads->Get(2), pads->Get(3), pads->Get(1), pads->Get(0),
                                         convert_fuse_code_to_nnapi(fuse), multiplier,
                                         weight_name,
                                         (bias ? std::make_optional(bias->str()) : std::nullopt),
                                         output_name, quant_info);
                break;
            }
            case DNN::LayerType::AvePool: {
                auto param = layer->avepool_param();
                auto strides = param->strides();
                auto pads = param->pads();
                auto kernel_shape = param->kernel_shape();
                auto fuse = param->fuse();
                auto input_name = param->input()->str();
                auto output_name = param->output()->str();
                const auto *daq_quant_info = GetQuantInfo(model, output_name);
                const auto quant_info = DaqQuantInfoToModelBuilderQuantInfo(daq_quant_info);
                VLOG(5) << "Average pool, input: " << input_name << ", output: " << output_name;
                builder.AddPool(input_name, strides->Get(1), strides->Get(0),
                                pads->Get(2), pads->Get(3), pads->Get(0), pads->Get(1),
                                kernel_shape->Get(0), kernel_shape->Get(1),
                                convert_fuse_code_to_nnapi(fuse),
                                ModelBuilder::PoolingType::AVE_POOL, output_name,
                                quant_info);
                break;
            }
            case DNN::LayerType::MaxPool: {
                auto param = layer->maxpool_param();
                auto strides = param->strides();
                auto pads = param->pads();
                auto kernel_shape = param->kernel_shape();
                auto fuse = param->fuse();
                auto input_name = param->input()->str();
                auto output_name = param->output()->str();
                const auto *daq_quant_info = GetQuantInfo(model, output_name);
                const auto quant_info = DaqQuantInfoToModelBuilderQuantInfo(daq_quant_info);
                VLOG(5) << "Max pool, input: " << input_name << ", output: " << output_name;
                builder.AddPool(input_name, strides->Get(1), strides->Get(0),
                                pads->Get(2), pads->Get(3), pads->Get(0), pads->Get(1),
                                kernel_shape->Get(0), kernel_shape->Get(1),
                                convert_fuse_code_to_nnapi(fuse),
                                ModelBuilder::PoolingType::MAX_POOL, output_name,
                                quant_info);
                break;
            }
            case DNN::LayerType::Relu: {
                auto param = layer->relu_param();
                auto input_name = param->input()->str();
                auto output_name = param->output()->str();
                VLOG(5) << "Relu, input " << input_name << ", output: " << output_name;
                builder.AddReLU(input_name, output_name);
                break;
            }
            case DNN::LayerType::Add: {
                auto param = layer->add_param();
                auto input1_name = param->input1()->str();
                auto input2_name = param->input2()->str();
                auto output_name = param->output()->str();
                const auto *daq_quant_info = GetQuantInfo(model, output_name);
                const auto quant_info = DaqQuantInfoToModelBuilderQuantInfo(daq_quant_info);
                VLOG(5) << "Add, input1 " << input1_name << ", input2 " << input2_name << ", output: " << output_name;
                builder.AddOperationAdd(input1_name, input2_name, output_name, quant_info);
                break;
            }
            case DNN::LayerType::AddScalar: {
                auto param = layer->add_scalar_param();
                auto input1_name = param->input1()->str();
                auto input2 = param->input2();
                auto output_name = param->output()->str();
                VLOG(5) << "Add, input1 " << input1_name << ", input2 " << input2 << ", output: " << output_name;
                builder.AddOperationAdd(input1_name, input2, output_name);
                break;
            }
            case DNN::LayerType::Mul: {
                auto param = layer->mul_param();
                auto input1_name = param->input1()->str();
                auto input2_name = param->input2()->str();
                auto output_name = param->output()->str();
                const auto *daq_quant_info = GetQuantInfo(model, output_name);
                const auto quant_info = DaqQuantInfoToModelBuilderQuantInfo(daq_quant_info);
                VLOG(5) << "Mul, input1 " << input1_name << ", input2 " << input2_name << ", output: " << output_name;
                builder.AddMul(input1_name, input2_name, output_name, quant_info);
                break;
            }
            case DNN::LayerType::MulScalar: {
                auto param = layer->mul_scalar_param();
                auto input1_name = param->input1()->str();
                auto input2 = param->input2();
                auto output_name = param->output()->str();
                VLOG(5) << "Mul, input1 " << input1_name << ", input2 " << input2 << ", output: " << output_name;
                builder.AddMul(input1_name, input2, output_name);
                break;
            }
            case DNN::LayerType::FC: {
                auto param = layer->fc_param();
                auto fuse = param->fuse();
                auto weight_name = param->weight()->str();
                auto bias_name = param->bias()->str();
                auto input_name = param->input()->str();
                auto output_name = param->output()->str();
                const auto *daq_quant_info = GetQuantInfo(model, output_name);
                const auto quant_info = DaqQuantInfoToModelBuilderQuantInfo(daq_quant_info);
                VLOG(5) << "FC, input " << input_name << ", output: " << output_name;
                builder.AddFC(input_name, convert_fuse_code_to_nnapi(fuse),
                              weight_name, bias_name, output_name, quant_info);
                break;
            }
            case DNN::LayerType::Softmax: {
                auto param = layer->softmax_param();
                auto input_name = param->input()->str();
                auto output_name = param->output()->str();
                VLOG(5) << "Softmax, input " << input_name << ", output: " << output_name;
                builder.AddSoftMax(input_name, 1.f, output_name);
                break;
            }
            case DNN::LayerType::Concat: {
                auto param = layer->concat_param();
                auto axis = param->axis();
                auto inputs = param->inputs();
                auto output_name = param->output()->str();
                std::vector<std::string> input_names;
                for (size_t i = 0; i < inputs->size(); i++) {
                    input_names.push_back(inputs->Get(static_cast<flatbuffers::uoffset_t>(i))->str());
                }
                VLOG(5) << "Concat, input " << input_names << ", output: " << output_name;
                builder.AddConcat(input_names, axis, output_name);
                break;
            }
            case DNN::LayerType::BatchToSpace: {
#if __ANDROID_API__ >= __ANDROID_API_P__
                auto param = layer->batch_to_space_param();
                auto input_name = param->input()->str();
                auto block_sizes_fbs = param->block_sizes();
                auto output_name = param->output()->str();
                std::vector<int> block_sizes;
                for (size_t i = 0; i < block_sizes_fbs->size(); i++) {
                    block_sizes.push_back(block_sizes_fbs->Get(static_cast<flatbuffers::uoffset_t>(i)));
                }
                VLOG(5) << "BatchToSpaceND, input " << input_name
                    << ", block sizes " << block_sizes << ", output: " << output_name;
                builder.AddBatchToSpaceND(input_name, block_sizes, output_name);
                break;
#endif
            }
            case DNN::LayerType::SpaceToBatch: {
#if __ANDROID_API__ >= __ANDROID_API_P__
                auto param = layer->space_to_batch_param();
                auto input_name = param->input()->str();
                auto block_sizes_fbs = param->block_sizes();
                auto pads_fbs = param->pads();
                auto output_name = param->output()->str();
                std::vector<int> block_sizes = unpack_fbs(block_sizes_fbs);
                std::vector<int> pads = unpack_fbs(pads_fbs);
                VLOG(5) << "SpaceToBatchND, input " << input_name
                    << ", block sizes " << block_sizes << ", pads " << pads << "output: " << output_name;
                builder.AddSpaceToBatchND(input_name, block_sizes, pads, output_name);
                break;
#endif
            }
            case DNN::LayerType::StridedSlice: {
#if __ANDROID_API__ >= __ANDROID_API_P__
                auto param = layer->strided_slice_param();
                auto input_name = param->input()->str();
                auto starts = unpack_fbs(param->starts());
                auto ends = unpack_fbs(param->ends());
                auto strides = unpack_fbs(param->strides());
                int32_t begin_mask = param->begin_mask();
                int32_t end_mask = param->end_mask();
                int32_t shrink_axis_mask = param->shrink_axis_mask();
                auto output_name = param->output()->str();
                VLOG(5) << "StridedSlice, input " << input_name
                    << ", starts " << starts << ", ends " << ends << ", strides " << strides
                    << ", begin_mask " << begin_mask << ", end_mask " << end_mask
                    << ", shrink_axis_mask " << shrink_axis_mask;
                builder.AddStridedSlice(input_name, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask,
                        output_name);
#else
                throw std::invalid_argument("Unsupported layer " + layer_type_to_str(layer->type()) + " in API 28");
#endif
                break;
            }
            default: {
                throw std::invalid_argument("Unsupported layer " + layer_type_to_str(layer->type()));
            }
        }
    }
}

/**
 * It is designed to read a regular file. For reading file in assets folder of Android app,
 * read the content into a char array and call readFromBuffer
 *
 * It will throw an exception when opening file failed
 *
 * @param filepath , like "/data/local/tmp/squeezenet.daq"
 * @param builder a ModelBuilder object
 */
void DaqReader::ReadDaq(const std::string &filepath, ModelBuilder &builder, const bool use_mmap) {
    if (use_mmap) {
        const auto fd = open(filepath.c_str(), O_RDONLY);
        ReadDaq(fd, builder);
    } else {
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::unique_ptr<uint8_t []> buf(new uint8_t[size]);
        if (!file.read(reinterpret_cast<char *>(buf.get()), size)) {
            throw std::invalid_argument("Read file error");
        }
        ReadDaq(std::move(buf), builder);
    }
}

void DaqReader::ReadDaq(const int &fd, ModelBuilder &builder, const off_t offset, size_t fsize) {
    if (fd == -1) {
        throw std::invalid_argument("Open file error " + std::to_string(errno));
    }
    if (fsize == 0) {
        fsize = static_cast<size_t>(lseek(fd, offset, SEEK_END));
    }
    auto data = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, offset);
    if (data == MAP_FAILED) {
        throw std::invalid_argument("mmap failed, errno = " + std::to_string(errno));
    }
    builder.SetMemory(fd, fsize, offset);
    builder.SetBasePtr(static_cast<unsigned char*>(data));
    auto ret = close(fd);
    if (ret == -1) {
        throw std::runtime_error("close file error, errno = " + std::to_string(errno));
    }
    VLOG(4) << "Read daq from mmap";
    ReadDaqImpl(static_cast<const uint8_t *>(data), builder);
}

void DaqReader::ReadDaq(std::unique_ptr<uint8_t []> buf, ModelBuilder &builder) {
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
