//
// Created by daquexian on 8/13/18.
//

#include <common/data_types.h>
#include <common/internal_vars.h>
#include <dnnlibrary/DaqReader.h>
#include <dnnlibrary/android_log_helper.h>
#include <dnnlibrary/flatbuffers_helper.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

namespace dnn {
expected<Unit, std::string> ReadDaqImpl(const uint8_t *buf,
                                        ModelBuilder &builder);

std::string layer_type_to_str(DNN::LayerType type) {
    switch (type) {
            // DaqReader auto generated layer_type_to_str start
        case DNN::LayerType::CONV_2D:
            return "CONV_2D";
        case DNN::LayerType::AVERAGE_POOL_2D:
            return "AVERAGE_POOL_2D";
        case DNN::LayerType::MAX_POOL_2D:
            return "MAX_POOL_2D";
        case DNN::LayerType::RELU:
            return "RELU";
        case DNN::LayerType::SOFTMAX:
            return "SOFTMAX";
        case DNN::LayerType::FULLY_CONNECTED:
            return "FULLY_CONNECTED";
        case DNN::LayerType::ADD:
            return "ADD";
        case DNN::LayerType::CONCATENATION:
            return "CONCATENATION";
        case DNN::LayerType::DEPTHWISE_CONV_2D:
            return "DEPTHWISE_CONV_2D";
        case DNN::LayerType::BATCH_TO_SPACE_ND:
            return "BATCH_TO_SPACE_ND";
        case DNN::LayerType::SPACE_TO_BATCH_ND:
            return "SPACE_TO_BATCH_ND";
        case DNN::LayerType::STRIDED_SLICE:
            return "STRIDED_SLICE";
        case DNN::LayerType::MUL:
            return "MUL";
        case DNN::LayerType::DEQUANTIZE:
            return "DEQUANTIZE";
        case DNN::LayerType::LOCAL_RESPONSE_NORMALIZATION:
            return "LOCAL_RESPONSE_NORMALIZATION";
        case DNN::LayerType::TANH:
            return "TANH";
        case DNN::LayerType::FLOOR:
            return "FLOOR";
        case DNN::LayerType::LOGISTIC:
            return "LOGISTIC";
        case DNN::LayerType::PRELU:
            return "PRELU";
        case DNN::LayerType::POW:
            return "POW";
        case DNN::LayerType::NEG:
            return "NEG";
        case DNN::LayerType::MINIMUM:
            return "MINIMUM";
        case DNN::LayerType::MAXIMUM:
            return "MAXIMUM";
        case DNN::LayerType::LOG:
            return "LOG";
        case DNN::LayerType::ABS:
            return "ABS";
        case DNN::LayerType::EXP:
            return "EXP";
        case DNN::LayerType::SUB:
            return "SUB";
            // DaqReader auto generated layer_type_to_str end
    }
}

int convert_fuse_code_to_nnapi(const DNN::FuseCode fuse_code) {
    switch (fuse_code) {
        case DNN::FuseCode::None:
            return ANEURALNETWORKS_FUSED_NONE;
        case DNN::FuseCode::Relu:
            return ANEURALNETWORKS_FUSED_RELU;
        case DNN::FuseCode::Relu1:
            return ANEURALNETWORKS_FUSED_RELU1;
        case DNN::FuseCode::Relu6:
            return ANEURALNETWORKS_FUSED_RELU6;
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

dnn::optional<ModelBuilder::QuantInfo> DaqQuantInfoToModelBuilderQuantInfo(
    const DNN::QuantInfo *daq_quant_info) {
    if (daq_quant_info == nullptr) {
        return dnn::nullopt;
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

bool CheckVersion(const DNN::Model *model) {
    return model->version() == dnn::CURRENT_MODEL_VERSION;
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
            OperandType operand_type(Type::TENSOR_FLOAT32, shape);
            builder.AddInput(input_name, operand_type);
        }
    }
}

void AddOutputs(const DNN::Model &model, ModelBuilder &builder) {
    using namespace android::nn::wrapper;
    if (model.outputs() == nullptr) {
        return;
    }
    for (const auto &output : *model.outputs()) {
        css output_name = output->str();
        builder.AddOutput(output_name);
    }
}

expected<Unit, std::string> AddLayers(const DNN::Model &model,
                                      ModelBuilder &builder) {
    for (const auto layer : *model.layers()) {
        switch (layer->type()) {
                // auto generated layer reader start
            case DNN::LayerType::CONV_2D: {
                UNPACK_LAYER_QUANT(CONV_2D, input, weight, bias, padding_left,
                                   padding_right, padding_top, padding_bottom,
                                   stride_x, stride_y, fuse_code, nchw,
                                   dilation_x, dilation_y);
                const dnn::optional<std::string> bias_right_type =
                    (bias == "") ? dnn::nullopt : dnn::make_optional(bias);

                TRY(builder.AddLayer_CONV_2D(
                    input, weight, bias_right_type, padding_left, padding_right,
                    padding_top, padding_bottom, stride_x, stride_y, fuse_code,
                    nchw, dilation_x, dilation_y, output, quant_info));
                break;
            }
            case DNN::LayerType::AVERAGE_POOL_2D: {
                UNPACK_LAYER_QUANT(AVERAGE_POOL_2D, input, padding_left,
                                   padding_right, padding_top, padding_bottom,
                                   stride_x, stride_y, kernel_width,
                                   kernel_height, fuse_code);

                TRY(builder.AddLayer_AVERAGE_POOL_2D(
                    input, padding_left, padding_right, padding_top,
                    padding_bottom, stride_x, stride_y, kernel_width,
                    kernel_height, fuse_code, output, quant_info));
                break;
            }
            case DNN::LayerType::MAX_POOL_2D: {
                UNPACK_LAYER_QUANT(MAX_POOL_2D, input, padding_left,
                                   padding_right, padding_top, padding_bottom,
                                   stride_x, stride_y, kernel_width,
                                   kernel_height, fuse_code);

                TRY(builder.AddLayer_MAX_POOL_2D(
                    input, padding_left, padding_right, padding_top,
                    padding_bottom, stride_x, stride_y, kernel_width,
                    kernel_height, fuse_code, output, quant_info));
                break;
            }
            case DNN::LayerType::RELU: {
                UNPACK_LAYER_QUANT(RELU, input);

                TRY(builder.AddLayer_RELU(input, output));
                break;
            }
            case DNN::LayerType::SOFTMAX: {
                UNPACK_LAYER_QUANT(SOFTMAX, input, beta);

                TRY(builder.AddLayer_SOFTMAX(input, beta, output));
                break;
            }
            case DNN::LayerType::FULLY_CONNECTED: {
                UNPACK_LAYER_QUANT(FULLY_CONNECTED, input, weight, bias,
                                   fuse_code);
                const dnn::optional<std::string> bias_right_type =
                    (bias == "") ? dnn::nullopt : dnn::make_optional(bias);

                TRY(builder.AddLayer_FULLY_CONNECTED(input, weight,
                                                     bias_right_type, fuse_code,
                                                     output, quant_info));
                break;
            }
            case DNN::LayerType::ADD: {
                UNPACK_LAYER_QUANT(ADD, input1, input2, fuse_code);

                TRY(builder.AddLayer_ADD(input1, input2, fuse_code, output,
                                         quant_info));
                break;
            }
            case DNN::LayerType::CONCATENATION: {
                UNPACK_LAYER_QUANT(CONCATENATION, inputs, axis);

                TRY(builder.AddLayer_CONCATENATION(inputs, axis, output));
                break;
            }
            case DNN::LayerType::DEPTHWISE_CONV_2D: {
                UNPACK_LAYER_QUANT(DEPTHWISE_CONV_2D, input, weight, bias,
                                   padding_left, padding_right, padding_top,
                                   padding_bottom, stride_x, stride_y,
                                   depth_multiplier, fuse_code);
                const dnn::optional<std::string> bias_right_type =
                    (bias == "") ? dnn::nullopt : dnn::make_optional(bias);

                TRY(builder.AddLayer_DEPTHWISE_CONV_2D(
                    input, weight, bias_right_type, padding_left, padding_right,
                    padding_top, padding_bottom, stride_x, stride_y,
                    depth_multiplier, fuse_code, output, quant_info));
                break;
            }
            case DNN::LayerType::BATCH_TO_SPACE_ND: {
                UNPACK_LAYER_QUANT(BATCH_TO_SPACE_ND, input, block_sizes);

                TRY(builder.AddLayer_BATCH_TO_SPACE_ND(input, block_sizes,
                                                       output));
                break;
            }
            case DNN::LayerType::SPACE_TO_BATCH_ND: {
                UNPACK_LAYER_QUANT(SPACE_TO_BATCH_ND, input, block_sizes, pads);

                TRY(builder.AddLayer_SPACE_TO_BATCH_ND(input, block_sizes, pads,
                                                       output));
                break;
            }
            case DNN::LayerType::STRIDED_SLICE: {
                UNPACK_LAYER_QUANT(STRIDED_SLICE, input, starts, ends, strides,
                                   begin_mask, end_mask, shrink_axis_mask);

                TRY(builder.AddLayer_STRIDED_SLICE(input, starts, ends, strides,
                                                   begin_mask, end_mask,
                                                   shrink_axis_mask, output));
                break;
            }
            case DNN::LayerType::MUL: {
                UNPACK_LAYER_QUANT(MUL, input1, input2, fuse_code);

                TRY(builder.AddLayer_MUL(input1, input2, fuse_code, output,
                                         quant_info));
                break;
            }
            case DNN::LayerType::DEQUANTIZE: {
                UNPACK_LAYER_QUANT(DEQUANTIZE, input);

                TRY(builder.AddLayer_DEQUANTIZE(input, output));
                break;
            }
            case DNN::LayerType::LOCAL_RESPONSE_NORMALIZATION: {
                UNPACK_LAYER_QUANT(LOCAL_RESPONSE_NORMALIZATION, input, radius,
                                   bias, alpha, beta);

                TRY(builder.AddLayer_LOCAL_RESPONSE_NORMALIZATION(
                    input, radius, bias, alpha, beta, output));
                break;
            }
            case DNN::LayerType::TANH: {
                UNPACK_LAYER_QUANT(TANH, input);

                TRY(builder.AddLayer_TANH(input, output));
                break;
            }
            case DNN::LayerType::FLOOR: {
                UNPACK_LAYER_QUANT(FLOOR, input);

                TRY(builder.AddLayer_FLOOR(input, output));
                break;
            }
            case DNN::LayerType::LOGISTIC: {
                UNPACK_LAYER_QUANT(LOGISTIC, input);

                TRY(builder.AddLayer_LOGISTIC(input, output));
                break;
            }
            case DNN::LayerType::PRELU: {
                UNPACK_LAYER_QUANT(PRELU, input, alpha);

                TRY(builder.AddLayer_PRELU(input, alpha, output));
                break;
            }
            case DNN::LayerType::POW: {
                UNPACK_LAYER_QUANT(POW, input, exp);

                TRY(builder.AddLayer_POW(input, exp, output));
                break;
            }
            case DNN::LayerType::NEG: {
                UNPACK_LAYER_QUANT(NEG, input);

                TRY(builder.AddLayer_NEG(input, output));
                break;
            }
            case DNN::LayerType::MINIMUM: {
                UNPACK_LAYER_QUANT(MINIMUM, input1, input2);

                TRY(builder.AddLayer_MINIMUM(input1, input2, output));
                break;
            }
            case DNN::LayerType::MAXIMUM: {
                UNPACK_LAYER_QUANT(MAXIMUM, input1, input2);

                TRY(builder.AddLayer_MAXIMUM(input1, input2, output));
                break;
            }
            case DNN::LayerType::LOG: {
                UNPACK_LAYER_QUANT(LOG, input);

                TRY(builder.AddLayer_LOG(input, output));
                break;
            }
            case DNN::LayerType::ABS: {
                UNPACK_LAYER_QUANT(ABS, input);

                TRY(builder.AddLayer_ABS(input, output));
                break;
            }
            case DNN::LayerType::EXP: {
                UNPACK_LAYER_QUANT(EXP, input);

                TRY(builder.AddLayer_EXP(input, output));
                break;
            }
            case DNN::LayerType::SUB: {
                UNPACK_LAYER_QUANT(SUB, input1, input2, fuse_code);

                TRY(builder.AddLayer_SUB(input1, input2, fuse_code, output));
                break;
            }
                // auto generated layer reader end
                // case DNN::LayerType::CONV_2D: {
                //     UNPACK_LAYER_QUANT(CONV_2D, strides, pads, fuse, input,
                //     weight,
                //                        bias, output);
                //     builder.AddCONV_2D(
                //         input, strides[1], strides[0], pads[2], pads[3],
                //         pads[0], pads[1], fuse, weight, (bias != "" ?
                //         dnn::make_optional(bias) : dnn::nullopt), output,
                //         quant_info);
                //     break;
                // }
                // case DNN::LayerType::DepthwiseConv2D: {
                //     UNPACK_LAYER_QUANT(depthwise_conv2d, strides, pads,
                //     multiplier,
                //                        fuse, input, weight, bias, output);
                //     builder.AddDepthWiseConv(
                //         input, strides[1], strides[0], pads[2], pads[3],
                //         pads[1], pads[0], fuse, multiplier, weight, (bias !=
                //         "" ? dnn::make_optional(bias) : dnn::nullopt),
                //         output, quant_info);
                //     break;
                // }
                // case DNN::LayerType::AvePool: {
                //     UNPACK_LAYER_QUANT(avepool, strides, pads, kernel_shape,
                //     fuse,
                //                        input, output);
                //     builder.AddPool(
                //         input, strides[1], strides[0], pads[1], pads[3],
                //         pads[0], pads[2], kernel_shape[0], kernel_shape[1],
                //         fuse, ModelBuilder::PoolingType::AVE_POOL, output,
                //         quant_info);
                //     break;
                // }
                // case DNN::LayerType::MaxPool: {
                //     UNPACK_LAYER_QUANT(maxpool, strides, pads, kernel_shape,
                //     fuse,
                //                        input, output);
                //     builder.AddPool(
                //         input, strides[1], strides[0], pads[1], pads[3],
                //         pads[0], pads[2], kernel_shape[0], kernel_shape[1],
                //         fuse, ModelBuilder::PoolingType::MAX_POOL, output,
                //         quant_info);
                //     break;
                // }
                // case DNN::LayerType::Relu: {
                //     ADD_LAYER(relu, ReLU, input, output);
                //     break;
                // }
                // case DNN::LayerType::Add: {
                //     ADD_LAYER_QUANT(add, Add, input1, input2, fuse, output);
                //     break;
                // }
                // case DNN::LayerType::AddScalar: {
                //     ADD_LAYER(add_scalar, Add, input1, input2, fuse, output);
                //     break;
                // }
                // case DNN::LayerType::Mul: {
                //     ADD_LAYER_QUANT(mul, Mul, input1, input2, fuse, output);
                //     break;
                // }
                // case DNN::LayerType::MulScalar: {
                //     ADD_LAYER(mul_scalar, Mul, input1, input2, fuse, output);
                //     break;
                // }
                // case DNN::LayerType::FC: {
                //     UNPACK_LAYER_QUANT(fc, input, weight, bias, fuse,
                //     output); builder.AddFC(
                //         input, fuse, weight,
                //         (bias != "" ? dnn::make_optional(bias) :
                //         dnn::nullopt), output, quant_info);
                //     break;
                // }
                // case DNN::LayerType::Softmax: {
                //     UNPACK_LAYER(softmax, input, output);
                //     builder.AddSoftmax(input, 1.f, output);
                //     break;
                // }
                // case DNN::LayerType::Concat: {
                //     ADD_LAYER(concat, Concat, inputs, axis, output)
                //     break;
                // }
                // case DNN::LayerType::Dequantize: {
                //     ADD_LAYER(dequantize, Dequantize, input, output);
                //     break;
                // }
                // case DNN::LayerType::BatchToSpace: {
                //     ADD_LAYER(batch_to_space, BatchToSpaceND, input,
                //     block_sizes,
                //               output);
                //     break;
                // }
                // case DNN::LayerType::SpaceToBatch: {
                //     ADD_LAYER(space_to_batch, SpaceToBatchND, input,
                //     block_sizes,
                //               pads, output);
                //     break;
                // }
                // case DNN::LayerType::StridedSlice: {
                //     ADD_LAYER(strided_slice, StridedSlice, input, starts,
                //     ends,
                //               strides, begin_mask, end_mask,
                //               shrink_axis_mask, output);
                //     break;
                // }
                // case DNN::LayerType::LRN: {
                //     ADD_LAYER(lrn, LRN, input, radius, bias, alpha, beta,
                //     output); break;
                // }
                // case DNN::LayerType::Tanh: {
                //     ADD_LAYER(tanh, Tanh, input, output);
                //     break;
                // }
                // case DNN::LayerType::Floor: {
                //     ADD_LAYER(floor, Floor, input, output);
                //     break;
                // }
                // case DNN::LayerType::Logistic: {
                //     ADD_LAYER(logistic, Logistic, input, output);
                //     break;
                // }
        }
    }
    return Unit();
}

/**
 * It is designed to read a regular file. For reading file in assets folder of
 * Android app, read the content into a char array and call readFromBuffer
 *
 * It will return an unexpected object when opening file failed
 *
 * @param filepath , like "/data/local/tmp/squeezenet.daq"
 * @param builder a ModelBuilder object
 */
expected<Unit, std::string> DaqReader::ReadDaq(const std::string &filepath,
                                               ModelBuilder &builder,
                                               const bool use_mmap) {
    if (use_mmap) {
        const auto fd = open(filepath.c_str(), O_RDONLY);
        return ReadDaq(fd, builder);
    } else {
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::unique_ptr<uint8_t[]> buf(new uint8_t[size]);
        if (!file.read(reinterpret_cast<char *>(buf.get()), size)) {
            return make_unexpected("Read file error");
        }
        return ReadDaq(std::move(buf), builder);
    }
}

expected<Unit, std::string> DaqReader::ReadDaq(const int &fd,
                                               ModelBuilder &builder,
                                               const off_t offset,
                                               size_t fsize) {
    if (fd == -1) {
        return make_unexpected("Open file error " + std::to_string(errno));
    }
    if (fsize == 0) {
        fsize = static_cast<size_t>(lseek(fd, offset, SEEK_END));
    }
    auto data = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, offset);
    if (data == MAP_FAILED) {
        return make_unexpected("mmap failed, errno = " + std::to_string(errno));
    }
    builder.SetMemory(fd, fsize, offset);
    builder.SetBasePtr(static_cast<unsigned char *>(data));
    auto ret = close(fd);
    if (ret == -1) {
        return make_unexpected("close file error, errno = " +
                               std::to_string(errno));
    }
    VLOG(4) << "Read daq from mmap";
    return ReadDaqImpl(static_cast<const uint8_t *>(data), builder);
}

expected<Unit, std::string> DaqReader::ReadDaq(std::unique_ptr<uint8_t[]> buf,
                                               ModelBuilder &builder) {
    TRY(ReadDaq(buf.get(), builder));
    builder.RegisterBufferPointer(std::move(buf));
    return Unit();
}

expected<Unit, std::string> DaqReader::ReadDaq(const uint8_t *buf,
                                               ModelBuilder &builder) {
    VLOG(4) << "Read daq from buffer";
    return ReadDaqImpl(buf, builder);
}

expected<Unit, std::string> ReadDaqImpl(const uint8_t *buf,
                                        ModelBuilder &builder) {
    builder.Prepare();  // a daq file should be a full model, so prepare here
    auto model = DNN::GetModel(buf);
    if (!CheckVersion(model)) {
        return make_unexpected(
            "The model is out-dated. Please re-generated your model");
    }
    AddInitializersFromBuffer(*model, builder);
    AddInputs(*model, builder);
    TRY(AddLayers(*model, builder));
    AddOutputs(*model, builder);
    return Unit();
}
}  // namespace dnn
