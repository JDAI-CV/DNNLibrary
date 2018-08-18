//
// Created by daquexian on 8/13/18.
//

#include "DaqReader.h"

#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <unistd.h>

#include <android_log_helper.h>

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
    }
}

int convert_fuse_code_to_nnapi(DNN::FuseCode fuse_code) {
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

bool DaqReader::ReadDaq(std::string filepath, ModelBuilder &builder) {
    builder.prepare();
    std::ifstream ifs(filepath, std::ifstream::ate | std::ifstream::binary);
    ifs.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    auto size = static_cast<size_t>(ifs.tellg());
    ifs.close();
    bool success = true;
    std::map<std::string, ModelBuilder::Index> operand_indexes;
    try {
        auto fd = open(filepath.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::invalid_argument("Open file error " + std::to_string(errno));
        }
        auto data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        builder.setBuffer(static_cast<unsigned char *>(data), size);
        builder.setMemory(fd, size, 0);
        // close(fd);
        auto model = DNN::GetModel(data);

        for (const auto &tensor : *model->initializers()) {
            if (tensor->data_type() == DNN::DataType::Float32) {
                ModelBuilder::Shape shape(tensor->shape()->begin(), tensor->shape()->end());
                operand_indexes[tensor->name()->str()] = builder.addTensorFromMemory(tensor->float32_data()->Data(),
                                                                                     shape);
            }
        }

        for (const auto &input : *model->inputs()) {
            ModelBuilder::Shape shape(input->shape()->begin(), input->shape()->end());
            operand_indexes[input->name()->str()] = builder.addInput(shape[2], shape[3], shape[1]);
        }

        for (auto layer : *model->layers()) {
            switch (layer->type()) {
                case DNN::LayerType::Conv2D: {
                    auto param = layer->conv2d_param();
                    auto strides = param->strides();
                    auto pads = param->pads();
                    auto fuse = param->fuse();
                    auto input_name = param->input()->str();
                    auto weight_name = param->weight()->str();
                    auto bias_name = param->bias()->str();
                    auto output_name = param->output()->str();
                    operand_indexes[output_name] = 
                        builder.addConv(operand_indexes.at(input_name), strides->Get(1), strides->Get(0), 
                                pads->Get(2), pads->Get(3), pads->Get(0), pads->Get(1),
                                convert_fuse_code_to_nnapi(fuse), operand_indexes.at(weight_name), 
                                (bias_name.empty() ? std::nullopt : std::make_optional(operand_indexes.at(bias_name))));
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
                    auto bias_name = param->bias()->str();
                    auto output_name = param->output()->str();
                    operand_indexes[output_name] = 
                        builder.addDepthWiseConv(operand_indexes.at(input_name), strides->Get(1), strides->Get(0), 
                                pads->Get(2), pads->Get(3), pads->Get(1), pads->Get(0), 
                                convert_fuse_code_to_nnapi(fuse), multiplier, operand_indexes.at(weight_name), 
                                (bias_name.empty() ? std::nullopt : std::make_optional(operand_indexes.at(bias_name))));
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
                    operand_indexes[output_name] = 
                        builder.addPool(operand_indexes.at(input_name), strides->Get(1), strides->Get(0),
                                pads->Get(2), pads->Get(3), pads->Get(0), pads->Get(1),
                                kernel_shape->Get(0), kernel_shape->Get(1), convert_fuse_code_to_nnapi(fuse),
                                ModelBuilder::AVE_POOL);
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
                    operand_indexes[output_name] = 
                        builder.addPool(operand_indexes.at(input_name), strides->Get(1), strides->Get(0),
                                pads->Get(2), pads->Get(3), pads->Get(0), pads->Get(1),
                                kernel_shape->Get(0), kernel_shape->Get(1), convert_fuse_code_to_nnapi(fuse),
                                ModelBuilder::MAX_POOL);
                    break;
                }
                case DNN::LayerType::Relu: {
                    auto param = layer->relu_param();
                    auto input_name = param->input()->str();
                    auto output_name = param->output()->str();
                    operand_indexes[output_name] =
                        builder.addReLU(operand_indexes.at(input_name));
                    break;
                }
                case DNN::LayerType::Add: {
                    auto param = layer->add_param();
                    auto input1_name = param->input1()->str();
                    auto input2_name = param->input2()->str();
                    auto output_name = param->output()->str();
                    operand_indexes[output_name] = 
                        builder.addAddTensor(operand_indexes.at(input1_name),
                                operand_indexes.at(input2_name));
                    break;
                }
                case DNN::LayerType::FC: {
                    auto param = layer->fc_param();
                    auto fuse = param->fuse();
                    auto weight_name = param->weight()->str();
                    auto bias_name = param->bias()->str();
                    auto input_name = param->input()->str();
                    auto output_name = param->output()->str();
                    operand_indexes[output_name] = 
                        builder.addFC(operand_indexes.at(input_name), convert_fuse_code_to_nnapi(fuse),
                                operand_indexes.at(weight_name), operand_indexes.at(bias_name));
                    break;
                }
                case DNN::LayerType::Softmax: {
                    auto param = layer->softmax_param();
                    auto input_name = param->input()->str();
                    auto output_name = param->output()->str();
                    operand_indexes[output_name] =
                        builder.addSoftMax(operand_indexes.at(input_name), 1.f);
                    break;
                }
                case DNN::LayerType::Concat: {
                    auto param = layer->concat_param();
                    auto axis = param->axis();
                    auto inputs = param->inputs();
                    auto output_name = param->output()->str();
                    std::vector<ModelBuilder::Index> input_idxes;
                    for (size_t i = 0; i < inputs->size(); i++) {
                        input_idxes.push_back(operand_indexes.at(inputs->Get(i)->str()));
                    }
                    operand_indexes[output_name] =
                        builder.addConcat(input_idxes, axis);
                    break;
                }
                default: {
                    throw std::invalid_argument("Unsupported layer" + layer_type_to_str(layer->type()));
                }
            }
        }
    } catch (const std::ifstream::failure &e) {
        LOGE("Open file %s error", filepath.c_str());
        success = false;
    }

    builder.blobNameToIndex = operand_indexes;
    return success;
}
