#include "OnnxConverter.h"

#include <string>
#include <fstream>
#include <numeric>
#include <map>

#include <glog/logging.h>
#include <onnx/optimizer/optimize.h>
#include <common/StrKeyMap.h>
#include <common/Shaper.h>
#include "NodeAttrHelper.h"

using std::string; using std::vector;
using Shape = Shaper::Shape;

std::string OnnxConverter::m(const std::string &str) {
    if (name_map_.find(str) != name_map_.end()) {
        return name_map_[str];
    }
    
    return str;
}

DNN::FuseCode OnnxConverter::ConvertFuseCodeType(FuseCode fuse_code) {
    switch (fuse_code) {
        case FuseCode::FUSED_NONE:
            return DNN::FuseCode::None;
        case FuseCode::FUSED_RELU:
            return DNN::FuseCode::Relu;
        case FuseCode::FUSED_RELU1:
            return DNN::FuseCode::Relu1;
        case FuseCode::FUSED_RELU6:
            return DNN::FuseCode::Relu6;
    }
    throw std::invalid_argument("Invalid FuseCode");
}

std::pair<nonstd::optional<std::string>, OnnxConverter::FuseCode> OnnxConverter::FindActivation(const ONNX_NAMESPACE::ModelProto &model_proto, css &output_name) {
    std::pair<nonstd::optional<string>, FuseCode> activation{{}, FuseCode::FUSED_NONE};
    for (const auto &_node : model_proto.graph().node()) {
        if (!_node.input().empty() && output_name == _node.input(0) && _node.op_type() == "Relu") {
            // If there are two branches after a conv/pool and both branches has a relu on the top, we have to add two normal relu layers
            if (activation.second != FuseCode::FUSED_NONE) {
                return {{}, FuseCode::FUSED_NONE};
            }
            activation = std::make_pair(nonstd::make_optional(_node.name()), FuseCode::FUSED_RELU);
        }
    }
    return activation;
}

void OnnxConverter::AddConv(const string &input_name, const std::vector<int> &strides, const std::vector<int> &pads, 
        const std::vector<int> &dilations, int group, 
        const std::pair<nonstd::optional<std::string>, FuseCode>& activation,
        const string &ori_weight_name, const nonstd::optional<std::string> &bias_name, const string &output_name) {
    flatbuffers::Offset<DNN::Layer> layer;
    if (dilations != vector<int>{1, 1}) {
        if (strides != vector<int>{1, 1}) {
            throw std::invalid_argument("Both dilations and strides > 1 is not supported for now");
        }
        LOG(INFO) << "Dilations of conv: " << dilations << ", converting..";
        const auto s2b_name = input_name + "_s2b";
        const auto im_name = input_name + "_conv_imm";
        const auto b2s_name = input_name + "_b2s";
        std::vector<int> new_pads = pads;
        auto input_shape = shaper_[input_name];
        new_pads[1] = (input_shape[1] + pads[1] + (dilations[0] - 1)) / dilations[0] * dilations[0] - input_shape[1];
        new_pads[3] = (input_shape[2] + pads[3] + (dilations[1] - 1)) / dilations[1] * dilations[1] - input_shape[2];
        LOG(INFO) << input_shape << ", " << pads << ", " << dilations << ", " << new_pads;
        {
            shaper_.SpaceToBatch(input_name, dilations, new_pads, s2b_name);
            auto param = DNN::CreateSpaceToBatchDirect(builder_, input_name.c_str(), &dilations, &new_pads, s2b_name.c_str());
            layer = DNN::CreateLayer(builder_, DNN::LayerType::SpaceToBatch, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);
        }
        {
            // paddings are applied in spacetobatch
            AddConv(s2b_name, strides, vector<int>{0, 0, 0, 0}, vector<int>{1, 1}, group, activation, ori_weight_name, bias_name, im_name);
        }
        {
            shaper_.BatchToSpace(im_name, dilations, b2s_name);
            auto param = DNN::CreateBatchToSpaceDirect(builder_, im_name.c_str(), &dilations, b2s_name.c_str());
            layer = DNN::CreateLayer(builder_, DNN::LayerType::BatchToSpace, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);
        }
        {
            auto b2s_shape = shaper_[b2s_name];
            std::vector<int32_t> starts{0, 0, 0, 0};
            std::vector<int32_t> ends{static_cast<int32_t>(b2s_shape[0]), 
                static_cast<int32_t>(b2s_shape[1]) - (new_pads[1] - pads[0]), 
                static_cast<int32_t>(b2s_shape[2]) - (new_pads[3] - pads[3]), 
                static_cast<int32_t>(b2s_shape[3])};
            std::vector<int32_t> strides_in_ss{1, 1, 1, 1};
            int32_t begin_mask = 0;
            int32_t end_mask = 0;
            int32_t shrink_axis_mask = 0;
            shaper_.StridedSlice(b2s_name, starts, ends, strides_in_ss, begin_mask, end_mask, shrink_axis_mask, output_name);
            auto param = DNN::CreateStridedSliceDirect(builder_, b2s_name.c_str(), &starts, &ends, &strides_in_ss,
                    begin_mask, end_mask, shrink_axis_mask, output_name.c_str());
            layer = DNN::CreateLayer(builder_, DNN::LayerType::StridedSlice, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);
        }
        return;
    }

    const auto &onnx_weight = onnx_tensors_.at(ori_weight_name);
    string weight_name;
    Tensor weight_tensor;
    if (group == 1) {
        LOG(INFO) << "Vanilla conv";
        weight_name = ori_weight_name + "_conv_w";
        weight_tensor = OnnxToNnapiVanilla(onnx_weight, weight_name);
        shaper_.AddShape(weight_name, weight_tensor.shape);
        shaper_.Conv(input_name, strides[1], strides[0], 1, 1, pads[2], pads[3], pads[0], pads[1], weight_name, output_name);
        nnapi_tensors_[weight_name] = weight_tensor;

        auto param = DNN::CreateConv2DDirect(builder_, input_name.c_str(), weight_name.c_str(),
                bias_name ? bias_name.value().c_str() : nullptr,
                &pads, &strides, ConvertFuseCodeType(activation.second), output_name.c_str());
        layer = DNN::CreateLayer(builder_, DNN::LayerType::Conv2D, param);
    } else if (onnx_weight.shape[1] == 1) {    // depthwise
        LOG(INFO) << "Depthwise conv";
        weight_name = ori_weight_name + "_conv_w";
        weight_tensor = OnnxToNnapiDw(onnx_weight, weight_name);
        shaper_.AddShape(weight_name, weight_tensor.shape);
        shaper_.DepthwiseConv(input_name, strides[1], strides[0], 1, 1, pads[2], pads[3], pads[0], pads[1], weight_name, output_name);
        nnapi_tensors_[weight_name] = weight_tensor;
        auto multiplier = nnapi_tensors_.at(weight_name).shape[3] / group;
        auto param = DNN::CreateDepthwiseConv2DDirect(builder_, input_name.c_str(), weight_name.c_str(),
                bias_name ? bias_name.value().c_str() : nullptr,
                &pads, &strides, multiplier, ConvertFuseCodeType(activation.second), output_name.c_str());
        layer = DNN::CreateLayer(builder_, DNN::LayerType::DepthwiseConv2D, 0, 0, 0, 0, 0, 0, 0, 0, param);
    } else {
        // TODO: Support it
        throw std::invalid_argument("group != 1 is not supported");
    }
    flatbuffers::Offset<DNN::Tensor> flat_tensor;
    if (weight_tensor.data_type == Tensor::DataType::FLOAT32) {
        const auto weight_data = weight_tensor.float_data();
        flat_tensor = DNN::CreateTensorDirect(builder_, DNN::DataType::Float32, nullptr, 
                &weight_data, &weight_tensor.shape, weight_name.c_str());
    } else if (weight_tensor.data_type == Tensor::DataType::UINT8) {
        const auto quant_info = quant_infos_.at(weight_tensor.name);
        const auto weight_data = weight_tensor.uint8_data();
        DNN::DataType daq_data_type;
        if (quant_info.scales.size() == 1 && quant_info.zero_point.has_value()) {
            daq_data_type = DNN::DataType::QUANT8_ASYMM;
        } else if (quant_info.scales.size() == 1 && !quant_info.zero_point.has_value()) {
            daq_data_type = DNN::DataType::QUANT8_SYMM;
        } else {
            daq_data_type = DNN::DataType::QUANT8_SYMM_PER_CHANNEL;
        }
        flat_tensor = DNN::CreateTensorDirect(builder_, daq_data_type, &weight_data, 
                nullptr, &weight_tensor.shape, weight_name.c_str());
    } else {
        DNN_ASSERT(false, "Unknown data type of tensor");
    }
    tensors_.push_back(flat_tensor);
    layers_.push_back(layer);
}

// The reason that we only store weights rather than directly add them in daq model is that there may be different transform (nchw->nhwc or not) on the weights
void OnnxConverter::HandleInitializer() {
    for (const auto &tensor : model_proto_.graph().initializer()) {
        DNN_ASSERT(tensor.has_name(), "");
        css name = tensor.name();
        Shape shape;
        for (auto dim : tensor.dims()) {
            shape.push_back(static_cast<uint32_t>(dim));
        }
        if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            const char *ptr = tensor.float_data().empty() ?
                tensor.raw_data().data() : reinterpret_cast<const char *>(tensor.float_data().data());
            const auto data_vec = vector<char>(ptr, ptr + Product(shape) * sizeof(float));
            onnx_tensors_[name] = {name, data_vec, shape, Tensor::DataType::FLOAT32};
        } else if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
            const auto *ptr = reinterpret_cast<const char *>(tensor.raw_data().data());
            const auto data_vec = vector<char>(ptr, ptr + Product(shape));
            onnx_tensors_[name] = {name, data_vec, shape, Tensor::DataType::UINT8};
        } else if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
            const char *ptr = tensor.int32_data().empty() ?
                tensor.raw_data().data() : reinterpret_cast<const char *>(tensor.int32_data().data());
            const auto data_vec = vector<char>(ptr, ptr + Product(shape) * sizeof(int32_t));
            onnx_tensors_[name] = {name, data_vec, shape, Tensor::DataType::INT32};
        } else {
            LOG(INFO) << "invalid " << tensor.data_type();
        }
        operands_.push_back(name);
    }

}

std::vector<flatbuffers::Offset<DNN::Input>> OnnxConverter::GetInputOfOnnxModel() {
    vector<flatbuffers::Offset<DNN::Input>> inputs;

    for (const auto &input : model_proto_.graph().input()) {
        if (std::find(operands_.begin(), operands_.end(), input.name()) != operands_.end()) {
            continue;
        }

        Shape shape;
        for (const auto &dim : input.type().tensor_type().shape().dim()) {
            if (dim.value_case() == ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimValue) {
                shape.push_back(static_cast<uint32_t>(dim.dim_value()));
            } else {
                throw std::invalid_argument("The input of graph doesn't have dim_value");
            }
        }
        Shape nnapi_shape{shape[0], shape[2], shape[3], shape[1]};
        shaper_.AddShape(input.name(), nnapi_shape);
        auto flat_input = DNN::CreateInputDirect(builder_, &nnapi_shape, input.name().c_str());
        inputs.push_back(flat_input);
    }

    return inputs;
}

void OnnxConverter::ReadTableFile(css &table_file) {
    std::ifstream ifsf(table_file);
    DNN_ASSERT(!ifsf.fail(), "");
    for (std::string line; std::getline(ifsf, line); ) {
        std::stringstream ss;
        ss << line;
        std::string name;
        int scale_num, zero_point_num;
        std::vector<float> scales;
        nonstd::optional<int> zero_point;

        ss >> name >> scale_num;
        if (scale_num > 0) {
            FORZ(i, scale_num) {
                float scale;
                ss >> scale;
                scales.push_back(scale);
            }
        } else {
            scale_num = -scale_num;
            LOG(INFO) << scale_num;
            FORZ(j, scale_num) {
                std::string mul;
                ss >> mul;
                LOG(INFO) << mul;
                if (j == 0) {
                    scales = std::vector<float>(quant_infos_.at(mul).scales.size(), 1.f);
                }
                FORZ(i, scales.size()) {
                    scales[i] *= quant_infos_.at(mul).scales[i];
                }
            }
        }
        ss >> zero_point_num;
        if (zero_point_num > 0) {
            int tmp;
            ss >> tmp;
            zero_point = tmp;
        }
        std::string quant_type_str;
        QuantInfo::Type quant_type;
        ss >> quant_type_str;
        if (quant_type_str == "quant8_asymm") {
            quant_type = QuantInfo::Type::QUANT8_ASYMM;
        } else if (quant_type_str == "int32") {
            quant_type = QuantInfo::Type::INT32;
        } else {
            throw std::invalid_argument(name + " has unknown quant type: " + quant_type_str);
        }
        quant_infos_[name] = {scales, zero_point, quant_type};
        LOG(INFO) << "quant info of " << name;
    }
}

std::vector<flatbuffers::Offset<DNN::QuantInfo>> OnnxConverter::ConvertQuantInfosToFbs() {
    std::map<QuantInfo::Type, DNN::DataType> mapping_table{{QuantInfo::Type::INT32, DNN::DataType::Int32}, 
        {QuantInfo::Type::QUANT8_ASYMM, DNN::DataType::QUANT8_ASYMM}};
    std::vector<flatbuffers::Offset<DNN::QuantInfo>> ret;
    for (const auto x : quant_infos_) {
        const auto &quant_info = x.second;
        ret.push_back(DNN::CreateQuantInfoDirect(builder_, x.first.c_str(), mapping_table.at(quant_info.type), &quant_info.scales, quant_info.zero_point.value_or(0)));
    }
    return ret;
}

void OnnxConverter::Convert(const ONNX_NAMESPACE::ModelProto &model_proto, const std::string &filepath, const css &table_file) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    model_proto_ = ONNX_NAMESPACE::optimization::Optimize(model_proto, vector<string>{"extract_constant_to_initializer", "eliminate_identity", "eliminate_nop_transpose", "eliminate_nop_pad", "fuse_bn_into_conv"});

    if (!table_file.empty()) {
        ReadTableFile(table_file);
    }

    HandleInitializer();

    auto inputs = GetInputOfOnnxModel();

    bool has_reshape = false;
    for (const auto &node : model_proto_.graph().node()) {
        if (has_reshape) {
            throw std::invalid_argument("Reshape can only be the last layer for now");
        }
        NodeAttrHelper helper(node);
        const auto &op = node.op_type();
        LOG(INFO) << "Node " << node.name();
        if (op == "Conv") {
            LOG(INFO) << "Start converting Conv";
            auto strides = helper.get("strides", vector<int>{1, 1});
            auto pads = helper.get("pads", vector<int>{0, 0, 0, 0});
            auto dilations = helper.get("dilations", vector<int>{1, 1});
            CHECK_EQ(pads.size(), 4ul);
            CHECK_EQ(strides.size(), 2ul);
            CHECK_EQ(dilations.size(), 2ul);
            auto group = helper.get("group", 1);
            auto activation = FindActivation(model_proto_, node.output(0));
            if (activation.first.has_value()) {
                skipped_act_.push_back(activation.first.value());
            }
            nonstd::optional<string> bias_name;
            if (node.input_size() >= 3) {
                auto ori_bias_name = m(node.input(2));
                bias_name = ori_bias_name + "_conv_b";
                nnapi_tensors_[bias_name.value()] = onnx_tensors_.at(ori_bias_name);
                flatbuffers::Offset<DNN::Tensor> flat_tensor;
                if (nnapi_tensors_[bias_name.value()].data_type == Tensor::DataType::FLOAT32) {
                    const auto bias_data = nnapi_tensors_.at(bias_name.value()).float_data();
                    flat_tensor = DNN::CreateTensorDirect(builder_, DNN::DataType::Float32, nullptr, 
                            &bias_data, &nnapi_tensors_.at(bias_name.value()).shape, 
                            bias_name.value().c_str());
                } else if (nnapi_tensors_[bias_name.value()].data_type == Tensor::DataType::INT32) {
                    const auto bias_data = nnapi_tensors_.at(bias_name.value()).int32_data();
                    flat_tensor = DNN::CreateTensorDirect(builder_, DNN::DataType::Int32, nullptr, 
                            nullptr, &nnapi_tensors_.at(bias_name.value()).shape, 
                            bias_name.value().c_str(), nullptr, nullptr, &bias_data);
                } else {
                    std::invalid_argument("Unknown data type");
                }
                tensors_.push_back(flat_tensor);
            }

            auto ori_weight_name = m(node.input(1));
            AddConv(m(node.input(0)), strides, pads, dilations, group, activation, ori_weight_name, bias_name, m(node.output(0)));
            LOG(INFO) << "Converting Conv completed";
        } else if (op == "AveragePool" || op == "MaxPool" || op == "GlobalAveragePool" || op == "GlobalMaxPool") {
            LOG(INFO) << "Start converting Pool";
            auto input_name = m(node.input(0));
            auto output_name = m(node.output(0));
            vector<int> strides, pads, kernel_shape;
            if (op == "AveragePool" || op == "MaxPool") {
                strides = helper.get("strides", vector<int>{1, 1});
                pads = helper.get("pads", vector<int>{0, 0, 0, 0});
                kernel_shape = helper.get("kernel_shape", vector<int>{0, 0});
                auto count_include_pad = helper.get("count_include_pad", 0);
                if (count_include_pad == 1) {
                    throw std::invalid_argument("count_include_pad == 1 is not supported");
                }
                auto storage_order = helper.get("storage_order", 0);
                if (storage_order == 1) {
                    throw std::invalid_argument("storage_order == 1 is not supported");
                }
                if (helper.has_attr("auto_pad")) {
                    throw std::invalid_argument("auto_pad is not supported");
                }
            } else {
                strides = {0, 0};
                pads = {0, 0, 0, 0};
                kernel_shape = {-1, -1};    // -1 for global
            }
            CHECK_EQ(pads.size(), 4ul);
            CHECK_EQ(kernel_shape.size(), 2ul);
            CHECK_EQ(strides.size(), 2ul);
            addLayerPool(op, input_name, kernel_shape, pads, strides, output_name);
            // operand_indexes[node.output(0)] = builder_.addPool(operand_indexes.at(node.input(0)), strides[1], strides[0],
            // pads[2], pads[3], pads[0], pads[1],
            // kernel_shape[0], kernel_shape[1], activation.second,
            // op == "AveragePool" ? ModelBuilder::AVE_POOL
            // : ModelBuilder::MAX_POOL);
            LOG(INFO) << "Converting Pool completed";
        } else if (op == "Relu") {
            LOG(INFO) << "Start converting Relu";
            auto input_name = m(node.input(0));
            auto output_name = m(node.output(0));
            addLayerRelu(input_name, output_name);
            LOG(INFO) << "Converting Relu completed";
            // operand_indexes[node.output(0)] = builder_.addReLU(operand_indexes.at(node.input(0)));

        } else if (op == "PRelu") {
            LOG(INFO) << "Start converting PRelu";
            const auto input_name = m(node.input(0));
            const auto slope_name = m(node.input(1));
            const auto output_name = m(node.output(0));
            const auto imm1_name = output_name + "_imm1";
            const auto imm2_name = output_name + "_imm2";
            const auto imm3_name = output_name + "_imm3";
            const auto imm4_name = output_name + "_imm4";
            if (onnx_tensors_[slope_name].shape != Shape{1}) {
                // TODO: support it
                throw std::invalid_argument("Only support one element slope.");
            }
            addLayerRelu(input_name, imm1_name);
            addLayerMul(input_name, -onnx_tensors_[slope_name].data[0], imm2_name);
            addLayerRelu(imm2_name, imm3_name);
            addLayerMul(imm3_name, -1.f, imm4_name);
            addLayerAdd(imm1_name, imm4_name, output_name);
            // TODO:
            LOG(INFO) << "Converting PRelu completed";
        } else if (op == "Add") {
            LOG(INFO) << "Start converting Add";
            auto input1_name = m(node.input(0));
            auto input2_name = m(node.input(1));
            auto output_name = m(node.output(0));
            addLayerAdd(input1_name, input2_name, output_name);
            LOG(INFO) << "Converting Add completed";
            // auto input1 = operand_indexes.at(node.input(0));
            // auto input2 = operand_indexes.at(node.input(1));
            // operand_indexes[node.output(1)] = builder.addAddTensor(input1, input2);

        } else if (op == "Mul") {
            LOG(INFO) << "Start converting Mul";
            const auto input1_name = m(node.input(0));
            const auto input2_name = m(node.input(1));
            const auto output_name = m(node.output(0));
            addLayerMul(input1_name, input2_name, output_name);
            LOG(INFO) << "Converting Mul completed";
        } else if (op == "Gemm") {
            LOG(INFO) << "Start converting Gemm";
            auto input_name = m(node.input(0));
            auto weight_name = m(node.input(1));
            auto output_name = m(node.output(0));
            nonstd::optional<string> bias_name;
            if (node.input_size() >= 3) {
                bias_name = m(node.input(2));
            }
            auto transA = helper.get("transA", 0);
            auto transB = helper.get("transB", 0);
            auto alpha = helper.get("alpha", 1.0f);
            auto beta = helper.get("beta", 1.0f);
            addLayerGemm(input_name, weight_name, bias_name, transA, transB, alpha, beta, output_name);
            LOG(INFO) << "Converting Gemm completed";
        } else if (op == "Softmax") {
            LOG(INFO) << "Start converting Softmax";
            auto input_name = m(node.input(0));
            auto output_name = m(node.output(0));
            addLayerSoftmax(input_name, output_name);
            LOG(INFO) << "Converting Softmax completed";
        } else if (op == "Concat") {
            LOG(INFO) << "Start converting Concat";
            vector<std::string> concat_inputs_str;
            for (const auto &onnx_input : node.input()) {
                concat_inputs_str.push_back(onnx_input);
            }
            auto axis = helper.get("axis", 1);
            auto output_name = m(node.output(0));
            addLayerConcat(concat_inputs_str, output_name, axis);
            LOG(INFO) << "Converting Concat completed";
        } else if (op == "Dropout") {
            LOG(INFO) << "Start converting Dropout";
            // Dropout does nothing, so the output is the same as the input
            name_map_[node.output(0)] = m(node.input(0));
            LOG(INFO) << "Converting Dropout completed";
        } else if (op == "Reshape") {
            LOG(INFO) << "Start converting Reshape";
            has_reshape = true;
            LOG(INFO) << "Converting Reshape completed";
        } else {
            throw std::invalid_argument("Unsupported operator " + op);
        }
    }
    auto flat_layers = builder_.CreateVector(layers_);
    auto flat_inputs = builder_.CreateVector(inputs);
    auto flat_tensors = builder_.CreateVector(tensors_);
    const auto flat_quant_infos = builder_.CreateVector(ConvertQuantInfosToFbs());
    auto flat_model = DNN::CreateModel(builder_, flat_layers, flat_tensors, flat_inputs, flat_quant_infos);

    builder_.Finish(flat_model);

    LOG(INFO) << "Shapes: ";
    LOG(INFO) << shaper_;

    std::ofstream ofs(filepath);
    ofs.write(reinterpret_cast<char *>(builder_.GetBufferPointer()), builder_.GetSize());
    ofs.close();

    skipped_act_.clear();
    layers_.clear();
    operands_.clear();
    tensors_.clear();
    name_map_.clear();
    nnapi_tensors_.clear();
    onnx_tensors_.clear();
    shaper_.Clear();
}
