//
// Created by daquexian on 8/22/18.
//

#ifndef DNNLIBRARY_FLATBUFFERS_HELPER_H
#define DNNLIBRARY_FLATBUFFERS_HELPER_H

#include <common/daq_generated.h>
#include <common/data_types.h>
#include <common/helper.h>
#include <flatbuffers/flatbuffers.h>

#include <string>
#include <vector>

#define UNPACK(name) const auto name = unpack_fbs(param->name());

#define UNPACK_LAYER(name, ...)                         \
    const auto *param = layer->name##_param()->input(); \
    FOR_EACH(UNPACK, __VA_ARGS__)                       \
    VLOG(5) << "Layer: " << XSTR(name);                 \
    PNT_TO(VLOG(5), __VA_ARGS__);

#define UNPACK_LAYER_QUANT(name, ...)                                          \
    const auto *param = layer->name##_param()->input();                        \
    FOR_EACH(UNPACK, __VA_ARGS__)                                              \
    VLOG(5) << "Layer: " << XSTR(name);                                        \
    PNT_TO(VLOG(5), __VA_ARGS__);                                              \
    const auto output = unpack_fbs(layer->name##_param()->output()->output()); \
    const auto *daq_quant_info = GetQuantInfo(model, output);                  \
    const auto quant_info = DaqQuantInfoToModelBuilderQuantInfo(daq_quant_info);

#define ADD_LAYER(param_name, layer_name, ...) \
    UNPACK_LAYER(param_name, __VA_ARGS__);     \
    builder.Add##layer_name(__VA_ARGS__);

#define ADD_LAYER_QUANT(param_name, layer_name, ...) \
    UNPACK_LAYER_QUANT(param_name, __VA_ARGS__);     \
    builder.Add##layer_name(__VA_ARGS__, quant_info);

// quick fix
inline const std::string get_input(const std::vector<std::string> inputs) {
    return inputs[0];
}

inline const std::string get_input(const std::string input) {
    return input;
}

inline dnn::FuseCode unpack_fbs(const DNN::FuseCode fbs) {
    switch (fbs) {
        case DNN::FuseCode::None:
            return dnn::FuseCode::NONE;
        case DNN::FuseCode::Relu:
            return dnn::FuseCode::RELU;
        case DNN::FuseCode::Relu1:
            return dnn::FuseCode::RELU1;
        case DNN::FuseCode::Relu6:
            return dnn::FuseCode::RELU6;
    }
    throw std::invalid_argument("Invalid fuse_code");
}

inline float unpack_fbs(const float fbs) {
    return fbs;
}

inline uint32_t unpack_fbs(const uint32_t fbs) {
    return fbs;
}

inline int32_t unpack_fbs(const int32_t fbs) {
    return fbs;
}

inline std::string unpack_fbs(const flatbuffers::String *fbs) {
    if (fbs == nullptr) {
        return "";
    }
    return fbs->str();
}

inline std::vector<std::string> unpack_fbs(
    const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>
        *fbs_vec) {
    using fbsoff_t = flatbuffers::uoffset_t;
    std::vector<std::string> std_vec;
    for (size_t i = 0; i < fbs_vec->size(); i++) {
        std_vec.push_back(fbs_vec->Get(static_cast<fbsoff_t>(i))->str());
    }
    return std_vec;
}

inline std::vector<int32_t> unpack_fbs(
    const flatbuffers::Vector<int32_t> *fbs_vec) {
    using fbsoff_t = flatbuffers::uoffset_t;
    std::vector<int32_t> std_vec;
    for (size_t i = 0; i < fbs_vec->size(); i++) {
        std_vec.push_back(fbs_vec->Get(static_cast<fbsoff_t>(i)));
    }
    return std_vec;
}

inline std::vector<float> unpack_fbs(
    const flatbuffers::Vector<float> *fbs_vec) {
    using fbsoff_t = flatbuffers::uoffset_t;
    std::vector<float> std_vec;
    for (size_t i = 0; i < fbs_vec->size(); i++) {
        std_vec.push_back(fbs_vec->Get(static_cast<fbsoff_t>(i)));
    }
    return std_vec;
}
#endif  // DNNLIBRARY_FLATBUFFERS_HELPER_H
