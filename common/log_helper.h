#ifndef DNN_LOG_HELPER_H
#define DNN_LOG_HELPER_H

#include <iostream>
#include <vector>

#include <common/data_types.h>

template <typename T>
std::ostream& operator<<(std::ostream& output, std::vector<T> const& values) {
    output << "[";
    for (size_t i = 0; i < values.size(); i++) {
        output << values[i];
        if (i != values.size() - 1) {
            output << ", ";
        }
    }
    output << "]";
    return output;
}

inline std::ostream& operator<<(std::ostream& output, const dnn::FuseCode& value) {
    switch (value) {
        case dnn::FuseCode::NONE: {
            output << "FuseCode::NONE";
            break;
        }
        case dnn::FuseCode::RELU: {
            output << "FuseCode::RELU";
            break;
        }
        case dnn::FuseCode::RELU1: {
            output << "FuseCode::RELU1";
            break;
        }
        case dnn::FuseCode::RELU6: {
            output << "FuseCode::RELU6";
            break;
        }
    }
    return output;
}

#endif
