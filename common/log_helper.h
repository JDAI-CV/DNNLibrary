#ifndef DNN_LOG_HELPER_H
#define DNN_LOG_HELPER_H

#include <iostream>
#include <vector>

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

#endif
