#ifndef DNNLIBRARY_HELPER_H
#define DNNLIBRARY_HELPER_H

#include <vector>
#include <numeric>
#include <common/log_helper.h>

template<typename T>
T Product(const std::vector<T> &v) {
    return static_cast<T> (accumulate(v.begin(), v.end(), 1, std::multiplies<T>()));
}

using css = const std::string;

#endif /* DNNLIBRARY_HELPER_H */
