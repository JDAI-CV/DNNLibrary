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

#define STR(a) #a
#define XSTR(a) STR(a)

#define DNN_ASSERT(condition, note) \
    if (!(condition)) { \
        std::stringstream ss;   \
        ss << std::string(XSTR(condition)) << std::string(" is not satisfied on ") << std::to_string(__LINE__) << " of " << __FILE__ << note; \
        LOG(INFO) << ss.str();   \
        throw std::runtime_error(ss.str()); \
    }

#endif /* DNNLIBRARY_HELPER_H */
