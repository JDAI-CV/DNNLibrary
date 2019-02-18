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

// Make a FOREACH macro
#define FE_1(WHAT, X) WHAT(X) 
#define FE_2(WHAT, X, ...) WHAT(X)FE_1(WHAT, __VA_ARGS__)
#define FE_3(WHAT, X, ...) WHAT(X)FE_2(WHAT, __VA_ARGS__)
#define FE_4(WHAT, X, ...) WHAT(X)FE_3(WHAT, __VA_ARGS__)
#define FE_5(WHAT, X, ...) WHAT(X)FE_4(WHAT, __VA_ARGS__)
#define FE_6(WHAT, X, ...) WHAT(X)FE_5(WHAT, __VA_ARGS__)
#define FE_7(WHAT, X, ...) WHAT(X)FE_6(WHAT, __VA_ARGS__)
//... repeat as needed

#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,NAME,...) NAME 
#define FOR_EACH(action,...) \
  GET_MACRO(__VA_ARGS__,FE_7,FE_6,FE_5,FE_4,FE_3,FE_2,FE_1)(action,__VA_ARGS__)

#define FORZS(var, end, step) for (auto var = decltype(end){0}; var < end; var += (step))

#define FORZ(var, end) for (auto var = decltype(end){0}; var < end; var++)

#define FOR(var, start, end) for (auto var = decltype(end){start}; var < end; var++)

#define STR(a) #a
#define XSTR(a) STR(a)

#define PNT_STR(var) << XSTR(var) << " = " << (var) << ", "
#define PNT(...) LOG(INFO) FOR_EACH(PNT_STR, __VA_ARGS__);

#define DNN_ASSERT(condition, note) \
    if (!(condition)) { \
        std::stringstream ss;   \
        ss << std::string(XSTR(condition)) << std::string(" is not satisfied on ") << std::to_string(__LINE__) << " of " << __FILE__ << note; \
        LOG(INFO) << ss.str();   \
        throw std::runtime_error(ss.str()); \
    }

#endif /* DNNLIBRARY_HELPER_H */
