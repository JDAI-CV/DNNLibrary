#ifndef DNNLIBRARY_HELPER_H
#define DNNLIBRARY_HELPER_H

#include <common/log_helper.h>
#include <glog/logging.h>

#include <numeric>
#include <vector>

template <typename T>
T Product(const std::vector<T> &v) {
    return static_cast<T>(
        accumulate(v.begin(), v.end(), 1, std::multiplies<T>()));
}

using css = const std::string;

// Make a FOREACH macro
#define FE_1(WHAT, X) WHAT(X)
#define FE_2(WHAT, X, ...) WHAT(X) FE_1(WHAT, __VA_ARGS__)
#define FE_3(WHAT, X, ...) WHAT(X) FE_2(WHAT, __VA_ARGS__)
#define FE_4(WHAT, X, ...) WHAT(X) FE_3(WHAT, __VA_ARGS__)
#define FE_5(WHAT, X, ...) WHAT(X) FE_4(WHAT, __VA_ARGS__)
#define FE_6(WHAT, X, ...) WHAT(X) FE_5(WHAT, __VA_ARGS__)
#define FE_7(WHAT, X, ...) WHAT(X) FE_6(WHAT, __VA_ARGS__)
#define FE_8(WHAT, X, ...) WHAT(X) FE_7(WHAT, __VA_ARGS__)
#define FE_9(WHAT, X, ...) WHAT(X) FE_8(WHAT, __VA_ARGS__)
#define FE_10(WHAT, X, ...) WHAT(X) FE_9(WHAT, __VA_ARGS__)
#define FE_11(WHAT, X, ...) WHAT(X) FE_10(WHAT, __VA_ARGS__)
#define FE_12(WHAT, X, ...) WHAT(X) FE_11(WHAT, __VA_ARGS__)
#define FE_13(WHAT, X, ...) WHAT(X) FE_12(WHAT, __VA_ARGS__)
#define FE_14(WHAT, X, ...) WHAT(X) FE_13(WHAT, __VA_ARGS__)
#define FE_15(WHAT, X, ...) WHAT(X) FE_14(WHAT, __VA_ARGS__)
#define FE_16(WHAT, X, ...) WHAT(X) FE_15(WHAT, __VA_ARGS__)
#define FE_17(WHAT, X, ...) WHAT(X) FE_16(WHAT, __VA_ARGS__)
#define FE_18(WHAT, X, ...) WHAT(X) FE_17(WHAT, __VA_ARGS__)
#define FE_19(WHAT, X, ...) WHAT(X) FE_18(WHAT, __VA_ARGS__)
#define FE_20(WHAT, X, ...) WHAT(X) FE_19(WHAT, __VA_ARGS__)
//... repeat as needed

#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, \
                  _15, _16, _17, _18, _19, _20, NAME, ...)                     \
    NAME
#define FOR_EACH(action, ...)                                                 \
    GET_MACRO(__VA_ARGS__, FE_20, FE_19, FE_18, FE_17, FE_16, FE_15, FE_14,   \
              FE_13, FE_12, FE_11, FE_10, FE_9, FE_8, FE_7, FE_6, FE_5, FE_4, \
              FE_3, FE_2, FE_1)                                               \
    (action, __VA_ARGS__)

#define FORZS(var, end, step) \
    for (auto var = decltype(end){0}; var < end; var += (step))

#define FORZ(var, end) for (auto var = decltype(end){0}; var < end; var++)

#define FOR(var, start, end) \
    for (auto var = decltype(end){start}; var < end; var++)

#define STR(a) #a
#define XSTR(a) STR(a)

#define PNT_STR(s) << s << " "
#define PNT_VAR(var) << XSTR(var) << " = " << (var) << ", "
#define PNT_TO(stream, ...) stream FOR_EACH(PNT_VAR, __VA_ARGS__);
#define PNT(...) PNT_TO(LOG(INFO), __VA_ARGS__)

#define DNN_ASSERT(condition, ...)                \
    if (!(condition)) {                           \
        std::stringstream ss;                     \
        ss << std::string(XSTR(condition))        \
           << std::string(" is not satisfied! ")  \
                  FOR_EACH(PNT_STR, __VA_ARGS__); \
        LOG(INFO) << ss.str();                    \
        throw std::runtime_error(ss.str());       \
    }

#define DNN_ASSERT_EQ(actual, expected)                                \
    DNN_ASSERT((actual) == (expected), XSTR(actual), "=", actual,      \
               ", the expected value is", XSTR(expected), "(which is", \
               expected, ")")

#endif /* DNNLIBRARY_HELPER_H */
