#ifndef DNN_DATA_TYPES_H
#define DNN_DATA_TYPES_H

#include <common/optional.h>

#include <common/expected.hpp>

namespace dnn {
using nonstd::bad_optional_access;
using nonstd::make_optional;
using nonstd::nullopt;
using nonstd::nullopt_t;
using nonstd::optional;
using tl::expected;
using tl::make_unexpected;
using Unit = tl::monostate;
using tl::unexpected;

#define TRY(x)            \
    const auto ret = (x); \
    if (!ret) {           \
        return ret;       \
    }
}  // namespace dnn

#endif /* DNN_DATA_TYPES_H */
