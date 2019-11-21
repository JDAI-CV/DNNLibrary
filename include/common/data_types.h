#include <common/optional.h>
#include <common/expected.hpp>

namespace dnn {
    using nonstd::bad_optional_access;
    using nonstd::optional;
    using nonstd::make_optional;
    using nonstd::nullopt;
    using nonstd::nullopt_t;
    using tl::expected;
    using tl::make_unexpected;
    using Unit = tl::monostate;
    using tl::unexpected;
}
