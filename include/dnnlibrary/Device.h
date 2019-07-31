#include <string>

namespace dnn {
struct Device {
    const std::string name;
    const int64_t feature_level;
    const int type;
    const std::string version;
};
}  // namespace dnn
