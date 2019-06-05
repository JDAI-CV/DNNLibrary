#include <string>
#include <vector>

#include <common/Shaper.h>

namespace dnn {
struct Tensor {
    enum class DataType { FLOAT32, UINT8, INT32 };
    std::string name;
    std::vector<char> data;
    Shaper::Shape shape;
    DataType data_type;
    const std::vector<float> float_data() const {
        std::vector<float> float_vec(data.size() / 4);
        memcpy(&float_vec[0], &data[0], data.size());
        return float_vec;
    }
    const std::vector<uint8_t> uint8_data() const {
        std::vector<uint8_t> uint8_vec(data.size());
        memcpy(&uint8_vec[0], &data[0], data.size());
        return uint8_vec;
    }
    const std::vector<int32_t> int32_data() const {
        std::vector<int32_t> int32_vec(data.size() / 4);
        memcpy(&int32_vec[0], &data[0], data.size());
        return int32_vec;
    }
};

/**
 * onnx: [filter_out_channel, filter_in_channel / group, height, width]
 * nnapi: [1, height, width, depth_out]
 */
Tensor OnnxToNnapiDwConvWeight(const Tensor &src);

/**
 * onnx: [filter_out_channel, filter_in_channel, height, width]
 * nnapi: [depth_out, height, width, depth_in]
 */
Tensor OnnxToNnapiVanillaConvWeight(const Tensor &src);

/**
 * Just return the same tensor
 */
Tensor OnnxToNnapiIdentity(const Tensor &src);

}  // namespace dnn
