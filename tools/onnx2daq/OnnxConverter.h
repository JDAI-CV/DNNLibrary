#include <onnx/onnx.pb.h>
#include <glog/logging.h>
#include <common/daq_generated.h>
#include <common/helper.h>
#include <common/StrKeyMap.h>
#include <common/Shaper.h>

class OnnxConverter {
private:
    Shaper shaper_;

    template <typename T>
    struct Tensor {
        std::vector<T> data;
        Shaper::Shape shape;
    };

    using FTensor = Tensor<float>;

    enum class FuseCode {
        FUSED_NONE,
        FUSED_RELU,
        FUSED_RELU1,
        FUSED_RELU6
    };

    std::map<std::string, std::string> name_map_;

    std::string m(const std::string &str);

    flatbuffers::FlatBufferBuilder builder_;

    std::vector<std::string> operands_;
    StrKeyMap<FTensor> nnapi_tensors_;
    StrKeyMap<FTensor> onnx_tensors_;
    std::vector<flatbuffers::Offset<DNN::Layer>> layers_;

    std::vector<flatbuffers::Offset<DNN::Tensor>> tensors_;

    DNN::FuseCode ConvertFuseCodeType(FuseCode fuse_code);
    std::pair<std::optional<std::string>, FuseCode> FindActivation(const ONNX_NAMESPACE::ModelProto &model_proto, const ONNX_NAMESPACE::NodeProto &node);

    void AddConv(const std::string &input_name, const std::vector<int> &strides, const std::vector<int> &pads, 
            const std::vector<int> &dilations, int group, 
            const std::pair<std::optional<std::string>, FuseCode>& activation,
            const std::string &ori_weight_name, const std::optional<std::string> &bias_name, const std::string &output_name);

    /**
     * onnx: [filter_out_channel, filter_in_channel / group, height, width]
     * nnapi: [1, height, width, depth_out]
     */
    template <typename T>
    Tensor<T> OnnxToNnapiDw(const Tensor<T> &src) {
        Tensor<T> dest;
        dest.data.resize(Product(src.shape));
        // t for total
        auto out_t = src.shape[0], in_t = src.shape[1], h_t = src.shape[2], w_t = src.shape[3];
        CHECK_EQ(in_t, 1u);
        for (uint32_t out = 0; out < out_t; out++) {
            for (uint32_t in = 0; in < in_t; in++) {
                for (uint32_t h = 0; h < h_t; h++) {
                    for (uint32_t w = 0; w < w_t; w++) {
                        auto onnx_idx = out * in_t * h_t * w_t + in * h_t * w_t + h * w_t + w;
                        auto nnapi_idx = h * w_t * out_t + w * out_t + out;
                        dest.data[nnapi_idx] = src.data[onnx_idx];
                    }
                }
            }
        }
        dest.shape = {in_t, h_t, w_t, out_t};
        return dest;
    }

    /**
     * onnx: [filter_out_channel, filter_in_channel, height, width]
     * nnapi: [depth_out, height, width, depth_in]
     */
    template <typename T>
    Tensor<T> OnnxToNnapiVanilla(const Tensor<T> &src) {
        Tensor<T> dest;
        dest.data.resize(Product(src.shape));
        // t for total
        auto out_t = src.shape[0], in_t = src.shape[1], h_t = src.shape[2], w_t = src.shape[3];
        for (uint32_t out = 0; out < out_t; out++) {
            for (uint32_t in = 0; in < in_t; in++) {
                for (uint32_t h = 0; h < h_t; h++) {
                    for (uint32_t w = 0; w < w_t; w++) {
                        auto onnx_idx = out * in_t * h_t * w_t + in * h_t * w_t + h * w_t + w;
                        auto nnapi_idx = out * h_t * w_t * in_t + h * w_t * in_t + w * in_t + in;
                        dest.data[nnapi_idx] = src.data[onnx_idx];
                    }
                }
            }
        }
        dest.shape = {out_t, h_t, w_t, in_t};
        return dest;
    }

public:
    void Convert(const ONNX_NAMESPACE::ModelProto &model, const std::string &filepath);
};
