#include <common/Shaper.h>
#include <common/StrKeyMap.h>
#include <common/daq_generated.h>
#include <common/data_types.h>
#include <glog/logging.h>
#include <onnx/onnx_pb.h>

namespace dnn {
class OnnxConverter {
   private:
    Shaper shaper_;

    struct Tensor {
        enum class DataType { FLOAT32, UINT8, INT32 };
        std::string name;
        std::vector<char> data;
        Shaper::Shape shape;
        DataType data_type;
        Tensor() = default;
        Tensor(Tensor &&) = default;
        Tensor(const Tensor &) = default;
        Tensor &operator=(const Tensor &) = default;
        Tensor(const std::string &name, const std::vector<char> &data,
               const Shaper::Shape &shape, const DataType &data_type)
            : name(name), data(data), shape(shape), data_type(data_type) {
        }
        Tensor(const std::string &name, const std::vector<float> &float_data,
               const Shaper::Shape &shape) {
            const char *char_ptr =
                reinterpret_cast<const char *>(float_data.data());
            this->name = name;
            this->data = std::vector<char>(
                char_ptr, char_ptr + float_data.size() * sizeof(float));
            this->shape = shape;
            this->data_type = Tensor::DataType::FLOAT32;
        }
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

    struct QuantInfo {
        enum class Type {
            QUANT8_SYMM,
            QUANT8_ASYMM,
            INT32,
            QUANT8_SYMM_PER_CHANNEL,
            QUANT16_SYMM,
            QUANT16_ASYMM
        };
        std::vector<float> scales;
        dnn::optional<int32_t> zero_point;
        Type type;
    };
    StrKeyMap<QuantInfo> quant_infos_;

    std::map<std::string, std::string> name_map_;

    std::string m(const std::string &str) const;

    ONNX_NAMESPACE::ModelProto model_proto_;
    flatbuffers::FlatBufferBuilder builder_;
    std::vector<int> skipped_act_;
    std::vector<std::string> dequantize_after_;

    std::vector<std::string> operands_;
    StrKeyMap<Tensor> nnapi_tensors_;
    StrKeyMap<Tensor> onnx_tensors_;
    std::vector<flatbuffers::Offset<DNN::Layer>> layers_;

    std::vector<flatbuffers::Offset<DNN::Tensor>> tensors_;

    DNN::FuseCode ConvertFuseCodeType(FuseCode fuse_code);
    std::pair<dnn::optional<std::pair<int, ONNX_NAMESPACE::NodeProto>>,
              FuseCode>
    FindActivation(const ONNX_NAMESPACE::ModelProto &model_proto,
                   const std::string &output_name);
    void CreateTensorFb(const Tensor &tensor, const DNN::DataType &data_type);
    void CreateTensorFb(const std::string &name, const Tensor &tensor);
    void CreateTensorFb(const std::string &name, const Tensor &tensor,
                        const DNN::DataType &data_type);
    std::vector<flatbuffers::Offset<flatbuffers::String>> FbStrVector(
        const std::vector<std::string> &std_str_vector);

    void HandleInitializer();
    std::vector<flatbuffers::Offset<DNN::Input>> GetInputOfOnnxModel();
    std::vector<flatbuffers::Offset<flatbuffers::String>>
    GetOutputOfOnnxModel();
    void ReadTableFile(const std::string &table_file);
    std::vector<flatbuffers::Offset<DNN::QuantInfo>> ConvertQuantInfosToFbs();

    std::pair<bool, std::string> IsNodeSupported(
        const ONNX_NAMESPACE::ModelProto &model_proto,
        const ONNX_NAMESPACE::NodeProto &node_proto) const;

    void SetIdentity(const std::string &input_name,
                     const std::string &output_name);
    // OnnxConverter auto generated methods start
    void WriteDaqLayer_CONV_2D(
        const std::string &input, const std::string &weight,
        const dnn::optional<std::string> &bias, int32_t padding_left,
        int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
        int32_t stride_x, int32_t stride_y, FuseCode fuse_code, bool nchw,
        int32_t dilation_x, int32_t dilation_y, const std::string &output);
    void WriteDaqLayer_AVERAGE_POOL_2D(
        const std::string &input, int32_t padding_left, int32_t padding_right,
        int32_t padding_top, int32_t padding_bottom, int32_t stride_x,
        int32_t stride_y, int32_t kernel_width, int32_t kernel_height,
        FuseCode fuse_code, const std::string &output);
    void WriteDaqLayer_MAX_POOL_2D(const std::string &input,
                                   int32_t padding_left, int32_t padding_right,
                                   int32_t padding_top, int32_t padding_bottom,
                                   int32_t stride_x, int32_t stride_y,
                                   int32_t kernel_width, int32_t kernel_height,
                                   FuseCode fuse_code,
                                   const std::string &output);
    void WriteDaqLayer_RELU(const std::string &input,
                            const std::string &output);
    void WriteDaqLayer_SOFTMAX(const std::string &input, float beta,
                               const std::string &output);
    void WriteDaqLayer_FULLY_CONNECTED(const std::string &input,
                                       const std::string &weight,
                                       const dnn::optional<std::string> &bias,
                                       FuseCode fuse_code,
                                       const std::string &output);
    void WriteDaqLayer_ADD(const std::string &input1, const std::string &input2,
                           FuseCode fuse_code, const std::string &output);
    void WriteDaqLayer_CONCATENATION(const std::vector<std::string> &inputs,
                                     int32_t axis, const std::string &output);
    void WriteDaqLayer_DEPTHWISE_CONV_2D(
        const std::string &input, const std::string &weight,
        const dnn::optional<std::string> &bias, int32_t padding_left,
        int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
        int32_t stride_x, int32_t stride_y, int32_t depth_multiplier,
        FuseCode fuse_code, const std::string &output);
    void WriteDaqLayer_BATCH_TO_SPACE_ND(
        const std::string &input, const std::vector<int32_t> &block_sizes,
        const std::string &output);
    void WriteDaqLayer_SPACE_TO_BATCH_ND(
        const std::string &input, const std::vector<int32_t> &block_sizes,
        const std::vector<int32_t> &pads, const std::string &output);
    void WriteDaqLayer_STRIDED_SLICE(const std::string &input,
                                     const std::vector<int32_t> &starts,
                                     const std::vector<int32_t> &ends,
                                     const std::vector<int32_t> &strides,
                                     int32_t begin_mask, int32_t end_mask,
                                     int32_t shrink_axis_mask,
                                     const std::string &output);
    void WriteDaqLayer_MUL(const std::string &input1, const std::string &input2,
                           FuseCode fuse_code, const std::string &output);
    void WriteDaqLayer_DEQUANTIZE(const std::string &input,
                                  const std::string &output);
    void WriteDaqLayer_LOCAL_RESPONSE_NORMALIZATION(const std::string &input,
                                                    int32_t radius, float bias,
                                                    float alpha, float beta,
                                                    const std::string &output);
    void WriteDaqLayer_TANH(const std::string &input,
                            const std::string &output);
    void WriteDaqLayer_FLOOR(const std::string &input,
                             const std::string &output);
    void WriteDaqLayer_LOGISTIC(const std::string &input,
                                const std::string &output);
    void WriteDaqLayer_PRELU(const std::string &input, const std::string &alpha,
                             const std::string &output);
    void WriteDaqLayer_POW(const std::string &input, const std::string &exp,
                           const std::string &output);
    void WriteDaqLayer_NEG(const std::string &input, const std::string &output);
    void WriteDaqLayer_MINIMUM(const std::string &input1,
                               const std::string &input2,
                               const std::string &output);
    void WriteDaqLayer_MAXIMUM(const std::string &input1,
                               const std::string &input2,
                               const std::string &output);
    void WriteDaqLayer_LOG(const std::string &input, const std::string &output);
    void WriteDaqLayer_ABS(const std::string &input, const std::string &output);
    void WriteDaqLayer_EXP(const std::string &input, const std::string &output);
    void WriteDaqLayer_SUB(const std::string &input1, const std::string &input2,
                           FuseCode fuse_code, const std::string &output);
    // OnnxConverter auto generated methods end

    /**
     * transpose axes to [1, 2, 3, 0]
     * for onnx dw conv weight to nnapi dw conv weight
     * onnx: [filter_out_channel, filter_in_channel / group, height, width]
     * nnapi: [1, height, width, depth_out]
     */
    Tensor OnnxToNnapiAxes1230(const Tensor &src);

    /**
     * transpose axes to [0, 2, 3, 1]
     * for nchw (onnx) -> nhwc (nnapi)
     * or onnx conv weight to nnapi conv (not dw conv) weight:
     * onnx: [filter_out_channel, filter_in_channel, height, width]
     * nnapi: [depth_out, height, width, depth_in]
     */
    Tensor OnnxToNnapiAxes0231(const Tensor &src);

    /**
     * Just return the same tensor
     */
    Tensor OnnxToNnapiIdentity(const Tensor &src);

    void Clear();

   public:
    expected<std::vector<std::vector<int>>, std::string> GetSupportedNodes(
        ONNX_NAMESPACE::ModelProto model_proto);
    void Convert(const std::string &model_str, const std::string &filepath,
                 const std::string &table_file = "");
    void Convert(const ONNX_NAMESPACE::ModelProto &model,
                 const std::string &table_file = "");
    void Save(const std::string &filename);
    std::unique_ptr<uint8_t[]> GetBuf();
};
}  // namespace dnn
