#include <common/Shaper.h>
#include <common/StrKeyMap.h>
#include <common/daq_generated.h>
#include <common/helper.h>
#include <glog/logging.h>
#include <onnx/onnx_pb.h>
#include "optional.h"

class OnnxConverter {
   private:
    Shaper shaper_;

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

    enum class FuseCode { FUSED_NONE, FUSED_RELU, FUSED_RELU1, FUSED_RELU6 };

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
        nonstd::optional<int32_t> zero_point;
        Type type;
    };
    StrKeyMap<QuantInfo> quant_infos_;

    std::map<std::string, std::string> name_map_;

    std::string m(const std::string &str);

    ONNX_NAMESPACE::ModelProto model_proto_;
    flatbuffers::FlatBufferBuilder builder_;
    std::vector<std::string> skipped_act_;
    std::vector<std::string> dequantize_after_;

    std::vector<std::string> operands_;
    StrKeyMap<Tensor> nnapi_tensors_;
    StrKeyMap<Tensor> onnx_tensors_;
    std::vector<flatbuffers::Offset<DNN::Layer>> layers_;

    std::vector<flatbuffers::Offset<DNN::Tensor>> tensors_;

    DNN::FuseCode ConvertFuseCodeType(FuseCode fuse_code);
    std::pair<nonstd::optional<std::string>, FuseCode> FindActivation(
        const ONNX_NAMESPACE::ModelProto &model_proto, css &output_name);
    void CreateTensorFb(const Tensor &tensor, const DNN::DataType &data_type);
    void CreateTensorFb(const std::string &name, const Tensor &tensor);
    void CreateTensorFb(const std::string &name, const Tensor &tensor,
                        const DNN::DataType &data_type);
    std::vector<flatbuffers::Offset<flatbuffers::String>> FbStrVector(
        const std::vector<std::string> &std_str_vector);

    void HandleInitializer();
    std::vector<flatbuffers::Offset<DNN::Input>> GetInputOfOnnxModel();
    void ReadTableFile(css &table_file);
    std::vector<flatbuffers::Offset<DNN::QuantInfo>> ConvertQuantInfosToFbs();

    void AddConv(const std::string &input_name, const std::vector<int> &strides,
                 const std::vector<int> &pads,
                 const std::vector<int> &dilations, int group,
                 const std::string &ori_weight_name,
                 const nonstd::optional<std::string> &bias_name,
                 const std::string &output_name);
    void AddLayerPool(css &op, css &input_name,
                      const std::vector<int> &kernel_shape,
                      const std::vector<int> &pads,
                      const std::vector<int> &strides, css &output_name);
    void AddLayerRelu(css &input_name, css &output_name);
    void AddLayerGemm(css &input_name, css &weight_name,
                      nonstd::optional<std::string> bias_name, const int transA,
                      const int transB, const float alpha, const float beta,
                      css &output_name);
    // axis here is for onnx nchw
    void AddLayerConcat(const std::vector<std::string> &inputs,
                        css &output_name, const int axis);
    void AddLayerDequantize(css &input_name, css &output_name);
    void SetIdentity(css &input_name, css &output_name);
    // OnnxConverter auto generated methods start
    void AddLayerConvImpl(const std::string &input, const std::string &weight,
                          const nonstd::optional<std::string> &bias,
                          const std::vector<int32_t> &pads,
                          const std::vector<int32_t> &strides,
                          const std::string &output);
    void AddLayerAvePoolImpl(const std::string &input,
                             const std::vector<int32_t> &kernel_shape,
                             const std::vector<int32_t> &pads,
                             const std::vector<int32_t> &strides,
                             const std::string &output);
    void AddLayerMaxPoolImpl(const std::string &input,
                             const std::vector<int32_t> &kernel_shape,
                             const std::vector<int32_t> &pads,
                             const std::vector<int32_t> &strides,
                             const std::string &output);
    void AddLayerReLU(const std::string &input, const std::string &output);
    void AddLayerSoftmax(const std::string &input, const std::string &output);
    void AddLayerFC(const std::string &input, const std::string &weight,
                    const nonstd::optional<std::string> &bias,
                    const std::string &output);
    void AddLayerAdd(const std::string &input1, const std::string &input2,
                     const std::string &output);
    void AddLayerConcatImpl(const std::vector<std::string> &inputs,
                            int32_t axis, const std::string &output);
    void AddLayerDepthwiseConvImpl(const std::string &input,
                                   const std::string &weight,
                                   const nonstd::optional<std::string> &bias,
                                   const std::vector<int32_t> &pads,
                                   const std::vector<int32_t> &strides,
                                   int32_t depth_multiplier,
                                   const std::string &output);
    void AddLayerBatchToSpaceND(const std::string &input,
                                const std::vector<int32_t> &block_sizes,
                                const std::string &output);
    void AddLayerSpaceToBatchND(const std::string &input,
                                const std::vector<int32_t> &block_sizes,
                                const std::vector<int32_t> &pads,
                                const std::string &output);
    void AddLayerStridedSlice(const std::string &input,
                              const std::vector<int32_t> &starts,
                              const std::vector<int32_t> &ends,
                              const std::vector<int32_t> &strides,
                              int32_t begin_mask, int32_t end_mask,
                              int32_t shrink_axis_mask,
                              const std::string &output);
    void AddLayerMul(const std::string &input1, const std::string &input2,
                     const std::string &output);
    void AddLayerAdd(const std::string &input, float scalar,
                     const std::string &output);
    void AddLayerMul(const std::string &input, float scalar,
                     const std::string &output);
    void AddLayerLRN(const std::string &input, int32_t size, float bias,
                     float alpha, float beta, int32_t dim,
                     const std::string &output);
    // OnnxConverter auto generated methods end

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

   public:
    void Convert(const ONNX_NAMESPACE::ModelProto &model,
                 const std::string &filepath, const css &table_file = "");
};
