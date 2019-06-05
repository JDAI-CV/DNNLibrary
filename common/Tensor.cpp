#include <common/Tensor.h>

#include <common/helper.h>

namespace dnn {
/**
 * onnx: [filter_out_channel, filter_in_channel / group, height, width]
 * nnapi: [1, height, width, depth_out]
 */
Tensor OnnxToNnapiDwConvWeight(
    const Tensor &src) {
    Tensor dest = src;
    size_t elemsize = 0;
    if (src.data_type == Tensor::DataType::UINT8) {
        elemsize = 1;
    } else if (src.data_type == Tensor::DataType::FLOAT32) {
        elemsize = 4;
    }
    dest.data.resize(Product(src.shape) * elemsize);
    // t for total
    auto out_t = src.shape[0], in_t = src.shape[1], h_t = src.shape[2],
         w_t = src.shape[3];
    DNN_ASSERT_EQ(in_t, 1u);
    for (uint32_t out = 0; out < out_t; out++) {
        for (uint32_t in = 0; in < in_t; in++) {
            for (uint32_t h = 0; h < h_t; h++) {
                for (uint32_t w = 0; w < w_t; w++) {
                    auto onnx_idx =
                        out * in_t * h_t * w_t + in * h_t * w_t + h * w_t + w;
                    auto nnapi_idx = h * w_t * out_t + w * out_t + out;
                    FORZ(i, elemsize) {
                        dest.data[elemsize * nnapi_idx + i] =
                            src.data[elemsize * onnx_idx + i];
                    }
                }
            }
        }
    }
    dest.shape = {in_t, h_t, w_t, out_t};
    return dest;
}

Tensor OnnxToNnapiVanillaConvWeight(
    const Tensor &src) {
    Tensor dest = src;
    size_t elemsize = 0;
    if (src.data_type == Tensor::DataType::UINT8) {
        elemsize = 1;
    } else if (src.data_type == Tensor::DataType::FLOAT32) {
        elemsize = 4;
    }
    dest.data.resize(Product(src.shape) * elemsize);
    // t for total
    auto out_t = src.shape[0], in_t = src.shape[1], h_t = src.shape[2],
         w_t = src.shape[3];
    for (uint32_t out = 0; out < out_t; out++) {
        for (uint32_t in = 0; in < in_t; in++) {
            for (uint32_t h = 0; h < h_t; h++) {
                for (uint32_t w = 0; w < w_t; w++) {
                    auto onnx_idx =
                        out * in_t * h_t * w_t + in * h_t * w_t + h * w_t + w;
                    auto nnapi_idx =
                        out * h_t * w_t * in_t + h * w_t * in_t + w * in_t + in;
                    FORZ(i, elemsize) {
                        dest.data[elemsize * nnapi_idx + i] =
                            src.data[elemsize * onnx_idx + i];
                    }
                }
            }
        }
    }
    dest.shape = {out_t, h_t, w_t, in_t};
    return dest;
}

Tensor OnnxToNnapiIdentity(const Tensor &src) {
    return src;
}


}
