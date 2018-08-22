#ifndef DNN_SHAPER_H
#define DNN_SHAPER_H

#include <vector>
#include <string>

#include <common/log_helper.h>
#include <common/StrKeyMap.h>

class Shaper {
public:
    using Shape = std::vector<uint32_t>;

    void Conv(const std::string &input_name, int32_t strideX, int32_t strideY, int32_t dilationX, int32_t dilationY,
                      int32_t paddingLeft, int32_t paddingRight,
                      int32_t paddingTop, int32_t paddingBottom, const std::string &weight_name,
                      const std::string &output_name);
    void DepthwiseConv(const std::string &input_name, int32_t strideX, int32_t strideY, int32_t dilationX, int32_t dilationY,
                      int32_t paddingLeft, int32_t paddingRight,
                      int32_t paddingTop, int32_t paddingBottom, const std::string &weight_name,
                      const std::string &output_name);
    void StridedSlice(const std::string &input_name, const std::vector<int32_t> &starts, const std::vector<int32_t> &ends,
                              const std::vector<int32_t> &strides, int32_t beginMask, int32_t endMask,
                              int32_t shrinkAxisMask, const std::string &output_name);
    void Pool(const std::string &input_name, int32_t strideX, int32_t strideY,
                                          int32_t paddingLeft, int32_t paddingRight,
                                          int32_t paddingTop, int32_t paddingBottom, int32_t height, int32_t width,
                                          const std::string &output_name);
    void Softmax(const std::string &input_name, const std::string &output_name);
    void Relu(const std::string &input_name, const std::string &output_name);
    void Concat(const std::vector<std::string> &input_names, uint32_t axis, const std::string &output_name);
    void LRN(const std::string &input_name, const std::string &output_name);
    void FC(const std::string &input_name, const std::string &weight_name, const std::string &output_name);
    void Eltwise(const std::string &input1_name, const std::string &input2_name, const std::string &output_name);
    void Eltwise(const std::string &input1_name, const std::string &output_name);
    void BatchToSpace(const std::string &input_name, const std::vector<int32_t> &block_sizes,
        const std::string &output_name);
    void SpaceToBatch(const std::string &input_name, const std::vector<int32_t> &block_sizes,
        const std::vector<int32_t> &pads, const std::string &output_name);
    void AddShape(const std::string &name, const Shape &shape);
    void clear();

    inline const Shape& operator[](const std::string &key) {
        return shape_map_.at(key);
    }
    friend std::ostream &operator<<(std::ostream &os, const Shaper &shaper) {
        for (const auto &p : shaper.shape_map_) {
            os << (p.first + ": ") << p.second << std::endl;
        }
        return os;
    }
private:
    StrKeyMap<Shape> shape_map_;
};

#endif
