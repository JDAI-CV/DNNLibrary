//
// Created by daquexian on 2017/11/8.
// A wrapper for ANeuralNetworksExecution
//

#ifndef NNAPIEXAMPLE_MODEL_H
#define NNAPIEXAMPLE_MODEL_H

#include <common/Shaper.h>
#include <common/StrKeyMap.h>
#include <dnnlibrary/NeuralNetworksWrapper.h>

#include <memory>
#include <vector>

namespace dnn {
class Model {
    friend class ModelBuilder;

   private:
    ANeuralNetworksModel *model_ = nullptr;
    ANeuralNetworksCompilation *compilation_ = nullptr;
    ANeuralNetworksExecution *execution_ = nullptr;
    ANeuralNetworksMemory *memory_ = nullptr;
    unsigned char *data_ = nullptr;
    size_t data_size_ = 0;
    std::vector<std::unique_ptr<uint8_t[]>> uint8_buf_pointers_;
    std::vector<std::unique_ptr<int8_t[]>> int8_buf_pointers_;
    std::vector<std::unique_ptr<float[]>> float_buf_pointers_;
    std::vector<std::unique_ptr<int32_t[]>> int32_buf_pointers_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    Shaper shaper_;
    void AddInput(const std::string &name, const Shaper::Shape &shape);
    void AddOutput(const std::string &name, const Shaper::Shape &shape);
    void SetInputBuffer(const int32_t index, const float *buffer);
    void SetInputBuffer(const int32_t index, const uint8_t *buffer);
    void SetInputBuffer(const int32_t index, const void *buffer,
                        const size_t elemsize);
    void PrepareForExecution();
    void PredictAfterSetInputBuffer();
    bool prepared_for_exe_ = false;
    const NnApi *nnapi_ = nullptr;
    Model();

   public:
    template <typename T>
    void Predict(const std::vector<T> &input);
    template <typename T>
    void Predict(const std::vector<std::vector<T>> &inputs);
    template <typename T>
    void Predict(const T *input);
    template <typename T>
    void Predict(const std::vector<T *> &inputs);

    ~Model();
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    void SetOutputBuffer(const int32_t index, float *buffer);
    void SetOutputBuffer(const int32_t index, uint8_t *buffer);
    void SetOutputBuffer(const int32_t index, char *buffer);
    void SetOutputBuffer(const int32_t index, void *buffer,
                         const size_t elemsize);
    size_t GetSize(const std::string &name);
    Shaper::Shape GetShape(const std::string &name);
    std::vector<std::string> GetInputs();
    std::vector<std::string> GetOutputs();
};
}  // namespace dnn

#endif  // NNAPIEXAMPLE_MODEL_H
