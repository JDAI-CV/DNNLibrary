//
// Created by daquexian on 2017/11/8.
// A wrapper for ANeuralNetworksExecution
//

#ifndef NNAPIEXAMPLE_MODEL_H
#define NNAPIEXAMPLE_MODEL_H

#include <memory>
#include <vector>

#include <NeuralNetworksWrapper.h>
#include <common/Shaper.h>
#include <common/StrKeyMap.h>

class Model {
    friend class ModelBuilder;

   private:
    ANeuralNetworksModel *model_;
    ANeuralNetworksCompilation *compilation_;
    ANeuralNetworksExecution *execution_;
    ANeuralNetworksMemory *memory_;
    unsigned char *data_;
    size_t data_size_;
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
    void SetInputBuffer(const int32_t index, const void *buffer, const size_t elemsize);
    void PrepareForExecution();
    void PredictAfterSetInputBuffer();
    bool prepared_for_exe_;
    Model() = default;

   public:
    template <typename T>
    void Predict(const std::vector<T> &input) {
        DNN_ASSERT_EQ(input.size, GetSize(GetInputs()[0]));
        // const_cast is a ugly workaround, vector<const T*> causes strange errors
        Predict<T>({const_cast<T *>(input.data())});
    }
    template <typename T>
    void Predict(const std::vector<std::vector<T>> &inputs) {
        std::vector<T *> input_ptrs;
        for (size_t i = 0; i < inputs.size(); i++) {
            auto &input = inputs[i];
            DNN_ASSERT_EQ(input.size(), GetSize(GetInputs()[i]));
            // const_cast is a ugly workaround, vector<const T*> causes strange errors
            input_ptrs.push_back(const_cast<T *>(input.data()));
        }
        Predict<T>(input_ptrs);
    }
    template <typename T>
    void Predict(const T *input) {
        Predict<T>(std::vector<T*>{input});
    }
    template <typename T>
    void Predict(const std::vector<T *> &inputs) {
        DNN_ASSERT_EQ(inputs.size(), GetInputs().size());
        if (!prepared_for_exe_) PrepareForExecution();
        for (size_t i = 0; i < inputs.size(); i++) {
            SetInputBuffer(i, inputs[i]);
        }
        PredictAfterSetInputBuffer();
    }

    ~Model();
    void SetOutputBuffer(const int32_t index, float *buffer);
    void SetOutputBuffer(const int32_t index, uint8_t *buffer);
    void SetOutputBuffer(const int32_t index, char *buffer);
    void SetOutputBuffer(const int32_t index, void *buffer, const size_t elemsize);
    size_t GetSize(const std::string &name);
    Shaper::Shape GetShape(const std::string &name);
    std::vector<std::string> GetInputs();
    std::vector<std::string> GetOutputs();
};

#endif  // NNAPIEXAMPLE_MODEL_H
