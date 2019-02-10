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
    void SetInputBuffer(int32_t index, float *buffer);
    void SetInputBuffer(int32_t index, uint8_t *buffer);
    void SetInputBuffer(int32_t index, void *buffer, size_t elemsize);
    void PrepareForExecution();
    bool prepared_for_exe_;
    Model() = default;

   public:
    template <typename T>
    void Predict(std::vector<T *> inputs) {
        if (!prepared_for_exe_) PrepareForExecution();
        for (size_t i = 0; i < inputs.size(); i++) {
            SetInputBuffer(i, inputs[i]);
        }
        ANeuralNetworksEvent *event = nullptr;
        if (int ret = ANeuralNetworksExecution_startCompute(execution_, &event);
            ret != ANEURALNETWORKS_NO_ERROR) {
            throw std::invalid_argument(
                "Error in startCompute, return value: " + std::to_string(ret));
        }

        if (int ret = ANeuralNetworksEvent_wait(event);
            ret != ANEURALNETWORKS_NO_ERROR) {
            throw std::invalid_argument("Error in wait, return value: " +
                                        std::to_string(ret));
        }

        ANeuralNetworksEvent_free(event);
        ANeuralNetworksExecution_free(execution_);
        prepared_for_exe_ = false;
    }

    ~Model();
    void SetOutputBuffer(int32_t index, float *buffer);
    void SetOutputBuffer(int32_t index, uint8_t *buffer);
    void SetOutputBuffer(int32_t index, char *buffer);
    void SetOutputBuffer(int32_t index, void *buffer, size_t elemsize);
    size_t GetSize(const std::string &name);
    size_t GetInputSize(const int &index);
    size_t GetOutputSize(const int &index);
};

#endif  // NNAPIEXAMPLE_MODEL_H
