//
// Created by daquexian on 2017/11/8.
// A wrapper for ANeuralNetworksExecution
//

#ifndef NNAPIEXAMPLE_MODEL_H
#define NNAPIEXAMPLE_MODEL_H

#include <vector>
#include <memory>

#include <android/NeuralNetworks.h>
#include <common/Shaper.h>
#include <common/StrKeyMap.h>

class Model {
    friend class ModelBuilder;
private:
    ANeuralNetworksModel* model_;
    ANeuralNetworksCompilation* compilation_;
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
    void PrepareForExecution();
    bool prepared_for_exe_;
public:
    // int Predict();
    void Predict(std::vector<float *> inputs);
    ~Model();
    void SetOutputBuffer(int32_t index, float *buffer);
    size_t GetSize(const std::string &name);
    size_t GetInputSize(const int &index);
    size_t GetOutputSize(const int &index);
};


#endif //NNAPIEXAMPLE_MODEL_H
