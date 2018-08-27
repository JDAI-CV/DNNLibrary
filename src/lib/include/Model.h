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
    ANeuralNetworksModel* model;
    ANeuralNetworksCompilation* compilation;
    ANeuralNetworksExecution *execution;
    ANeuralNetworksMemory *memory;
    unsigned char *data;
    size_t data_size;
    std::vector<std::unique_ptr<char[]>> charBufPointers;
    std::vector<std::unique_ptr<float[]>> floatBufPointers;
    std::vector<std::unique_ptr<int32_t[]>> int32BufPointers;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    Shaper shaper;
    void addInput(const std::string &name, const Shaper::Shape &shape);
    void addOutput(const std::string &name, const Shaper::Shape &shape);
    void setInputBuffer(int32_t index, float *buffer);
    void prepareForExecution();
    bool prepared_for_exe;
public:
    // int predict();
    void predict(std::vector<float *> inputs);
    ~Model();
    void setOutputBuffer(int32_t index, float *buffer);
    size_t getSize(const std::string &name);
    size_t getInputSize(const int &index);
    size_t getOutputSize(const int &index);
};


#endif //NNAPIEXAMPLE_MODEL_H
