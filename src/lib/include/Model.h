//
// Created by daquexian on 2017/11/8.
// A wrapper for ANeuralNetworksExecution
//

#ifndef NNAPIEXAMPLE_MODEL_H
#define NNAPIEXAMPLE_MODEL_H

#include <vector>

#include <android/NeuralNetworks.h>

class Model {
    friend class ModelBuilder;
private:
    ANeuralNetworksModel* model;
    ANeuralNetworksCompilation* compilation;
    ANeuralNetworksExecution *execution;
    ANeuralNetworksMemory *memory;
    unsigned char *data;
    size_t data_size;
    std::vector<char *> charBufPointers;
    std::vector<float *> floatBufPointers;
public:
    int predict();
    ~Model();
    void setInputBuffer(int32_t index, void *buffer, size_t length);
    void setOutputBuffer(int32_t index, void *buffer, size_t length);
    void prepareForExecution();
};


#endif //NNAPIEXAMPLE_MODEL_H
