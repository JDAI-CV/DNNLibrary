//
// Created by daquexian on 2017/11/8.
// A wrapper for ANeuralNetworksExecution
//

#ifndef NNAPIEXAMPLE_MODEL_H
#define NNAPIEXAMPLE_MODEL_H


#include <android/NeuralNetworks.h>

class Model {
    friend class ModelBuilder;
private:
    explicit Model(ANeuralNetworksExecution *execution);
    ANeuralNetworksExecution *execution;
    ANeuralNetworksMemory *memory;
    unsigned char *data;
    size_t data_size;
public:
    Model();
    int predict();
    ~Model();
    void setInputBuffer(int32_t index, void *buffer, size_t length);
    void setOutputBuffer(int32_t index, void *buffer, size_t length);
};


#endif //NNAPIEXAMPLE_MODEL_H
