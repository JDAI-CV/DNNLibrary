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
    Model(ANeuralNetworksExecution *execution);
    ANeuralNetworksExecution *execution;
public:
    int predict();
};


#endif //NNAPIEXAMPLE_MODEL_H
