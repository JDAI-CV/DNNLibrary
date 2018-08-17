//
// Created by daquexian on 2017/11/8.
//

#include <Model.h>

#include <string>
#include <stdexcept>
#include <sys/mman.h>

Model::Model() {

}

Model::Model(ANeuralNetworksExecution *execution) :execution(execution){

}

int Model::predict() {
    ANeuralNetworksEvent* event = nullptr;
    int ret;
    if ((ret = ANeuralNetworksExecution_startCompute(execution, &event)) != ANEURALNETWORKS_NO_ERROR) {
        return ret;
    }

    if ((ret = ANeuralNetworksEvent_wait(event)) != ANEURALNETWORKS_NO_ERROR) {
        return ret;
    }

    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);

    return 0;
}

Model::~Model() {
    munmap(data, data_size);
    ANeuralNetworksMemory_free(memory);
}

void Model::setInputBuffer(int32_t index, void *buffer, size_t length) {
    auto ret = ANeuralNetworksExecution_setInput(execution, index, nullptr, buffer, length);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Invalid index in setInputBuffer, return value: " + std::to_string(ret));
    }
}

void Model::setOutputBuffer(int32_t index, void *buffer, size_t length) {
    auto ret = ANeuralNetworksExecution_setOutput(execution, index, nullptr, buffer, length);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Invalid index in setInputBuffer, return value: " + std::to_string(ret));
    }
}

