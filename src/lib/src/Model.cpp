#include <utility>

//
// Created by daquexian on 2017/11/8.
//

#include <Model.h>

#include <string>
#include <stdexcept>
#include <sys/mman.h>

#include <glog/logging.h>
#include <src/lib/include/Model.h>


void Model::prepareForExecution() {
    if (compilation == nullptr) {
        throw std::invalid_argument("Error in prepareForExecution, compilation == nullptr");
    }
    auto ret = ANeuralNetworksExecution_create(compilation, &execution);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Error in prepareForExecution, ret: " + std::to_string(ret));
    }
    prepared_for_exe = true;
}

Model::~Model() {
    munmap(data, data_size);
    ANeuralNetworksModel_free(model);
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksMemory_free(memory);
}

void Model::setInputBuffer(int32_t index, float *buffer) {
    if (!prepared_for_exe) prepareForExecution();
    auto size = shaper.getSize(input_names[index]) * sizeof(float);
    auto ret = ANeuralNetworksExecution_setInput(execution, index, nullptr, buffer, size);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Invalid index in setInputBuffer, return value: " + std::to_string(ret));
    }
}

void Model::setOutputBuffer(int32_t index, float *buffer) {
    if (!prepared_for_exe) prepareForExecution();
    auto size = shaper.getSize(output_names[index]) * sizeof(float);
    auto ret = ANeuralNetworksExecution_setOutput(execution, index, nullptr, buffer, size);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Invalid index in setOutputBuffer, return value: " + std::to_string(ret));
    }
}

void Model::addInput(const std::string &name, const Shaper::Shape &shape) {
    input_names.push_back(name);
    shaper.AddShape(name, shape);
}

void Model::addOutput(const std::string &name, const Shaper::Shape &shape) {
    output_names.push_back(name);
    shaper.AddShape(name, shape);
}

void Model::predict(std::vector<float *> inputs) {
    if (!prepared_for_exe) prepareForExecution();
    for (int32_t i = 0; i < inputs.size(); i++) {
        setInputBuffer(i, inputs[i]);
    }
    ANeuralNetworksEvent* event = nullptr;
    if (int ret = ANeuralNetworksExecution_startCompute(execution, &event); ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Error in startCompute, return value: " + std::to_string(ret));
    }

    if (int ret = ANeuralNetworksEvent_wait(event); ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Error in wait, return value: " + std::to_string(ret));
    }

    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);
    prepared_for_exe = false;
}

size_t Model::getSize(const std::string &name) {
    return shaper.getSize(name);
}

size_t Model::getInputSize(const int &index) {
    return shaper.getSize(input_names[index]);
}

size_t Model::getOutputSize(const int &index) {
    return shaper.getSize(output_names[index]);
}

