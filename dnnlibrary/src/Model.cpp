//
// Created by daquexian on 2017/11/8.
//

#include <Model.h>

#include <string>
#include <stdexcept>
#include <sys/mman.h>
#include <utility>

#include <glog/logging.h>


void Model::PrepareForExecution() {
    if (compilation_ == nullptr) {
        throw std::invalid_argument("Error in PrepareForExecution, compilation_ == nullptr");
    }
    auto ret = ANeuralNetworksExecution_create(compilation_, &execution_);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Error in PrepareForExecution, ret: " + std::to_string(ret));
    }
    prepared_for_exe_ = true;
}

Model::~Model() {
    munmap(data_, data_size_);
    ANeuralNetworksModel_free(model_);
    ANeuralNetworksCompilation_free(compilation_);
    ANeuralNetworksMemory_free(memory_);
}

void Model::SetInputBuffer(int32_t index, float *buffer) {
    if (!prepared_for_exe_) PrepareForExecution();
    auto size = shaper_.GetSize(input_names_[index]) * sizeof(float);
    auto ret = ANeuralNetworksExecution_setInput(execution_, index, nullptr, buffer, size);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Invalid index in SetInputBuffer, return value: " + std::to_string(ret));
    }
}

void Model::SetOutputBuffer(int32_t index, float *buffer) {
    if (!prepared_for_exe_) PrepareForExecution();
    auto size = shaper_.GetSize(output_names_[index]) * sizeof(float);
    auto ret = ANeuralNetworksExecution_setOutput(execution_, index, nullptr, buffer, size);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Invalid index in SetOutputBuffer, return value: " + std::to_string(ret));
    }
}

void Model::AddInput(const std::string &name, const Shaper::Shape &shape) {
    input_names_.push_back(name);
    shaper_.AddShape(name, shape);
}

void Model::AddOutput(const std::string &name, const Shaper::Shape &shape) {
    output_names_.push_back(name);
    shaper_.AddShape(name, shape);
}

void Model::Predict(std::vector<float *> inputs) {
    if (!prepared_for_exe_) PrepareForExecution();
    for (size_t i = 0; i < inputs.size(); i++) {
        SetInputBuffer(i, inputs[i]);
    }
    ANeuralNetworksEvent* event = nullptr;
    if (int ret = ANeuralNetworksExecution_startCompute(execution_, &event); ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Error in startCompute, return value: " + std::to_string(ret));
    }

    if (int ret = ANeuralNetworksEvent_wait(event); ret != ANEURALNETWORKS_NO_ERROR) {
        throw std::invalid_argument("Error in wait, return value: " + std::to_string(ret));
    }

    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution_);
    prepared_for_exe_ = false;
}

size_t Model::GetSize(const std::string &name) {
    return shaper_.GetSize(name);
}

size_t Model::GetInputSize(const int &index) {
    return shaper_.GetSize(input_names_[index]);
}

size_t Model::GetOutputSize(const int &index) {
    return shaper_.GetSize(output_names_[index]);
}

