//
// Created by daquexian on 2017/11/8.
//

#include <dnnlibrary/Model.h>

#include <sys/mman.h>
#include <stdexcept>
#include <string>
#include <utility>

#include <common/helper.h>
#include <glog/logging.h>
#include <dnnlibrary/nnapi_helper.h>

namespace dnn {
template void Model::Predict<float>(const std::vector<float> &);
template void Model::Predict<uint8_t>(const std::vector<uint8_t> &);
template void Model::Predict<float>(const std::vector<std::vector<float>> &);
template void Model::Predict<uint8_t>(
    const std::vector<std::vector<uint8_t>> &);
template void Model::Predict<float>(const float *);
template void Model::Predict<uint8_t>(const uint8_t *);
template void Model::Predict<float>(const std::vector<float *> &);
template void Model::Predict<uint8_t>(const std::vector<uint8_t *> &);

void Model::PrepareForExecution() {
    if (compilation_ == nullptr) {
        throw std::invalid_argument(
            "Error in PrepareForExecution, compilation_ == nullptr");
    }
    THROW_ON_ERROR(
        nnapi_->ANeuralNetworksExecution_create(compilation_, &execution_));
    prepared_for_exe_ = true;
}

Model::Model() : nnapi_(NnApiImplementation()) {
}

Model::~Model() {
    munmap(data_, data_size_);
    nnapi_->ANeuralNetworksModel_free(model_);
    nnapi_->ANeuralNetworksCompilation_free(compilation_);
    nnapi_->ANeuralNetworksMemory_free(memory_);
}

void Model::SetInputBuffer(const int32_t index, const float *buffer) {
    SetInputBuffer(index, buffer, 4);
}

void Model::SetInputBuffer(const int32_t index, const uint8_t *buffer) {
    SetInputBuffer(index, buffer, 1);
}

void Model::SetInputBuffer(const int32_t index, const void *buffer,
                           const size_t elemsize) {
    if (!prepared_for_exe_) PrepareForExecution();
    auto size = shaper_.GetSize(input_names_[index]) * elemsize;
    THROW_ON_ERROR(nnapi_->ANeuralNetworksExecution_setInput(
        execution_, index, nullptr, buffer, size))
}

void Model::SetOutputBuffer(const int32_t index, float *buffer) {
    SetOutputBuffer(index, buffer, 4);
}

void Model::SetOutputBuffer(const int32_t index, uint8_t *buffer) {
    SetOutputBuffer(index, buffer, 1);
}

void Model::SetOutputBuffer(const int32_t index, char *buffer) {
    SetOutputBuffer(index, reinterpret_cast<uint8_t *>(buffer));
}

void Model::SetOutputBuffer(const int32_t index, void *buffer,
                            const size_t elemsize) {
    if (!prepared_for_exe_) PrepareForExecution();
    auto size = shaper_.GetSize(output_names_[index]) * elemsize;
    THROW_ON_ERROR(nnapi_->ANeuralNetworksExecution_setOutput(
        execution_, index, nullptr, buffer, size))
}

void Model::PredictAfterSetInputBuffer() {
    ANeuralNetworksEvent *event = nullptr;
    THROW_ON_ERROR(
        nnapi_->ANeuralNetworksExecution_startCompute(execution_, &event));
    THROW_ON_ERROR(nnapi_->ANeuralNetworksEvent_wait(event));

    nnapi_->ANeuralNetworksEvent_free(event);
    nnapi_->ANeuralNetworksExecution_free(execution_);
    prepared_for_exe_ = false;
}

void Model::AddInput(const std::string &name, const Shaper::Shape &shape) {
    input_names_.push_back(name);
    shaper_.AddShape(name, shape);
}

void Model::AddOutput(const std::string &name, const Shaper::Shape &shape) {
    output_names_.push_back(name);
    shaper_.AddShape(name, shape);
}

size_t Model::GetSize(const std::string &name) {
    return shaper_.GetSize(name);
}

Shaper::Shape Model::GetShape(const std::string &name) {
    return shaper_[name];
}

std::vector<std::string> Model::GetInputs() {
    return input_names_;
}

std::vector<std::string> Model::GetOutputs() {
    return output_names_;
}

template <typename T>
void Model::Predict(const std::vector<T> &input) {
    DNN_ASSERT_EQ(input.size(), GetSize(GetInputs()[0]));
    // const_cast is a ugly workaround, vector<const T*> causes strange errors
    Predict<T>({const_cast<T *>(input.data())});
}
template <typename T>
void Model::Predict(const std::vector<std::vector<T>> &inputs) {
    std::vector<T *> input_ptrs;
    for (size_t i = 0; i < inputs.size(); i++) {
        auto &input = inputs[i];
        DNN_ASSERT_EQ(input.size(), GetSize(GetInputs()[i]));
        // const_cast is a ugly workaround, vector<const T*> causes strange
        // errors
        input_ptrs.push_back(const_cast<T *>(input.data()));
    }
    Predict<T>(input_ptrs);
}
template <typename T>
void Model::Predict(const T *input) {
    // Predict<T>({input}) doesn't compile. Have no idea why.
    std::vector<T *> inputs;
    inputs.push_back(const_cast<T *>(input));
    Predict<T>(inputs);
}

template <typename T>
void Model::Predict(const std::vector<T *> &inputs) {
    DNN_ASSERT_EQ(inputs.size(), GetInputs().size());
    if (!prepared_for_exe_) PrepareForExecution();
    for (size_t i = 0; i < inputs.size(); i++) {
        SetInputBuffer(i, inputs[i]);
    }
    PredictAfterSetInputBuffer();
}
}  // namespace dnn
