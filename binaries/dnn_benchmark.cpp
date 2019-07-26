//
// Created by daquexian on 29/01/19.
//

#include <chrono>
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

#include <common/helper.h>
#include <dnnlibrary/DaqReader.h>
#include <dnnlibrary/ModelBuilder.h>
#ifdef DNN_READ_ONNX
#include <dnnlibrary/OnnxReader.h>
#endif
#include <glog/logging.h>

using std::cout;
using std::endl;
using std::string;
using Clock = std::chrono::high_resolution_clock;
using dnn::DaqReader;
using dnn::Model;
using dnn::ModelBuilder;
#ifdef DNN_READ_ONNX
using dnn::OnnxReader;
#endif

bool hasEnding(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(),
                                        ending.length(), ending));
    } else {
        return false;
    }
}

auto GetModel(css &daqName, const bool allow_fp16,
              const int compile_preference) {
    std::unique_ptr<Model> model;
    ModelBuilder builder;
    if (hasEnding(daqName, ".daq")) {
        DaqReader daq_reader;
        // Set the last argument to true to use mmap. It may be more efficient
        // than memory buffer.
        daq_reader.ReadDaq(daqName, builder, false);
#ifdef DNN_READ_ONNX
    } else if (hasEnding(daqName, ".onnx")) {
        OnnxReader onnx_reader;
        // Set the last argument to true to use mmap. It may be more efficient
        // than memory buffer.
        onnx_reader.ReadOnnx(daqName, builder);
#endif
    } else {
        throw std::invalid_argument(
            "Wrong model name " + daqName +
            ". It must end with .daq or .onnx (.onnx is only "
            "supported when DNN_READ_ONNX is ON)");
    }
    model = builder.AllowFp16(allow_fp16).Compile(compile_preference);
    return model;
}

auto PrefCodeToStr(const int &preference_code) {
    if (preference_code == ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER) {
        return "fast single";
    }
    if (preference_code == ANEURALNETWORKS_PREFER_SUSTAINED_SPEED) {
        return "sustained speed";
    }
    if (preference_code == ANEURALNETWORKS_PREFER_LOW_POWER) {
        return "low power";
    }
    return "Unknown preference code";
}

// ./dnn_benchmark daq_name
int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    FLAGS_logbuflevel = -1;
    FLAGS_v = 0;
    if (argc != 4) {
        return -1;
    }
    css daq_name = argv[1];
    const int number_running = std::atoi(argv[2]);
    const bool quant = std::atoi(argv[3]) != 0;

    size_t input_len, output_len;
    {
        auto model = GetModel(daq_name, false,
                              ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
        input_len = model->GetSize(model->GetInputs()[0]);
        output_len = model->GetSize(model->GetOutputs()[0]);
    }
#define WARM_UP                                                           \
    {                                                                     \
        auto model = GetModel(daq_name, false,                            \
                              ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER); \
        for (int i = 0; i < 10; i++) {                                    \
            model->SetOutputBuffer(0, output);                            \
            model->Predict(std::vector{data});                            \
        }                                                                 \
    }

#define BENCHMARK(fp16_candidates, preference_candidates)                      \
    for (const auto allow_fp16 : fp16_candidates) {                            \
        for (const auto compile_preference : preference_candidates) {          \
            auto model = GetModel(daq_name, allow_fp16, compile_preference);   \
            const auto t1 = Clock::now();                                      \
            for (int i = 0; i < number_running; i++) {                         \
                model->SetOutputBuffer(0, output);                             \
                model->Predict(std::vector{data});                             \
            }                                                                  \
            const auto t2 = Clock::now();                                      \
            const auto total_time =                                            \
                std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1) \
                    .count();                                                  \
            const auto single_time = 1. * total_time / number_running;         \
            LOG(INFO) << "AllowFp16: " << allow_fp16                           \
                      << ", compile preference: "                              \
                      << PrefCodeToStr(compile_preference)                     \
                      << ", time: " << total_time << "/" << number_running     \
                      << " = " << single_time;                                 \
        }                                                                      \
    }

    const std::vector<int> preference_candidates{
        ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER,
        ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
        ANEURALNETWORKS_PREFER_LOW_POWER};
    if (quant) {
        uint8_t data[input_len];
        float output[output_len];
        WARM_UP;
        const std::vector<bool> fp16_candidates{false};
        BENCHMARK(fp16_candidates, preference_candidates);
    } else {
        float data[input_len];
        FORZ(i, input_len) {
            data[i] = i;
        }
        float output[output_len];

        WARM_UP;

        const std::vector<bool> fp16_candidates =
            GetAndroidSdkVersion() >= __ANDROID_API_P__
                ? std::vector<bool>{false, true}
                : std::vector<bool>{false};
        BENCHMARK(fp16_candidates, preference_candidates);
    }
}
