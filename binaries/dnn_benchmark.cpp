//
// Created by daquexian on 29/01/19.
//

#include <string>
#include <sstream>
#include <istream>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

#include <glog/logging.h>
#include "android_log_helper.h"
#include <common/helper.h>
#include "ModelBuilder.h"
#include <DaqReader.h>

using std::string; using std::cout; using std::endl;
using Clock = std::chrono::high_resolution_clock;

auto get_model(css &daqName, css &outputBlob, const bool allowFp16, const PreferenceCode &compilePreference) {
    std::unique_ptr<Model> model;
    ModelBuilder builder;
    DaqReader daq_reader;
    // Set the last argument to true to use mmap. It may be more efficient than memory buffer.
    daq_reader.ReadDaq(daqName, builder, false);
#if __ANDROID_API__ >= __ANDROID_API_P__
    model = builder.AllowFp16(allowFp16).AddOutput(outputBlob).Compile(compilePreference);
#else
    model = builder.AddOutput(outputBlob).Compile(compilePreference);
#endif
    return model;
}

auto PrefCodeToStr(const PreferenceCode &preferenceCode) {
    if (preferenceCode == ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER) {
        return "fast single";
    }
    if (preferenceCode == ANEURALNETWORKS_PREFER_SUSTAINED_SPEED) {
        return "sustained speed";
    }
    if (preferenceCode == ANEURALNETWORKS_PREFER_LOW_POWER) {
        return "low power";
    }
    return "Unknown preference code";
}

// ./dnn_benchmark daqName
int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    FLAGS_logbuflevel = -1;
    if (argc != 5) {
        return -1;
    }
    css daqName = argv[1];
    css outputBlob = argv[2];
    const int numberRunning = std::atoi(argv[3]);
    const bool quant = std::atoi(argv[4]) != 0;

    size_t inputLen, outputLen;
    {
        auto model = get_model(daqName, outputBlob, false, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
        inputLen = model->GetInputSize(0);
        outputLen = model->GetOutputSize(0);
    }
#define WARM_UP \
    {   \
        auto model = get_model(daqName, outputBlob, false, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);  \
        for (int i = 0; i < 10; i++) {  \
            model->SetOutputBuffer(0, output);  \
            model->Predict(std::vector{data});  \
        }   \
    }

#define BENCHMARK(fp16Candidates, preferenceCandidates) \
    for (const auto allowFp16 : fp16Candidates) {    \
        for (const auto compilePreference : preferenceCandidates) {    \
            auto model = get_model(daqName, outputBlob, allowFp16, compilePreference);  \
            const auto t1 = Clock::now();   \
            for (int i = 0; i < numberRunning; i++) {   \
                model->SetOutputBuffer(0, output);  \
                model->Predict(std::vector{data});  \
            }   \
            const auto t2 = Clock::now();   \
            const auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();  \
            const auto singleTime = 1. * totalTime / numberRunning; \
            LOG(INFO) << "AllowFp16: " << allowFp16 << ", compile preference: " << PrefCodeToStr(compilePreference) <<  \
                ", time: " << totalTime << "/" << numberRunning << " = " << singleTime; \
        }   \
    }

    const std::vector<PreferenceCode> preferenceCandidates{ ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER,
        ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
        ANEURALNETWORKS_PREFER_LOW_POWER};
    if (quant) {
        uint8_t data[inputLen];
        uint8_t output[outputLen];
        WARM_UP;
        const std::vector<bool> fp16Candidates{false};
        BENCHMARK(fp16Candidates, preferenceCandidates);
    } else {
        float data[inputLen];
        FORZ(i, inputLen) {
            data[i] = i;
        }
        float output[outputLen];

        WARM_UP;

#if __ANDROID_API__ >= __ANDROID_API_P__
        const std::vector<bool> fp16Candidates{false, true};
#else
        const std::vector<bool> fp16Candidates{false};
#endif
        BENCHMARK(fp16Candidates, preferenceCandidates);
    }
}


