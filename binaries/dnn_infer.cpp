//
// Created by daquexian on 5/21/18.
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

#define WARM_UP 5
#define RUNS 20

// ./dnn_save_result daqName outputBlob [input]
int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "/data/local/tmp/log";
    FLAGS_logbuflevel = -1;
    if (argc < 3 || argc > 4) {
        return -1;
    }
    string daqName = argv[1];
    string outputBlob = argv[2];
    bool use_external_input = argc == 4;

    std::unique_ptr<Model> model;
    {
        ModelBuilder builder;
        DaqReader daq_reader;
        // Set the last argument to true to use mmap. It may be more efficient than memory buffer.
        daq_reader.ReadDaq(daqName, builder, false);
        model = builder.AddOutput(outputBlob).Compile(ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
    }
    auto inputLen = model->GetInputSize(0), outputLen = model->GetOutputSize(0);
    float data[inputLen];
    if (use_external_input) {
        std::ifstream ifs(argv[3]);
        float element;
        for (int i = 0; i < inputLen; i++) {
            if (!(ifs >> element)) {
                throw std::invalid_argument("Read file error");
            }
            data[i] = element;
        }
    } else {
        for (int i = 0; i < inputLen; i++) {
            data[i] = i;
        }
    }

    float output[outputLen];

    for (int i = 0; i < WARM_UP; i++) {
        model->SetOutputBuffer(0, output);
        model->Predict(std::vector{data});
    }
    auto t1 = Clock::now();
    for (int i = 0; i < RUNS; i++) {
        model->SetOutputBuffer(0, output);
        model->Predict(std::vector{data});
    }
    auto t2 = Clock::now();
    LOG(INFO) << RUNS << " times, " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms";
    std::ofstream ofs("/data/local/tmp/result");
    for (int i = 0; i < outputLen; i++) {
        ofs << output[i] << endl;
    }
}
