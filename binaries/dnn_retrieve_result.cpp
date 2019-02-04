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

// ./dnn_retrieve_result daqName outputBlob [input]
int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "/data/local/tmp/log";
    FLAGS_logbuflevel = -1;
    FLAGS_alsologtostderr = true;
    FLAGS_v = 5;
    if (argc < 4 || argc > 5) {
        return -1;
    }
    string daqName = argv[1];
    string outputBlob = argv[2];
    bool quant = std::atoi(argv[3]) != 0;
    bool use_external_input = argc == 5;

    std::unique_ptr<Model> model;
    {
        ModelBuilder builder;
        DaqReader daq_reader;
        // Set the last argument to true to use mmap. It may be more efficient than memory buffer.
        daq_reader.ReadDaq(daqName, builder, false);
        model = builder.AddOutput(outputBlob).Compile(ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
    }
    const auto inputLen = model->GetInputSize(0), outputLen = model->GetOutputSize(0);
    float data[inputLen];
    if (use_external_input) {
        std::ifstream ifs(argv[4]);
        float element;
        FORZ(i, inputLen) {
            if (!(ifs >> element)) {
                throw std::invalid_argument("Read file error");
            }
            data[i] = element;
        }
    } else {
        FORZ(i, inputLen) {
            data[i] = i;
        }
    }

    if (quant) {
        uint8_t output[outputLen];
        model->SetOutputBuffer(0, output);

        uint8_t uint8_data[inputLen];
        FORZ(i, inputLen) {
            uint8_data[i] = data[i];
        }
        model->Predict(std::vector{uint8_data});
        std::ofstream ofs("/data/local/tmp/result");
        FORZ(i, outputLen) {
            ofs << output[i] << endl;
        }
    } else {
        float output[outputLen];

        model->SetOutputBuffer(0, output);
        model->Predict(std::vector{data});
        std::ofstream ofs("/data/local/tmp/result");
        FORZ(i, outputLen) {
            ofs << output[i] << endl;
        }
    }
}
