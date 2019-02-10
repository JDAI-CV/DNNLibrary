//
// Created by daquexian on 5/21/18.
//

#include <chrono>
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

#include <DaqReader.h>
#include <common/helper.h>
#include <glog/logging.h>
#include "ModelBuilder.h"
#include "android_log_helper.h"

using std::cout;
using std::endl;
using std::string;
using Clock = std::chrono::high_resolution_clock;

// ./dnn_retrieve_result daqName outputBlob [input]
int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "/data/local/tmp/log";
    FLAGS_logbuflevel = -1;
    FLAGS_alsologtostderr = true;
    FLAGS_v = 5;
    if (argc < 5 || argc > 6) {
        return -1;
    }
    string daqName = argv[1];
    string outputBlob = argv[2];
    bool quant_input = std::atoi(argv[3]) != 0;
    bool quant_output = std::atoi(argv[4]) != 0;
    bool use_external_input = argc == 6;

    std::unique_ptr<Model> model;
    {
        ModelBuilder builder;
        DaqReader daq_reader;
        // Set the last argument to true to use mmap. It may be more efficient
        // than memory buffer.
        daq_reader.ReadDaq(daqName, builder, false);
        model = builder.AddOutput(outputBlob)
                    .Compile(ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
    }
    const auto inputLen = model->GetInputSize(0),
               outputLen = model->GetOutputSize(0);
    float data[inputLen];
    if (use_external_input) {
        std::ifstream ifs(argv[5]);
        float element;
        FORZ(i, inputLen) {
            if (!(ifs >> element)) {
                throw std::invalid_argument("Read file error");
            }
            data[i] = element;
        }
    } else {
        FORZ(i, inputLen) { data[i] = i; }
    }

    uint8_t output_uint8[outputLen];
    float output_float[outputLen];
    PNT(quant_input, quant_output);
    if (quant_output) {
        model->SetOutputBuffer(0, output_uint8);
    } else {
        model->SetOutputBuffer(0, output_float);
    }
    if (quant_input) {
        uint8_t uint8_data[inputLen];
        FORZ(i, inputLen) { uint8_data[i] = data[i]; }
        model->Predict(std::vector{uint8_data});
    } else {
        model->Predict(std::vector{data});
        std::ofstream ofs("/data/local/tmp/result");
    }
    std::ofstream ofs("/data/local/tmp/result");
    if (quant_output) {
        FORZ(i, outputLen) { ofs << static_cast<int>(output_uint8[i]) << endl; }
    } else {
        FORZ(i, outputLen) { ofs << output_float[i] << endl; }
    }
}
