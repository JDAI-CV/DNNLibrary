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
    bool use_external_input = argc >= 6;

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
    DNN_ASSERT(model->GetOutputs().size() == 1, "the number of outputs can only be 1 here");
    const auto outputLen = model->GetSize(model->GetOutputs()[0]);
    std::vector<std::vector<float>> inputs;
    for (int i = 5, n = 0; i < argc; i++, n++) {
        const auto &input_name = model->GetInputs()[n];
        const auto input_size = model->GetSize(input_name);
        std::vector<float> input_data;
        input_data.reserve(input_size);
        if (use_external_input) {
            std::ifstream ifs(argv[i]);
            float element;
            FORZ(i, model->GetSize(input_name)) {
                if (!(ifs >> element)) {
                    throw std::invalid_argument("Read file error");
                }
                input_data.push_back(element);
            }
        } else {
            FORZ(j, input_size) {
                input_data.push_back(j);
            }
        }
        inputs.push_back(input_data);
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
        std::vector<std::vector<uint8_t>> uint8_inputs;
        for (const auto &input : inputs) {
            std::vector<uint8_t> uint8_input(input.begin(), input.end());
            uint8_inputs.push_back(uint8_input);
        }
        model->Predict(uint8_inputs);
    } else {
        model->Predict(inputs);
    }
    std::ofstream ofs("/data/local/tmp/result");
    if (quant_output) {
        FORZ(i, outputLen) {
            ofs << static_cast<int>(output_uint8[i]) << endl;
        }
    } else {
        FORZ(i, outputLen) {
            ofs << output_float[i] << endl;
        }
    }
}
