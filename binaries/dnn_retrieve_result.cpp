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
#ifdef DNN_READ_ONNX
#include <OnnxReader.h>
#endif
#include <common/helper.h>
#include <glog/logging.h>
#include "ModelBuilder.h"
#include "android_log_helper.h"

using std::cout;
using std::endl;
using std::string;
using Clock = std::chrono::high_resolution_clock;

bool hasEnding(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(),
                                        ending.length(), ending));
    } else {
        return false;
    }
}

// ./dnn_retrieve_result daqName quant_input? quant_output? [input]
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
    bool quant_input = std::atoi(argv[2]) != 0;
    bool quant_output = std::atoi(argv[3]) != 0;
    bool use_external_input = argc >= 5;

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
        std::invalid_argument("Wrong model name " + daqName +
                              ". It must end with .daq or .onnx (.onnx is only "
                              "supported when DNN_READ_ONNX is ON)");
    }
    model = builder.Compile(ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
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
