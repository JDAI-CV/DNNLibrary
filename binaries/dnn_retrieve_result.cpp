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

#include <common/helper.h>
#include <dnnlibrary/DaqReader.h>
#include <dnnlibrary/ModelBuilder.h>
#ifdef DNN_READ_ONNX
#include <dnnlibrary/OnnxReader.h>
#endif
#include <glog/logging.h>
#include "argh.h"

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

template <typename T>
std::vector<T> NHWC2NCHW(const std::vector<T> &nhwc, const size_t n,
                         const size_t h, const size_t w, const size_t c) {
    std::vector<T> nchw;
    nchw.resize(n * h * w * c);
    FORZ(i, n) {
        FORZ(j, h) {
            FORZ(k, w) {
                FORZ(l, c) {
                    nchw[i * c * h * w + l * h * w + j * w + k] =
                        nhwc[i * h * w * c + j * w * c + k * c + l];
                }
            }
        }
    }
    return nchw;
}

// Usage: ./dnn_retrieve_result daqName [--quant_input] [--quant_output]
// [--nchw_result] [input1 ..]
int main(int argc, char **argv) {
    argh::parser cmdl(argc, argv);
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "/data/local/tmp/log";
    FLAGS_logbuflevel = -1;
    FLAGS_alsologtostderr = true;
    FLAGS_v = cmdl("v", 5);
    string daqName = cmdl[1];
    bool quant_input = cmdl["quant_input"];
    bool quant_output = cmdl["quant_output"];
    bool nchw_result = cmdl["nchw_result"];
    bool use_external_input = cmdl(2);
    PNT(use_external_input);

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
        throw std::invalid_argument("Wrong model name " + daqName +
                              ". It must end with .daq or .onnx (.onnx is only "
                              "supported when DNN_READ_ONNX is ON)");
    }
    model = builder.Compile(ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
    DNN_ASSERT(model->GetOutputs().size() == 1,
               "the number of outputs can only be 1 here");
    const auto outputLen = model->GetSize(model->GetOutputs()[0]);
    std::vector<std::vector<float>> inputs;
    for (size_t i = 2, n = 0; n < model->GetInputs().size(); i++, n++) {
        const auto &input_name = model->GetInputs()[n];
        const auto input_size = model->GetSize(input_name);
        std::vector<float> input_data;
        input_data.reserve(input_size);
        if (use_external_input) {
            std::ifstream ifs(cmdl[i]);
            float element;
            FORZ(_, model->GetSize(input_name)) {
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

    std::vector<uint8_t> output_uint8(outputLen);
    std::vector<float> output_float(outputLen);
    PNT(quant_input, quant_output);
    if (quant_output) {
        model->SetOutputBuffer(0, output_uint8.data());
    } else {
        model->SetOutputBuffer(0, output_float.data());
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
    const auto &output_shape = model->GetShape(model->GetOutputs()[0]);
    if (nchw_result && output_shape.size() == 4) {
        const size_t n = output_shape[0], h = output_shape[1],
                     w = output_shape[2], c = output_shape[3];
        if (quant_output) {
            output_uint8 = NHWC2NCHW(output_uint8, n, h, w, c);
        } else {
            output_float = NHWC2NCHW(output_float, n, h, w, c);
        }
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
