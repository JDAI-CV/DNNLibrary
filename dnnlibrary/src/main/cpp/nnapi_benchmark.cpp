//
// Created by daquexian on 5/10/18.
//

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <chrono>

#include "ModelBuilder.h"

using std::string; using std::cout; using std::endl;
typedef std::chrono::high_resolution_clock Clock;

// ./dnntest preference daqName outputBlob
int main(int argc, char** argv) {
    if (argc != 4) {
        return -1;
    }
    string daqName = argv[2];
    string outputBlob = argv[3];

    uint32_t preference = static_cast<uint32_t>(std::stoi(string(argv[1])));
    if (preference > 2) {
        preference = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER;
    }
    cout << "preference is " << preference << endl;

    ModelBuilder builder;
    builder.init();
    builder.readFromFile(daqName);
    builder.addIndexIntoOutput(builder.getBlobIndex(outputBlob));
    cout << ModelBuilder::getErrorProcedure(builder.compile(preference)) << endl;
    Model model;
    auto inputDim = builder.getBlobDim(builder.getInputIndexes()[0]);
    float data[inputDim[1] * inputDim[2] * inputDim[3]];
    uint32_t outputLen = product(builder.getBlobDim(builder.getOutputIndexes()[0]));

    float output[outputLen];

#define WARM_UP 5
    for (int i = 0; i < WARM_UP; i++) {
        builder.prepareForExecution(model);
        builder.setInputBuffer(model, builder.getInputIndexes()[0], data, sizeof(data));
        builder.setOutputBuffer(model, builder.getOutputIndexes()[0], output, sizeof(output));
        model.predict();
    }
#define RUNS 100
    auto t1 = Clock::now();
    for (int i = 0; i < RUNS; i++) {
        builder.prepareForExecution(model);
        builder.setInputBuffer(model, builder.getInputIndexes()[0], data, sizeof(data));
        builder.setOutputBuffer(model, builder.getOutputIndexes()[0], output, sizeof(output));
        model.predict();
    }
    auto t2 = Clock::now();

    cout << "time: " << (1. * std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / RUNS) << " microseconds." << endl;
}
