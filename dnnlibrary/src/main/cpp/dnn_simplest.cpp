//
// Created by daquexian on 5/21/18.
//

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <chrono>

#include "ModelBuilder.h"

using std::string; using std::cout; using std::endl;
typedef std::chrono::high_resolution_clock Clock;

int main(int argc, char** argv) {
    ModelBuilder builder;
    cout << builder.init() << endl;
    builder.simplestModel();
    int ret = builder.compile(ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    cout << ModelBuilder::getErrorProcedure(ret) << endl;
    cout << ModelBuilder::getErrorCause(ret) << endl;
    Model model;
    return 0;
}
