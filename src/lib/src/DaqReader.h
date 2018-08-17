//
// Created by daquexian on 8/13/18.
//

#ifndef DNNLIBRARY_DAQREADER_H
#define DNNLIBRARY_DAQREADER_H

#include <string>

#include <flatbuffers/flatbuffers.h>
#include <daq_generated.h>
#include <ModelBuilder.h>

class DaqReader {
public:
    bool ReadDaq(std::string filepath, ModelBuilder &builder);
};


#endif //DNNLIBRARY_DAQREADER_H
