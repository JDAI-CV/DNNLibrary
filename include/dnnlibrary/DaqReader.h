//
// Created by daquexian on 8/13/18.
//

#ifndef DNNLIBRARY_DAQREADER_H
#define DNNLIBRARY_DAQREADER_H

#include <memory>
#include <string>

#include <dnnlibrary/ModelBuilder.h>
#include <common/daq_generated.h>
#include <flatbuffers/flatbuffers.h>

namespace dnn {
class DaqReader {
   public:
    void ReadDaq(const std::string &filepath, Model &builder,
                 const bool use_mmap);
    void ReadDaq(const int &fd, Model &builder, const off_t offset = 0,
                 size_t fsize = 0);
    void ReadDaq(std::unique_ptr<uint8_t[]> buf, Model &builder);
    void ReadDaq(const uint8_t *buf, Model &builder);
};
}

#endif  // DNNLIBRARY_DAQREADER_H
