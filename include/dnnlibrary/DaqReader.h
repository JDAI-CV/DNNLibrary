//
// Created by daquexian on 8/13/18.
//

#ifndef DNNLIBRARY_DAQREADER_H
#define DNNLIBRARY_DAQREADER_H

#include <common/daq_generated.h>
#include <dnnlibrary/ModelBuilder.h>
#include <flatbuffers/flatbuffers.h>

#include <memory>
#include <string>

namespace dnn {
class DaqReader {
   public:
    expected<Unit, std::string> ReadDaq(const std::string &filepath,
                                        ModelBuilder &builder,
                                        const bool use_mmap);
    expected<Unit, std::string> ReadDaq(const int &fd, ModelBuilder &builder,
                                        const off_t offset = 0,
                                        size_t fsize = 0);
    expected<Unit, std::string> ReadDaq(std::unique_ptr<uint8_t[]> buf,
                                        ModelBuilder &builder);
    expected<Unit, std::string> ReadDaq(const uint8_t *buf,
                                        ModelBuilder &builder);
};
}  // namespace dnn

#endif  // DNNLIBRARY_DAQREADER_H
