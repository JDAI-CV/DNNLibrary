#ifndef DNNLIBRARY_NNAPI_HELPER_H
#define DNNLIBRARY_NNAPI_HELPER_H

#include <string>

#include <dnnlibrary/NeuralNetworksTypes.h>

#define THROW_ON_ERROR(val)                                                   \
    {                                                                         \
        const auto ret = (val);                                               \
        if (ret != ANEURALNETWORKS_NO_ERROR) {                                \
            throw std::invalid_argument(                                      \
                std::string("Error in ") + __FILE__ + std::string(":") +      \
                std::to_string(__LINE__) + std::string(", function name: ") + \
                std::string(__func__) + "error, ret: " + GetErrorCause(ret)); \
        }                                                                     \
    }

#define THROW_ON_ERROR_WITH_NOTE(val, note)                                   \
    {                                                                         \
        const auto ret = (val);                                               \
        if (ret != ANEURALNETWORKS_NO_ERROR) {                                \
            throw std::invalid_argument(                                      \
                std::string("Error in ") + __FILE__ + std::string(":") +      \
                std::to_string(__LINE__) + std::string(", function name: ") + \
                std::string(__func__) + "error, ret: " + GetErrorCause(ret) + \
                std::string(", ") + (note));                                  \
        }                                                                     \
    }

inline std::string GetErrorCause(int errorCode) {
    switch (errorCode) {
        case ANEURALNETWORKS_OUT_OF_MEMORY:
            return "Out of memory";
        case ANEURALNETWORKS_BAD_DATA:
            return "Bad data";
        case ANEURALNETWORKS_BAD_STATE:
            return "Bad state";
        case ANEURALNETWORKS_INCOMPLETE:
            return "Incomplete";
        case ANEURALNETWORKS_UNEXPECTED_NULL:
            return "Unexpected null";
        case ANEURALNETWORKS_OP_FAILED:
            return "Op failed";
        case ANEURALNETWORKS_UNMAPPABLE:
            return "Unmappable";
        case ANEURALNETWORKS_NO_ERROR:
            return "No error";
        default:
            return "Unknown error code";
    }
}

#endif /* DNNLIBRARY_NNAPI_HELPER_H */
