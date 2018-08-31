//
// Created by daquexian on 8/22/18.
//

#ifndef DNNLIBRARY_FLATBUFFERS_HELPER_H
#define DNNLIBRARY_FLATBUFFERS_HELPER_H

#include <vector>

#include <flatbuffers/flatbuffers.h>

template <typename T>
std::vector<T> fbs_to_std_vector(const flatbuffers::Vector<T> *fbs_vec) {
    std::vector<T> std_vec;
    for (size_t i = 0; i < fbs_vec->size(); i++) {
        std_vec.push_back(fbs_vec->Get(static_cast<flatbuffers::uoffset_t>(i)));
    }
    return std_vec;
}

#endif //DNNLIBRARY_FLATBUFFERS_HELPER_H
