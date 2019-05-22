#! /usr/bin/env bash
set -e

nproc=$(ci/get_cores.sh)

mkdir build_onnx2daq && cd build_onnx2daq
cmake ..
cmake --build . -- -j$nproc
cd -
