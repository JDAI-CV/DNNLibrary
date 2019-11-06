#! /usr/bin/env bash

set -e

wget -q "https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.tar.gz" -O squeezenet1.1.tar.gz
tar xf squeezenet1.1.tar.gz
python3 ci/validate_onnx.py squeezenet1.1 build_dnnlibrary/binaries/dnn_retrieve_result

wget -q "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz" -O mobilenetv2-1.0.tar.gz
tar xf mobilenetv2-1.0.tar.gz
python3 ci/validate_onnx.py mobilenetv2-1.0 build_dnnlibrary/binaries/dnn_retrieve_result

wget -q "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.tar.gz" -O resnet18v2.tar.gz
tar xf resnet18v2.tar.gz
python3 ci/validate_onnx.py resnet18v2 build_dnnlibrary/binaries/dnn_retrieve_result

wget -q "https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_googlenet.tar.gz" -O bvlc_googlenet.tar.gz
tar xf bvlc_googlenet.tar.gz
python3 ci/validate_onnx.py bvlc_googlenet build_dnnlibrary/binaries/dnn_retrieve_result

