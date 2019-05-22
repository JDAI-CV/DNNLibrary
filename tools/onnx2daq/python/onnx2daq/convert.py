import _onnx2daq

import onnx
import onnxsim


def convert(model, output, table_file=""):
    if type(model) == str:
        model = onnx.load(model)
        model = model.SerializeToString()
    elif type(model) == onnx.ModelProto:
        model = model.SerializeToString()
    elif type(model) == bytes:
        pass
    else:
        raise RuntimeError("Input of function convert can only be str, onnx.ModelProto or bytes")
    _onnx2daq.convert(model, output, table_file)


def simplify_and_convert(model, output, table_file=""):
    if type(model) == str:
        model = onnx.load(model)
    elif type(model) == onnx.ModelProto:
        pass
    else:
        raise RuntimeError("Input of function convert can only be str, onnx.ModelProto")

    model = onnxsim.simplify(model)
    convert(model, output, table_file)
