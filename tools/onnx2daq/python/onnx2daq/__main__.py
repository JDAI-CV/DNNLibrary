import argparse

import onnx
import onnxsim
import onnx2daq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input ONNX model')
    parser.add_argument('output_model', help='Output daq model')
    args = parser.parse_args()
    model = onnx.load(args.input_model)
    model = onnxsim.simplify(model)
    onnx2daq.convert(model.SerializeToString(), args.output_model)


if __name__ == '__main__':
    main()
