import argparse

import onnx
import onnxsim
import onnx2daq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input ONNX model')
    parser.add_argument('output_model', help='Output daq model')
    parser.add_argument('table_file', nargs='?', help='Table file for 8-bit quantization', default='')
    args = parser.parse_args()
    onnx2daq.simplify_and_convert(args.input_model, args.output_model, args.table_file)


if __name__ == '__main__':
    main()
