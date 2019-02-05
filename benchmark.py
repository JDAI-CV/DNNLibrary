import onnx
from onnx import numpy_helper
import os
import glob
import numpy as np
import tempfile

def run(onnx, onnx2daq, dnn_benchmark, output_name, number_running, table_file):
    quant = 1 if len(table_file) != 0 else 0
    daq = "temp.daq"
    os.system("{} {} {} {}".format(onnx2daq, onnx, daq, table_file))
    print("Converted to daq")

    os.system("adb push {} /data/local/tmp/".format(daq))
    os.system("adb push {} /data/local/tmp/dnn_benchmark".format(dnn_benchmark))
    os.system('adb shell "LD_LIBRARY_PATH=/data/local/tmp/ /data/local/tmp/dnn_benchmark /data/local/tmp/{} {} {} {}"'.format(os.path.basename(daq), output_name, number_running, quant))
    os.system("adb shell rm /data/local/tmp/dnn_benchmark")
    os.system("adb shell rm /data/local/tmp/{}".format(os.path.basename(daq)))
    os.system("rm {}".format(daq))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test onnx model on nnapi')
    parser.add_argument('onnx', type=str, help='onnx model file')
    parser.add_argument('onnx2daq', type=str, help='onnx2daq binary file')
    parser.add_argument('dnn_benchmark', type=str, help='dnn_benchmark binary file')
    parser.add_argument('output', type=str, help='Output name of the model')
    parser.add_argument('--number_running', type=int, help='The number of running', default=50)
    parser.add_argument('--table_file', type=str, help='table file for 8-bit quantization', default='')
    args = parser.parse_args()

    actual = run(args.onnx, args.onnx2daq, args.dnn_benchmark, args.output, args.number_running, args.table_file)
