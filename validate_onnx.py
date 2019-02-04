import onnx
from onnx import numpy_helper
import os
import glob
import numpy as np
import tempfile

def run(input_arr, onnx, onnx2daq, dnn_retrieve_result, output_name, table_file=''):
    daq = "temp.daq"
    os.system("{} {} {} {}".format(onnx2daq, onnx, daq, table_file))
    print("Converted to daq")
    nchw_shape = input_arr.shape
    nhwc_shape = (nchw_shape[0], nchw_shape[2], nchw_shape[3], nchw_shape[1])
    nhwc_input = np.moveaxis(input_arr, 1, -1)
    assert nhwc_input.shape == nhwc_shape
    np.savetxt('input.txt', nhwc_input.flatten(), delimiter='\n')

    txt = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))
    os.system("adb push {} /data/local/tmp/".format(daq))
    os.system("adb push input.txt /data/local/tmp/")
    os.system("adb push {} /data/local/tmp/dnn_retrieve_result".format(dnn_retrieve_result))
    os.system('adb shell "LD_LIBRARY_PATH=/data/local/tmp/ /data/local/tmp/dnn_retrieve_result /data/local/tmp/{} {} {} /data/local/tmp/input.txt"'.format(os.path.basename(daq), output_name, 1 if len(table_file) != 0 else 0))
    os.system("adb shell rm /data/local/tmp/input.txt")
    os.system("adb shell rm /data/local/tmp/dnn_retrieve_result")
    os.system("adb pull /data/local/tmp/result {}".format(txt))
    os.system("adb shell rm /data/local/tmp/result")
    os.system("adb shell rm /data/local/tmp/{}".format(os.path.basename(daq)))
    os.system("rm {}".format(daq))
    os.system("rm input.txt")
    actual = np.loadtxt(txt)
    assert not np.any(np.isnan(actual))
    os.system("rm {}".format(txt))

    return actual

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test onnx model on nnapi')
    parser.add_argument('onnx', type=str, help='onnx model file')
    parser.add_argument('onnx2daq', type=str, help='onnx2daq binary file')
    parser.add_argument('dnn_retrieve_result', type=str, help='dnn_retrieve_result binary file')
    parser.add_argument('output', type=str, help='Output name of the model')
    parser.add_argument('test_data_dir', type=str, help='e.g. test_data_set_0')
    parser.add_argument('--table_file', type=str, help='table file for 8-bit quantization', default='')
    parser.add_argument('--res_shape', type=str, help='The shape of result in nhwc, such as [1000] or [1,224,224,3]', default='-1')

    args = parser.parse_args()
    args.quant = len(args.table_file) == 0
    import ast
    args.res_shape = ast.literal_eval(args.res_shape)
    if type(args.res_shape) == int:
        args.res_shape = [args.res_shape]

    # Load inputs
    inputs = []
    inputs_num = len(glob.glob(os.path.join(args.test_data_dir, 'input_*.pb')))
    for i in range(inputs_num):
        input_file = os.path.join(args.test_data_dir, 'input_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        np_arr = numpy_helper.to_array(tensor)
        if args.quant:
            np_arr = np_arr.astype(np.uint8)
        inputs.append(np_arr)

    # Load reference outputs
    ref_outputs = []
    ref_outputs_num = len(glob.glob(os.path.join(args.test_data_dir, 'output_*.pb')))
    for i in range(ref_outputs_num):
        output_file = os.path.join(args.test_data_dir, 'output_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        ref_outputs.append(numpy_helper.to_array(tensor))

    assert inputs_num == ref_outputs_num
    for i in range(inputs_num):
        actual = run(inputs[i], args.onnx, args.onnx2daq, args.dnn_retrieve_result, args.output, args.table_file)
        if len(args.res_shape) == 4:
            actual = np.transpose(actual.reshape(args.res_shape), [0, 3, 1, 2]).flatten()
        expected = ref_outputs[i].flatten()

        print('====================')
        try:
            print("Max relative diff: {}".format(np.max(np.abs(expected - actual) / expected)))
            np.testing.assert_array_almost_equal(expected, actual, decimal=3)
            print('No.{} in {} passed'.format(i, args.test_data_dir))
        except AssertionError as e:
            print('No.{} in {} failed'.format(i, args.test_data_dir))
            print(str(e))
            print(expected)
            print('-----')
            print(actual)
