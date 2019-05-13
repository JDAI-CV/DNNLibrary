import onnx
from onnx import numpy_helper
import os
import glob
import numpy as np
import tempfile


def convert(onnx2daq, onnx, daq, table_file=''):
    daq = "temp.daq"
    os.system("{} {} {} {}".format(onnx2daq, onnx, daq, table_file))
    print("Converted to daq")


def finish(model):
    os.system("adb shell rm /data/local/tmp/{}".format(os.path.basename(model)))
    if model[-4:] == '.daq':
        os.system("rm {}".format(model))


def run(input_arrs, daq, dnn_retrieve_result, quant_input=False, quant_output=False):
    input_txts = []
    for i, input_arr in enumerate(input_arrs):
        nchw_shape = input_arr.shape
        nhwc_shape = (nchw_shape[0], nchw_shape[2], nchw_shape[3], nchw_shape[1])
        nhwc_input = np.moveaxis(input_arr, 1, -1)
        assert nhwc_input.shape == nhwc_shape
        input_txt = 'input{}.txt'.format(i)
        np.savetxt(input_txt, nhwc_input.flatten(), delimiter='\n')
        input_txts.append(input_txt)
    input_txts_arg = " ".join(input_txts)

    txt = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))
    os.system("adb push {} /data/local/tmp/".format(input_txts_arg))
    os.system("adb push {} /data/local/tmp/dnn_retrieve_result".format(dnn_retrieve_result))
    os.system('adb shell "LD_LIBRARY_PATH=/data/local/tmp/ /data/local/tmp/dnn_retrieve_result /data/local/tmp/{} {} {} {}"'.format(os.path.basename(daq), "--quant_input" if quant_input else "", "--quant_output" if quant_output else "", " ".join(map(lambda x: "/data/local/tmp/" + x, input_txts))))
    os.system("adb shell rm {}".format(input_txts_arg))
    os.system("adb shell rm /data/local/tmp/dnn_retrieve_result")
    os.system("adb pull /data/local/tmp/result {}".format(txt))
    os.system("adb shell rm /data/local/tmp/result")
    os.system("rm {}".format(input_txts_arg))
    actual = np.loadtxt(txt)
    assert not np.any(np.isnan(actual))

    return actual

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test onnx model on nnapi')
    parser.add_argument('onnx', type=str, help='onnx model file')
    parser.add_argument('dnn_retrieve_result', type=str, help='dnn_retrieve_result binary file')
    parser.add_argument('test_data_dir', type=str, help='e.g. test_data_set_0')
    parser.add_argument('--onnx2daq', type=str, help='onnx2daq binary file')
    parser.add_argument('--table_file', type=str, help='table file for 8-bit quantization', default='')
    parser.add_argument('--quant_input', help='whether the input is quant8', action='store_true')
    parser.add_argument('--quant_output', help='whether the output is quant8', action='store_true')
    parser.add_argument('--read_onnx', action='store_true', help='Read ONNX model directly, onnx2daq will not be used')

    args = parser.parse_args()

    # Load inputs
    inputs = []
    inputs_num = len(glob.glob(os.path.join(args.test_data_dir, 'input_*.pb')))
    for i in range(inputs_num):
        input_file = os.path.join(args.test_data_dir, 'input_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        np_arr = numpy_helper.to_array(tensor)
        if args.quant_input:
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

    if not args.read_onnx:
        model = "temp.daq"
        convert(args.onnx2daq, args.onnx, model, args.table_file)
    else:
        model = args.onnx
    os.system("adb push {} /data/local/tmp/".format(model))
    for i in range(inputs_num):
        actual = run(inputs, model, args.dnn_retrieve_result, args.quant_input, args.quant_output)
        expected = ref_outputs[i].flatten()

        print('====================')
        try:
            print("Max relative diff: {}".format(np.max(np.abs(expected - actual) / expected)))
            np.testing.assert_array_almost_equal(expected, actual, decimal=3)
            print('No.{} in {} passed'.format(i, args.test_data_dir))
        except (AssertionError, ValueError) as e:
            print('No.{} in {} failed'.format(i, args.test_data_dir))
            print(str(e))
            print(expected)
            print('-----')
            print(actual)
            print(np.argmax(actual))

    finish(model)
