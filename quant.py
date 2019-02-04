import onnx
from onnx import helper, shape_inference, optimizer
import numpy as np
import argparse

qmin = 0
qmax = 255

maxs = {}
mins = {}
scales = {}
zps = {}

def max_min_diff(arr):
    return np.max(arr) - np.min(arr)


def update_scale(key, arr):
    if key not in maxs:
        maxs[key] = np.max(arr)
    if key not in mins:
        mins[key] = np.min(arr)
    maxs[key] = max(maxs[key], np.max(arr))
    mins[key] = min(mins[key], np.min(arr))
    scales[key] = (maxs[key] - mins[key]) / (qmax - qmin)


def argmax(d):
    assert isinstance(d, dict)
    ret = d.keys()[0]
    for key in d:
        if d[key] > d[ret]:
            ret = key
    return ret


def set_zeropoints(tensors):
    for i, key in enumerate(tensors):
        zp = qmin - mins[key] / scales[key]
        zp = max(qmin, zp)
        zp = min(qmax, zp)
        zp = int(round(zp))
        zps[key] = zp


def modify_pb(m):
    for node in m.graph.node:
        if node.op_type == 'Conv':
            weight = node.input[1]
            if len(node.input) == 3:
                bias = node.input[2]
            for t in m.graph.initializer:
                if t.name == weight:
                    assert len(t.raw_data) == 0
                    w = np.array(t.float_data)
                    w = zps[weight] + w / scales[weight]
                    w = np.clip(w, qmin, qmax)
                    t.raw_data = w.astype(np.uint8).tobytes()
                    t.data_type = onnx.TensorProto.UINT8
                    del t.float_data[:]
                if len(node.input) == 3 and t.name == bias:
                    assert len(t.raw_data) == 0
                    b = np.array(t.float_data)
                    b /= scales[bias]
                    t.raw_data = b.astype(np.int32).tobytes()
                    t.data_type = onnx.TensorProto.INT32
                    del t.float_data[:]


def set_scales_of_weight(m):
    for node in m.graph.node:
        if node.op_type == 'Conv':
            weight = node.input[1]
            for t in m.graph.initializer:
                if t.name == weight:
                    update_scale(weight, t.raw_data if len(t.float_data) == 0 else t.float_data)


def make_scales_right(m):
    for node in m.graph.node:
        if node.op_type == 'Relu':
            ipt, opt = node.input[0], node.output[0]
            for l in (scales, mins, maxs):
                l[opt] = l[ipt]
            print("Set scale[{}] to {}".format(opt, scales[ipt]))
        elif node.op_type == 'Concat':
            k = argmax({k: v for (k, v) in scales.items() if k in node.input})
            for x in node.input:
                for l in (scales, mins, maxs):
                    l[x] = l[k]
                print("Set scale[{}] to {}".format(x, scales[k]))

    for node in m.graph.node:
        if node.op_type == 'Conv':
            ipt, weight, output = node.input[0], node.input[1], node.output[0]
            assert scales[ipt] * scales[weight] < scales[output]


def set_quant_info_of_bias(m):
    for node in m.graph.node:
        if node.op_type == 'Conv':
            ipt = node.input[0]
            weight = node.input[1]
            if len(node.input) == 3:
                bias = node.input[2]
                zps[bias] = 0
                scales[bias] = scales[ipt] * scales[weight]


def get_quant_list(m):
    features = [x.name for x in m.graph.output]
    weights = []
    biases = []
    three_tuple = []
    for node in m.graph.node:
        if node.op_type == 'Conv':
            ipt = node.input[0]
            weight = node.input[1]
            weights.append(weight)
            bias = None
            if len(node.input) == 3:
                bias = node.input[2]
                biases.append(bias)
            three_tuple.append((ipt, weight, bias))
    return features, weights, biases, three_tuple


def collect_scales_of_features():
    import onnxruntime as rt
    sess = rt.InferenceSession("/home/daquexian/models/mobilenetv2-1.0/imm-mobilenetv2-1.0.onnx")
    input_name = sess.get_inputs()[0].name
    output_names = [x.name for x in sess.get_outputs()]
    '''
    for _ in range(1):
        x = np.random.random((1, 3, 224, 224)).astype(np.float32)
        from collections import OrderedDict
        res = OrderedDict(zip(output_names, sess.run(None, {input_name: x})))
        for x in res:
            update_scale(x, res[x])
    '''
    from onnx import numpy_helper
    with open('/home/daquexian/models/mobilenetv2-1.0/test_data_set_0/input_0.pb', 'rb') as f:
        tensor = onnx.TensorProto()
        tensor.ParseFromString(f.read())
        x = numpy_helper.to_array(tensor)
    from collections import OrderedDict
    res = OrderedDict(zip(output_names, sess.run(None, {input_name: x})))
    for x in res:
        update_scale(x, res[x])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('table', help='name of the file storing scales and zeropoints', type=str)
    args = parser.parse_args()

    table_name = args.table

    m = onnx.load('/home/daquexian/models/mobilenetv2-1.0/mobilenetv2-1.0.onnx')
    passes = ['fuse_bn_into_conv']
    m = optimizer.optimize(m, passes)
    m = shape_inference.infer_shapes(m)
    del m.graph.output[:]
    m.graph.output.extend(m.graph.value_info)
    onnx.save(m, "/home/daquexian/models/mobilenetv2-1.0/imm-mobilenetv2-1.0.onnx")

    m = onnx.load("/home/daquexian/models/mobilenetv2-1.0/imm-mobilenetv2-1.0.onnx")
    features, weights, biases, three_tuples = get_quant_list(m)

    set_scales_of_weight(m)

    collect_scales_of_features()

    scales['data'] = 1
    zps['data'] = 0

    make_scales_right(m)

    set_quant_info_of_bias(m)

    set_zeropoints(features + weights)

    modify_pb(m)

    with open(table_name, 'w') as f:
        for i, key in enumerate(['data'] + features):
            # 1 is the number of the following elements, may be channels_num or 0 for scale and zeropoint in the future
            f.write('{} {} {} {} {} quant8_asymm\n'.format(key, 1, scales[key], 1, zps[key]))
        for i, key in enumerate(weights):
            # 1 is the number of the following elements, may be channels_num or 0 for scale and zeropoint in the future
            f.write('{} {} {} {} {} quant8_asymm\n'.format(key + "_conv_w", 1, scales[key], 1, zps[key]))
        for i, t in enumerate(three_tuples):
            if t[2] is None:
                continue
            # -2 means scales of 2 tensors multiply
            f.write('{} -2 {} {} {} int32\n'.format(t[2] + "_conv_b", t[0], t[1] + '_conv_w', 0))


    onnx.save(m, "/home/daquexian/models/mobilenetv2-1.0/quant-mobilenetv2-1.0.onnx")


if __name__ == '__main__':
    main()
