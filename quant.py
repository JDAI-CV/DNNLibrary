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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('table', help='name of the file storing scales and zeropoints', type=str)
    args = parser.parse_args()

    table_name = args.table

    # m = onnx.load('/home/daquexian/models/mobilenetv2-1.0/mobilenetv2-1.0.onnx')
    # passes = ['fuse_bn_into_conv']
    # m = optimizer.optimize(m, passes)
    # m = shape_inference.infer_shapes(m)
    # del m.graph.output[:]
    # m.graph.output.extend(m.graph.value_info)
    # print(len(m.graph.value_info))
    # print(len(m.graph.output))
    # onnx.save(m, "/home/daquexian/models/mobilenetv2-1.0/imm-mobilenetv2-1.0.onnx")

    m = onnx.load("/home/daquexian/models/mobilenetv2-1.0/imm-mobilenetv2-1.0.onnx")
    weights = []
    output_to_weight = {}
    for node in m.graph.node:
        if node.op_type == 'Conv':
            weight = node.input[1]
            output = node.output[0]
            weights.append(weight)
            output_to_weight[output] = weight
            for t in m.graph.initializer:
                if t.name == weight:
                    update_scale(weight, t.raw_data if len(t.float_data) == 0 else t.float_data)


    import onnxruntime as rt
    sess = rt.InferenceSession("/home/daquexian/models/mobilenetv2-1.0/imm-mobilenetv2-1.0.onnx")
    input_name = sess.get_inputs()[0].name
    output_names = [x.name for x in sess.get_outputs()]

    for _ in range(1):
        x = np.random.random((1,3,224,224)).astype(np.float32)
        from collections import OrderedDict
        res = OrderedDict(zip(output_names, sess.run(None, {input_name: x})))
        for x in res:
            update_scale(x, res[x])


    for i, key in enumerate(output_names):
        if i != 0:
            prev_key = output_names[i - 1]
            if 'conv' in key and not scales[key] > scales[prev_key] * scales[output_to_weight[key]]:
                scales[key] = 2 * scales[prev_key] * scales[output_to_weight[key]]
                print("Set scale[{}] to {}".format(key, scales[key]))
        for node in m.graph.node:
            if node.op_type == 'Relu':
                if node.input[0] == key:
                    for l in (scales, mins, maxs):
                        l[node.output[0]] = l[key]
                    print("Set scale[{}] to {}".format(node.output[0], scales[key]))
            elif node.op_type == 'Concat':
                k = argmax({k: v for (k, v) in scales.items() if k in node.input})
                for x in node.input:
                    for l in (scales, mins, maxs):
                        l[x] = l[k]
                    print("Set scale[{}] to {}".format(x, scales[k]))


    with open(table_name, 'w') as f:
        for i, key in enumerate(output_names + weights):
            zp = qmin - mins[key] / scales[key]
            zp = max(qmin, zp)
            zp = min(qmax, zp)
            zp = int(round(zp))
            zps[key] = zp

            # 1 is the number of the following elements, may be channels_num or 0 for scale and zeropoint in the future
            f.write('{} {} {} {} {}\n'.format(key, 1, scales[key], 1, zps[key]))


    for weight in weights:
        for t in m.graph.initializer:
            if t.name == weight:
                assert len(t.raw_data) == 0
                w = np.array(t.float_data)
                w = zps[weight] + w / scales[weight]
                w = np.clip(w, qmin, qmax)
                t.raw_data = w.tobytes()
                t.data_type = t.DataType.
                del t.float_data[:]
    
    onnx.save(m, "/home/daquexian/models/mobilenetv2-1.0/quant-mobilenetv2-1.0.onnx")


if __name__ == '__main__':
    main()
