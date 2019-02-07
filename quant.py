import argparse
import os
from typing import List, Tuple

import numpy as np
import onnx
from onnx import shape_inference, optimizer

qmin = 0
qmax = 255

maxs = {}
mins = {}
scales = {}
zps = {}


def update_scale_and_zp(key, arr):
    if key not in maxs:
        maxs[key] = np.max(arr)
    if key not in mins:
        mins[key] = np.min(arr)
    maxs[key] = max(maxs[key], np.max(arr))
    mins[key] = min(mins[key], np.min(arr))
    scales[key] = (maxs[key] - mins[key]) / (qmax - qmin)
    zp = qmin - mins[key] / scales[key]
    zp = max(qmin, zp)
    zp = min(qmax, zp)
    zp = int(round(zp))
    zps[key] = zp


def argmax(d):
    assert isinstance(d, dict)
    ret = None
    for key in d:
        if ret is None or d[key] > d[ret]:
            ret = key
    return ret


def modify_pb(m: onnx.ModelProto) -> None:
    """
    Modify proto buffers when all quantization infos are set correctly
    :param m: the model
    """
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
                    w = np.round(np.clip(w, qmin, qmax))
                    t.raw_data = w.astype(np.uint8).tobytes()
                    t.data_type = onnx.TensorProto.UINT8
                    del t.float_data[:]
                if len(node.input) == 3 and t.name == bias:
                    assert len(t.raw_data) == 0
                    b = np.array(t.float_data)
                    b /= scales[bias]
                    t.raw_data = np.round(b).astype(np.int32).tobytes()
                    t.data_type = onnx.TensorProto.INT32
                    del t.float_data[:]


def set_scales_of_weight(m):
    for node in m.graph.node:
        if node.op_type == 'Conv':
            weight = node.input[1]
            for t in m.graph.initializer:
                if t.name == weight:
                    update_scale_and_zp(weight, t.raw_data if len(t.float_data) == 0 else t.float_data)


def get_initializer(m, name):
    for t in m.graph.initializer:
        if t.name == name:
            from onnx import numpy_helper
            return numpy_helper.to_array(t)


def make_scales_right(m: onnx.ModelProto) -> None:
    """
    There are some requirement for quantization info, we assert and infer them here
    Some layer sequence need multiple runs to make infos right, like concat->conv->relu
    The range(3) is set arbitrarily, but it must not be lower than the number of if branch
    :param m: the model
    """
    for _ in range(3):
        for node in m.graph.node:
            if node.op_type == 'Relu':
                ipt, opt = node.input[0], node.output[0]
                for l in (scales, zps, mins, maxs):
                    l[ipt] = l[opt]
            elif node.op_type == 'Concat':
                k = argmax({k: v for (k, v) in scales.items() if k in node.input})
                for x in node.input:
                    for l in (scales, zps, mins, maxs):
                        l[x] = l[k]

        for node in m.graph.node:
            if node.op_type == 'Conv':
                ipt, weight, output = node.input[0], node.input[1], node.output[0]
                assert scales[ipt] * scales[weight] < scales[output]


def set_quant_info_of_bias(m: onnx.ModelProto) -> None:
    """
    NNAPI requires scales[bias] equals scales[input]*scales[weight] and zps[scale]=0
    :param m: the model
    """
    for node in m.graph.node:
        if node.op_type == 'Conv':
            ipt = node.input[0]
            weight = node.input[1]
            if len(node.input) == 3:
                bias = node.input[2]
                zps[bias] = 0
                scales[bias] = scales[ipt] * scales[weight]


def get_quant_list(m: onnx.ModelProto) -> Tuple[List[str], List[str], List[str], Tuple[str, str, str]]:
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


def collect_scales_of_features3(model: onnx.ModelProto, image_dir: str) -> None:
    """
    Collect infos of features by running model in onnxruntime
    :param model: the model
    :param image_dir: the directory of images
    """
    from queue import Queue
    import threading
    import glob

    q = Queue()

    def worker(paths):
        def read_img(path, norm=True):
            import cv2
            a = cv2.imread(path)
            a = cv2.resize(a, (224, 224))
            a = a.astype(np.float32)
            if norm:
                a /= 255
                a -= [0.485, 0.456, 0.406]
                a /= [0.229, 0.224, 0.225]
            a = np.moveaxis(a, -1, 0)
            return a

        bs = 128
        for i in range(0, len(paths), bs):
            xs = np.stack(list(map(lambda x: read_img(x, True), paths[i:i+bs])))
            q.put(xs)
        q.put(None)

    threads = []
    num_worker_threads = 1
    for i in range(num_worker_threads):
        t = threading.Thread(target=worker, args=(glob.glob(os.path.join(image_dir, '*.JPEG')),))
        t.start()
        threads.append(t)

    while True:
        xs = q.get()
        if xs is None:
            break
        update_scale_and_zp('data', xs)
        import onnxruntime as rt
        sess = rt.InferenceSession(model.SerializeToString())
        from collections import OrderedDict
        output_names = [x.name for x in sess.get_outputs()]
        res = OrderedDict(zip(output_names, sess.run(None, {'data': xs})))
        for key in res:
            update_scale_and_zp(key, res[key])

    for t in threads:
        t.join()


def collect_scales_of_features2(onnx_file, image_dir):
    import cv2
    import glob
    import os
    paths = []
    for i, path in enumerate(glob.glob(os.path.join(image_dir, '*.JPEG'))):
        paths.append(path)
        if i % 256 == 0:
            def read_img(img_path, norm=True):
                a = cv2.imread(img_path)
                a = cv2.resize(a, (224, 224))
                a = a.astype(np.float32)
                if norm:
                    a /= 255
                    a -= [0.485, 0.456, 0.406]
                    a /= [0.229, 0.224, 0.225]
                a = np.moveaxis(a, -1, 0)
                return a
            xs = np.stack(list(map(lambda x: read_img(x, True), paths)))
            update_scale_and_zp('data', xs)
            import onnxruntime as rt
            sess = rt.InferenceSession(onnx_file)
            from collections import OrderedDict
            output_names = [x.name for x in sess.get_outputs()]
            res = OrderedDict(zip(output_names, sess.run(None, {'data': xs})))
            for key in res:
                update_scale_and_zp(key, res[key])
            paths.clear()


def collect_scales_of_features(onnx_file, input_pb):
    inputs = {}
    from onnx import numpy_helper
    with open(input_pb, 'rb') as f:
        tensor = onnx.TensorProto()
        tensor.ParseFromString(f.read())
        x = numpy_helper.to_array(tensor)
    x = np.round(x)
    inputs['data'] = x

    import onnxruntime as rt
    m = onnx.load(onnx_file)
    sess = rt.InferenceSession(m.SerializeToString())
    from collections import OrderedDict
    output_names = [x.name for x in sess.get_outputs()]
    fpres = OrderedDict(zip(output_names, sess.run(None, {'data': x})))
    for i, vi in enumerate(m.graph.value_info):
        if i > 0:
            prev_vi = m.graph.value_info[i - 1]
            m.graph.input.extend([prev_vi])
            for node in m.graph.node:
                for j in range(len(node.output)):
                    if node.output[j] == prev_vi.name:
                        node.output[j] = node.output[j] + "_dummy"
        del m.graph.output[:]
        m.graph.output.extend([vi])

        # onnx.save(m, "/home/daquexian/models/mobilenetv2-1.0/imm3-mobilenetv2-1.0.onnx")
        sess = rt.InferenceSession(m.SerializeToString())
        assert len(sess.get_outputs()) == 1
        output_name = sess.get_outputs()[0].name

        res = sess.run(None, inputs)[0]
        update_scale_and_zp(output_name, res)
        res = zps[output_name] + res / scales[output_name]
        res = np.round(np.clip(res, qmin, qmax))
        res = (res - zps[output_name]) * scales[output_name]
        inputs[output_name] = res
        print(f"{output_name} ok")

    print("finish")

    '''
    for _ in range(0):
        x = np.random.random((1, 3, 224, 224)).astype(np.float32) * 255
        from collections import OrderedDict
        res = OrderedDict(zip(output_names, sess.run(None, {input_name: x})))
        for x in res:
            update_scale(x, res[x])
    '''


def quant_weight(m: onnx.ModelProto) -> None:
    """
    quant weights before collecting min and max, for simulating the effect of quantization
    :param m: the model
    """
    for node in m.graph.node:
        if node.op_type == 'Conv':
            weight = node.input[1]
            for t in m.graph.initializer:
                if t.name == weight:
                    assert len(t.raw_data) == 0
                    w = np.array(t.float_data)
                    w = zps[weight] + w / scales[weight]
                    w = np.round(np.clip(w, qmin, qmax))
                    w = (w - zps[weight]) * scales[weight]
                    del t.float_data[:]
                    t.float_data.extend(w)


def move_raw_to_float(m: onnx.ModelProto) -> None:
    """
    values of initializers may be stored in float_data or raw_data, if in raw_data, we move them to float_data
    for convenience
    :param m: the model
    """
    for t in m.graph.initializer:
        if t.data_type == onnx.TensorProto.FLOAT:
            if len(t.float_data) == 0:
                import struct
                import itertools
                it = struct.iter_unpack('f', t.raw_data)
                t.float_data.extend(itertools.chain.from_iterable(it))
                t.raw_data = bytes(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='model filename', type=str)
    parser.add_argument('table', help='name of the file storing scales and zeropoints', type=str)
    parser.add_argument('--input_pb', help='pb file storing model input', type=str)
    parser.add_argument('--image_dir', help='directory storing model input', type=str)
    args = parser.parse_args()

    model_path = args.model
    table_name = args.table
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)

    m = onnx.load(os.path.join(model_dir, model_name))
    move_raw_to_float(m)

    passes = ['fuse_bn_into_conv']
    m = optimizer.optimize(m, passes)
    m = shape_inference.infer_shapes(m)
    import copy
    model_opt = copy.deepcopy(m)
    del m.graph.output[:]
    m.graph.output.extend(m.graph.value_info)

    features, weights, biases, three_tuples = get_quant_list(m)

    set_scales_of_weight(m)
    quant_weight(m)

    collect_scales_of_features3(m, args.image_dir)
    make_scales_right(m)
    set_quant_info_of_bias(m)

    modify_pb(model_opt)

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


    onnx.save(model_opt, os.path.join(model_dir, "quant-" + model_name))


if __name__ == '__main__':
    main()
