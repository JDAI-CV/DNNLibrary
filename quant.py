import argparse
import copy
import collections
import os
from typing import List, Tuple
import itertools

import numpy as np
import onnx
from onnx import shape_inference, optimizer

qmin = 0
qmax = 255

maxs = {}
mins = {}
scales = {}
zps = {}


class OrderedSet(collections.Set):
    '''
    Althought there is a warning that
    "Using or importing the ABCs from 'collections' instead of
    from 'collections.abc' is deprecated, and in 3.8 it will
    stop working"
    but it is caused by python's own code, so just ignore it
    '''

    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)


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


def modify_pb(m: onnx.ModelProto, quant_layers: List[str]) -> None:
    """
    Modify proto buffers when all quantization infos are set correctly
    :param m: the model
    :param quant_layers: layers need to be quantized
    """
    for node in m.graph.node:
        if node.name not in quant_layers:
            continue
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


def add_features_to_output(m):
    del m.graph.output[:]
    m.graph.output.extend(m.graph.value_info)


def optimize(m):
    passes = ['fuse_bn_into_conv']
    m = optimizer.optimize(m, passes)
    m = shape_inference.infer_shapes(m)
    return m


def set_scales_of_weight(m, quant_layers: List[str]):
    for node in m.graph.node:
        if node.name not in quant_layers:
            continue
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


def make_scales_right(m: onnx.ModelProto, quant_layers: List[str], quant_tensors: List[str]) -> None:
    """
    There are some requirement for quantization info, we assert and infer them here
    Some layer sequence need multiple runs to make infos right, like concat->conv->relu
    The range(3) is set arbitrarily, but it must not be lower than the number of if branch
    :param m: the model
    :param quant_layers: layers need to be quantized
    :param quant_tensors: tensors need to be quantized
    """
    for _ in range(3):
        for node in m.graph.node:
            if node.op_type == 'Relu':
                ipt, opt = node.input[0], node.output[0]
                if ipt in quant_tensors and opt in quant_tensors:
                    for l in (scales, zps, mins, maxs):
                        l[ipt] = l[opt]
            elif node.op_type == 'Concat':
                assert all([x in quant_tensors for x in node.input]) or all(
                    [x not in quant_tensors for x in node.input])
                if all([x in quant_tensors for x in node.input]):
                    k = argmax({k: v for (k, v) in scales.items() if k in node.input})
                    for x in node.input:
                        for l in (scales, zps, mins, maxs):
                            l[x] = l[k]

        for node in m.graph.node:
            if node.name not in quant_layers:
                continue
            if node.op_type == 'Conv':
                ipt, weight, output = node.input[0], node.input[1], node.output[0]
                assert scales[ipt] * scales[weight] < scales[output]


def set_quant_info_of_bias(m: onnx.ModelProto, quant_layers: List[str]) -> None:
    """
    NNAPI requires scales[bias] equals scales[input]*scales[weight] and zps[scale]=0
    :param m: the model
    :param quant_layers: layers need to be quantized
    """
    for node in m.graph.node:
        if node.name not in quant_layers:
            continue
        if node.op_type == 'Conv':
            ipt = node.input[0]
            weight = node.input[1]
            if len(node.input) == 3:
                bias = node.input[2]
                zps[bias] = 0
                scales[bias] = scales[ipt] * scales[weight]


def get_quant_list(m: onnx.ModelProto, quant_layers: List[str]) -> Tuple[List[str], List[str], Tuple[str, str, str]]:
    weights = []
    biases = []
    three_tuple = []
    for node in m.graph.node:
        if node.name not in quant_layers:
            continue
        if node.op_type == 'Conv':
            ipt = node.input[0]
            weight = node.input[1]
            weights.append(weight)
            bias = None
            if len(node.input) == 3:
                bias = node.input[2]
                biases.append(bias)
            three_tuple.append((ipt, weight, bias))
    return weights, biases, three_tuple


def collect_scales_of_features(model: onnx.ModelProto, image_dir: str,
                               features: List[str] = None, batch_size=56,
                               num_workers=1, show_cls=False) -> None:
    """
    Collect infos of features by running model in onnxruntime
    :param model: the model
    :param image_dir: the directory of images
    :param features: names of features that need to collect, None for all features
    :param batch_size: batch size for net forward
    :param num_workers: number of thread fetching and preprocessing images
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
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            if norm:
                a /= 255
                # mean and std for RGB images
                a -= [0.485, 0.456, 0.406]
                a /= [0.229, 0.224, 0.225]
            a = np.moveaxis(a, -1, 0)
            return a

        for i in range(0, len(paths), batch_size):
            xs = np.stack(list(map(lambda x: read_img(x, True), paths[i:i + batch_size])))
            q.put(xs)
        q.put(None)

    image_exts = ['JPEG', 'jpg', 'jpeg', 'png']
    filenames = list(itertools.chain(*[glob.glob(os.path.join(image_dir, '**/*.' + ext), recursive=True)
                                       for ext in image_exts]))
    import random
    random.shuffle(filenames)
    filenames = filenames[:10000]
    file_num = len(filenames)
    num_workers = min((file_num + batch_size - 1) // batch_size, num_workers)
    chunk_num = file_num // num_workers
    threads = []
    for i in range(num_workers):
        t = threading.Thread(target=worker, args=((filenames[chunk_num * i:chunk_num * (i + 1)],)))
        t.start()
        threads.append(t)

    import onnxruntime as rt
    sess = rt.InferenceSession(model.SerializeToString())
    all_outputs = [x.name for x in sess.get_outputs()]
    features = all_outputs if features is None else list(OrderedSet(features) & OrderedSet(all_outputs))
    i = 0
    done_workers = 0
    if show_cls:
        from collections import defaultdict, Counter
        d = defaultdict(int)
    while True:
        xs = q.get()
        if xs is None:
            done_workers += 1
            if done_workers == num_workers:
                break
            continue
        i += xs.shape[0]
        update_scale_and_zp('data', xs)
        from collections import OrderedDict
        res = OrderedDict(zip(features, sess.run(features, {'data': xs})))
        if show_cls:
            cls = Counter(res['mobilenetv20_output_pred_fwd'].squeeze(axis=(2,3)).argmax(axis=1))
            for key in cls:
                d[key] += cls[key]
        for key in res:
            update_scale_and_zp(key, res[key])
        q.task_done()
        print("{}/{}".format(i, file_num))

    if show_cls:
        print(d)

    for t in threads:
        t.join()


def quant_weight(m: onnx.ModelProto, quant_layers: List[str]) -> None:
    """
    quant weights before collecting min and max, for simulating the effect of quantization
    :param m: the model
    :param quant_layers: layers need to be quantized
    """
    for node in m.graph.node:
        if node.name not in quant_layers:
            continue
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
    parser.add_argument('--image_dir', help='directory storing model input', type=str)
    parser.add_argument('--dequantize_after', help='The name of tensor we want to insert dequantize layer after',
                        type=str, default='')
    parser.add_argument('--float_input', help='whether the input of model is float, only for Android 29 (NNAPI 1.2)',
                        action="store_true")
    parser.add_argument('--quantize_after',
                        help='The name of tensor we want to insert quantize layer after, '
                             'only for Android 29 (NNAPI 1.2)',
                        type=str, default='')
    parser.add_argument('--batch_size',
                        help='batch size for net forwarding',
                        type=int,
                        default=56)
    parser.add_argument('--num_workers',
                        help='number of threads fetching and preprocessing images',
                        type=int,
                        default=1)
    args = parser.parse_args()

    model_path = args.model
    table_name = args.table
    float_input = args.float_input
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    if not float_input:
        assert args.quantize_after == ''
        args.quantize_after = 'data'

    m: onnx.ModelProto = onnx.load(os.path.join(model_dir, model_name))
    move_raw_to_float(m)

    m = optimize(m)

    model_opt = copy.deepcopy(m)
    add_features_to_output(m)

    quant_after_tensors = [args.quantize_after]
    dequant_after_tensors = [args.dequantize_after]
    inferred_quant_tensors, quant_layers = get_quant_layers_and_tensors(m, quant_after_tensors, dequant_after_tensors)

    weights, biases, three_tuples = get_quant_list(m, quant_layers)

    set_scales_of_weight(m, quant_layers)
    quant_weight(m, quant_layers)

    collect_scales_of_features(m, args.image_dir, inferred_quant_tensors, args.batch_size, args.num_workers)
    make_scales_right(m, quant_layers, inferred_quant_tensors)
    set_quant_info_of_bias(m, quant_layers)

    modify_pb(model_opt, quant_layers)

    with open(table_name, 'w') as f:
        for i, key in enumerate(['data'] + inferred_quant_tensors):
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
        for x in dequant_after_tensors:
            f.write('dequantize after: {}'.format(x))

    onnx.save(model_opt, os.path.join(model_dir, "quant-" + model_name))


def get_quant_layers_and_tensors(m, quant_after_tensors, dequant_after_tensors):
    inferred_quant_tensors = quant_after_tensors[:]
    quant_layers = []
    for node in m.graph.node:
        if node.input[0] in inferred_quant_tensors and node.input[0] not in dequant_after_tensors:
            inferred_quant_tensors.extend([x for x in node.output])
            quant_layers.append(node.name)
    inferred_quant_tensors = list(OrderedSet(inferred_quant_tensors) & OrderedSet([x.name for x in m.graph.output]))
    return inferred_quant_tensors, quant_layers


if __name__ == '__main__':
    main()
