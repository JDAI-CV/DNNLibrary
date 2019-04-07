import io
import yaml
from enum import Enum
from typing import Tuple

str_io = io.StringIO()


class Target(Enum):
    ModelBuilder = 1
    OnnxConverter = 2


def cogout(txt):
    print(txt, end='', file=str_io)


def cogoutl(txt):
    print(txt, file=str_io)


def get_param(elem: dict) -> Tuple[str, str]:
    """
    get parameter in function signature from yaml element, e.g.
    -
        name: input
        type: str
    produces ["const std::string &"] and ["input_name"]
    :param elem: yaml element of a input
    :return: A tuple, (type, name)
    """
    if elem['cpp_type'] == 'str':
        return 'const std::string &', elem['name']
    elif elem['cpp_type'] == 'optional_str':
        return 'const std::optional<std::string> &', elem['name']
    elif elem['cpp_type'] == 'str_list':
        return 'const std::vector<std::string> &', elem['name']
    elif elem['cpp_type'] == 'int32_list':
        return 'const std::vector<int32_t> &', elem['name']
    else:
        return elem['cpp_type'], elem['name']


def add_optional_bias():
    return '''uint32_t bias_idx_val;
        css bias_val = bias.value_or(weight + "_b");
        if (!bias.has_value()) {
            const auto weight_dimen = shaper_[weight];
            const Shape bias_dimen{weight_dimen[0]};
            const auto &weight_type = operand_types_.at(weight).type;
            if (weight_type == Type::TENSOR_FLOAT32) {
                bias_idx_val = FillOperand(bias_val, {Type::TENSOR_FLOAT32, bias_dimen}, 0.f);
            } else if (weight_type == Type::TENSOR_QUANT8_ASYMM) {
                const auto input_scale = operand_types_.at(input).operandType.scale;
                const auto weight_scale = operand_types_.at(weight).operandType.scale;
                bias_idx_val = FillOperand(bias_val, 
                        {Type::TENSOR_INT32, bias_dimen, input_scale * weight_scale}, 0);
            } else {
                throw std::invalid_argument("Unknown type " + typeToStr(weight_type));
            }
        } else {
            bias_idx_val = operand_indexes_.at(bias.value());
        }
        input_indexes.push_back(bias_idx_val);'''


def add_tensor_operand(operand):
    if operand['predefined'] == 'optional_bias':
        return add_optional_bias()
    if operand['cpp_type'] == 'str':
        return '''const auto {0}_idx = operand_indexes_.at({0});
input_indexes.push_back({0}_idx);'''.format(operand['name'])
    elif operand['cpp_type'] == 'float':
        return '''const auto {0}_idx = FillOperand("input_{0}_of_" + output, {{Type::TENSOR_FLOAT32, {{1}}}}, {0}); 
input_indexes.push_back({0}_idx);'''.format(operand['name'])
    elif operand['cpp_type'] == 'int32_list':
        return '''const auto {0}_idx = AddTensorFromBuffer("input_{0}_of_" + output, &{0}[0], {{Type::TENSOR_INT32, Shape{{static_cast<uint32_t>({0}.size())}}}}); 
input_indexes.push_back({0}_idx);'''.format(operand['name'])
    elif operand['cpp_type'] == 'str_list':
        return '''for (const auto &x : {}) {{
input_indexes.push_back(operand_indexes_.at(x));
}}'''.format(operand['name'])
    else:
        raise Exception('Unknown cpp_type {}'.format(operand['cpp_type']))


def infer_cfg(cfg, target: Target):
    next_pos = 0
    for i, op in enumerate(cfg):
        if 'input' not in op:
            op['input'] = []
        if 'base_input_num' not in op or op['base_input_num'] == 1:
            op['input'].insert(0,
                               {'name': 'input', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'needed_by_shaper': True})
        elif op['base_input_num'] == 2:
            op['input'] = [{'name': 'input1', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'needed_by_shaper': True},
                           {'name': 'input2', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'needed_by_shaper': True}] \
                          + op['input']
        elif op['base_input_num'] == 'n':
            op['input'].insert(0,
                               {'name': 'inputs', 'nnapi_type': 'tensor', 'cpp_type': 'str_list',
                                'needed_by_shaper': True})
        elif op['base_input_num'] == 0:
            pass
        else:
            raise Exception()

        if not 'pos' in op:
            op['pos'] = next_pos
        next_pos = op['pos'] + 1

        if not 'output' in op:
            op['output'] = [{'name': 'output', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'needed_by_shaper': True}]
        if not 'shaper' in op:
            op['shaper'] = op['name']
        if not 'nnapi' in op:
            op['nnapi'] = op['name'].upper()
        if 'fused' not in op:
            op['fused'] = False
        if op['fused'] and target == Target.ModelBuilder:
            op['input'].append({'name': 'fuse_code', 'nnapi_type': 'scalar', 'cpp_type': 'int32_t'})
        if 'support_quant_asymm' not in op:
            op['support_quant_asymm'] = False
        for ipt in op['input']:
            if 'learnable' not in ipt:
                ipt['learnable'] = False
            if 'predefined' not in ipt:
                ipt['predefined'] = ''
            if ipt['predefined'] == 'optional_bias':
                ipt['name'] = 'bias'
                ipt['nnapi_type'] = 'tensor'
                ipt['cpp_type'] = 'optional_str'


def update_code(file: str, label: str) -> None:
    """
    replace the text surrounded by "label start" and "label end" to new_code
    :param file: the .cpp or .h file
    :param label: the label surrounds the text to be replaced
    """
    global str_io
    with open(file, 'r') as f:
        s = f.read()
        start = '// {} start\n'.format(label)
        idx1 = s.find(start) + len(start)
        end = '// {} end'.format(label)
        idx2 = s.find(end)
    with open(file, 'w') as f:
        new_s = s[:idx1] + str_io.getvalue() + s[idx2:]
        f.write(new_s)
    str_io = io.StringIO()


def main():
    """
    with open('ops.yml') as f:
        cfg = yaml.load(f)

    for i, op in enumerate(cfg):
        infer_cfg(op, Target.ModelBuilder)
        if len(op['input']) == 0:
            continue
        cogoutl('#if __ANDROID_API__ >= {}'.format(op['api']))
        ipt_opt = op['input'] + op['output']
        params = list(map(get_param, ipt_opt))
        if op['support_quant_asymm']:
            params.append(('const std::optional<QuantInfo> &', 'output_quant_info'))
        params_str = ', '.join(map(lambda param: "{} {}".format(*param), params))
        cogoutl("ModelBuilder::Index ModelBuilder::Add{}({}) {{".format(op['name'], params_str))
        tensor_input = list(filter(lambda x: x['nnapi_type'] == 'tensor', op['input']))
        scalar_input = list(filter(lambda x: x['nnapi_type'] == 'scalar', op['input']))

        cogoutl('IndexSeq input_indexes;')
        for x in tensor_input:
            cogoutl(add_tensor_operand(x))
        # cogoutl('IndexSeq input_indexes{{{}}};'.format(', '.join([x['name'] + "_idx" for x in tensor_input])))
        if len(scalar_input) > 0:
            cogoutl('AddScalarOperands(input_indexes, {});'.format(', '.join([x['name'] for x in scalar_input])))
        cogoutl('shaper_.{}({});'.format(op['shaper'],
                                         ', '.join([x['name'] for x in ipt_opt if x.get('needed_by_shaper', False)])))
        if op['input'][0]['cpp_type'] == 'str_list':
            op_type_params = ['operand_types_.at({}[0]).type'.format(op['input'][0]['name']),
                              'shaper_[{}]'.format(op['output'][0]['name'])]
        else:
            op_type_params = ['operand_types_.at({}).type'.format(op['input'][0]['name']),
                              'shaper_[{}]'.format(op['output'][0]['name'])]
        if op['support_quant_asymm']:
            op_type_params.append('output_quant_info')
        cogoutl('const OperandType operand_type = GetOperandType({});'.format(', '.join(op_type_params)))
        cogoutl('const auto output_idx = '
                'AddOperation(ANEURALNETWORKS_{}, input_indexes, operand_type)[0];'.format(op['nnapi']))
        cogout(
            '''RegisterOperand(output, output_idx, operand_type);
    return output_idx;
    }
    '''
        )
        cogoutl('#endif // __ANDROID_API__ >= {}'.format(op['api']))

    update_code('dnnlibrary/src/ModelBuilder.cpp', 'ModelBuilder auto generated methods')

    for i, op in enumerate(cfg):
        if len(op['input']) == 0:
            continue
        cogoutl('#if __ANDROID_API__ >= {}'.format(op['api']))
        ipt_opt = op['input'] + op['output']
        params = list(map(get_param, ipt_opt))
        if op['support_quant_asymm']:
            params.append(('const std::optional<QuantInfo> &', 'output_quant_info'))
        params_str = ', '.join(map(lambda param: "{} {}".format(*param), params))
        cogoutl("ModelBuilder::Index Add{}({});".format(op['name'], params_str))
        cogoutl('#endif // __ANDROID_API__ >= {}'.format(op['api']))

    update_code('dnnlibrary/include/ModelBuilder.h', 'ModelBuilder auto generated methods')
    """

    global str_io
    str_io = None
    with open('ops.yml') as f:
        cfg = yaml.load(f)

    infer_cfg(cfg, Target.OnnxConverter)
    for i, op in enumerate(cfg):
        ipt_opt = op['input'] + op['output']
        params = list(map(get_param, ipt_opt))
        params_str = ', '.join(map(lambda param: "{} {}".format(*param), params))
        cogoutl(f'void OnnxConverter::AddLayer{op["name"]}({params_str}) {{')
        cogoutl(
            f"shaper_.{op['shaper']}({', '.join([x['name'] for x in ipt_opt if x.get('needed_by_shaper', False)])});")

        if op['fused']:
            cogoutl(f"const auto activation = FindActivation(model_proto_, output);")
            cogoutl("if (activation.first.has_value()) {")
            cogoutl("skipped_act_.push_back(activation.first.value());")
            cogoutl("name_map_[activation.first.value()] = output;")
            cogoutl(")}")
        cogout(f"const auto param = DNN::Create{op['name']}Direct(builder_, ")

        def get_input_param(x):
            if x['cpp_type'] == 'str':
                return f"m({x['name']}).c_str()"
            elif x['cpp_type'] == 'optional_str':
                return f"bias_name.has_value() ? bias_name.value().c_str() : nullptr"
            else:
                return x['name']

        cogout(', '.join(list(map(get_input_param, op['input']))))
        if op['fused']:
            cogout(', ConvertFuseCodeType(activation.second)')
        cogout(', ')
        cogout(', '.join(list(map(lambda x: f"{x['name']}.c_str()", op['output']))))
        cogoutl(');')
        cogout(f"const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::{op['name']}, ")
        cogout(', '.join(['0'] * (op['pos'])))
        cogoutl(', param);')
        cogoutl('layers_.push_back(layer);')
        cogoutl('}')


if __name__ == '__main__':
    main()
