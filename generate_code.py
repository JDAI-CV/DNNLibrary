import yaml
from typing import Tuple


def cogout(txt):
    print(txt, end='')


def cogoutl(txt):
    print(txt)


def get_param(elem: dict) -> Tuple[str, str]:
    """
    get parameter in function signature from yaml element, e.g.
    -
        name: input
        type: str
    produces ["const string &"] and ["input_name"]
    :param elem: yaml element of a input
    :return: A tuple, (type, name)
    """
    if elem['cpp_type'] == 'str':
        return 'const string &', elem['name']
    elif elem['cpp_type'] == 'optional_str':
        return 'const std::optional<string> &', elem['name']
    else:
        return elem['cpp_type'], elem['name']


def add_optional_bias():
    return '''uint32_t bias_idx_val;
        css bias_val = bias.value_or(weight_name + "_b");
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
            bias_idx_val = operand_indexes_[bias.value()];
        }'''


def add_tensor_operand(operand):
    if operand['predefined'] == 'optional_bias':
        return add_optional_bias()
    if operand['cpp_type'] == 'str':
            return '''const auto {0}_idx = operand_indexes_[{0}];
input_indexes.push_back({0}_idx);'''.format(operand['name'])
    elif operand['cpp_type'] == 'float':
        return '''const auto {0}_idx = FillOperand("input_{0}_of_" + output_name, {{Type::TENSOR_FLOAT32, {{1}}}}, {0}); 
input_indexes.push_back({0}_idx);'''.format(operand['name'])
    elif operand['cpp_type'] == 'str_list':
        return '''for (const auto &x : {}) {{
input_indexes.push_back(operand_indexes_[x]);
}}'''.format(operand['name'])
    else:
        raise Exception('Unknown cpp_type {}'.format(operand['cpp_type']))


with open('ops.yml') as f:
    cfg = yaml.load(f)


def infer_cfg(op):
    if 'input' not in op:
        op['input'] = []
    if 'base_input_num' not in op or op['base_input_num'] == 1:
        op['input'].insert(0, {'name': 'input', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'needed_by_shaper': True})
    elif op['base_input_num'] == 2:
        op['input'] = [{'name': 'input1', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'needed_by_shaper': True},
                       {'name': 'input2', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'needed_by_shaper': True}] \
                      + op['input']
    elif op['base_input_num'] == 'n':
        op['input'].insert(0, {'name': 'inputs', 'nnapi_type': 'tensor', 'cpp_type': 'str_list', 'needed_by_shaper': True})
    elif op['base_input_num'] == 0:
        pass
    else:
        raise Exception()

    if not 'output' in op:
        op['output'] = [{'name': 'output', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'needed_by_shaper': True}]
    if not 'shaper' in op:
        op['shaper'] = op['name']
    if not 'nnapi' in op:
        op['nnapi'] = op['name'].upper()
    if 'fused' not in op:
        op['fused'] = False
    if op['fused']:
        op['input'].append({'name': 'fuse_code', 'nnapi_type': 'scalar', 'cpp_type': 'int'})
    if 'support_quant8_asymm' not in op:
        op['support_quant8_asymm'] = False
    for ipt in op['input']:
        if 'predefined' not in ipt:
            ipt['predefined'] = ''
        if ipt['predefined'] == 'optional_bias':
            ipt['name'] = 'bias'
            ipt['nnapi_type'] = 'tensor'
            ipt['cpp_type'] = 'optional_str'


for i, op in enumerate(cfg):
    infer_cfg(op)
    if len(op['input']) == 0:
        continue
    ipt_opt = op['input'] + op['output']
    params = list(map(get_param, ipt_opt))
    if op['support_quant8_asymm']:
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
    op_type_params = ['operand_types.at({}).type'.format(op['input'][0]['name']),
                      'shaper_[{}]'.format(op['output'][0]['name'])]
    if op['support_quant8_asymm']:
        op_type_params.append('output_quant_info')
    cogoutl('const OperandType operand_type({});'.format(', '.join(op_type_params)))
    cogoutl('const auto output_idx = '
            'AddOperation(ANEURALNETWORKS_{}, input_indexes, operand_type)[0];'.format(op['nnapi']))
    cogout(
        '''RegisterOperand(output_name, output_index, operand_type);
return output_index;
}

'''
    )
