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
    else:
        return elem['cpp_type'], elem['name']


def add_tensor_operand(operand):
    if operand['cpp_type'] == 'str':
        return 'const auto {}_idx = operand_indexes_[{}]'.format(operand['name'], operand['name'])
    elif operand['cpp_type'] == 'float':
        return 'const auto {}_idx = FillOperand("input_{}_of_" + output_name, {{Type::TENSOR_FLOAT32, {{1}}}}, {}); '.format(
            operand['name'], operand['name'], operand['name'])
    else:
        raise Exception()


with open('config.yml') as f:
    cfg = yaml.load(f)


def infer_cfg(op):
    if 'input' not in op:
        op['input'] = []
    if 'base_input_num' not in op or op['base_input_num'] == 1:
        op['input'].insert(0, {'name': 'input', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'need_by_shaper': True})
    elif op['base_input_num'] == 2:
        op['input'] = [{'name': 'input1', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'need_by_shaper': True},
                       {'name': 'input2', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'need_by_shaper': True}] \
                      + op['input']
    elif op['base_input_num'] == 'n':
        op['input'].insert(0, {'name': 'inputs', 'nnapi_type': 'tensor', 'cpp_type': 'str_list', 'need_by_shaper': True})
    elif op['base_input_num'] == 0:
        pass
    else:
        raise Exception()

    if not 'output' in op:
        op['output'] = [{'name': 'output', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'need_by_shaper': True}]
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


for i, op in enumerate(cfg):
    infer_cfg(op)
    if len(op['input']) == 0:
        continue
    ipt_opt = op['input'] + op['output']
    params = list(map(get_param, ipt_opt))
    if op['fused']:
        params.append(('const std::optional<QuantInfo> &', 'output_quant_info'))
    params_str = ', '.join(map(lambda param: "{} {}".format(*param), params))
    cogoutl("ModelBuilder::Index ModelBuilder::Add{}({}) {{".format(op['name'], params_str))
    tensor_input = list(filter(lambda x: x['nnapi_type'] == 'tensor', op['input']))
    scalar_input = list(filter(lambda x: x['nnapi_type'] == 'scalar', op['input']))

    for x in tensor_input:
        cogoutl(add_tensor_operand(x))
    cogoutl('IndexSeq input_indexes{{{}}};'.format(', '.join([x['name'] + "_idx" for x in tensor_input])))
    if len(scalar_input) > 0:
        cogoutl('AddScalarOperands(input_indexes, {});'.format(', '.join([x['name'] for x in scalar_input])))
    cogoutl('shaper_.{}({});'.format(op['shaper'],
                                     ', '.join([x['name'] for x in ipt_opt if x.get('need_by_shaper', False)])))
    op_type_params = ['operand_types.at({}).type'.format(op['input'][0]['name']),
                      'shaper_[{}]'.format(op['output'][0]['name'])]
    if op['fused']:
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
