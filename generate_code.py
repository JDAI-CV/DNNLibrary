import io
import subprocess
import yaml
from enum import Enum
from typing import Dict

str_io = io.StringIO()


class Target(Enum):
    ModelBuilder = 1
    OnnxConverter = 2
    DaqReader = 3


def clang_format(filename: str):
    subprocess.run(['clang-format', '-i', filename])


def compile_fbs():
    subprocess.run(['flatc', '--cpp', '--scoped-enums', '-o', 'include/common/', 'common/daq.fbs'])


def cogout(txt):
    print(txt, end='', file=str_io)


def cogoutl(txt):
    print(txt, file=str_io)


def param_to_string_in_declaration(param: Dict[str, str]) -> str:
    ret = param['type'] + ' ' + param['name']
    if 'default' in param:
        ret += '=' + param['default']
    return ret


def param_to_string_in_definition(param: Dict[str, str]) -> str:
    ret = param['type'] + ' ' + param['name']
    return ret


def get_param(elem: dict) -> Dict[str, str]:
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
        ret = {'type': 'const std::string &', 'name': elem['name']}
    elif elem['cpp_type'] == 'optional_str':
        ret = {'type': 'const dnn::optional<std::string> &', 'name': elem['name']}
    elif elem['cpp_type'] == 'str_list':
        ret = {'type': 'const std::vector<std::string> &', 'name': elem['name']}
    elif elem['cpp_type'] == 'int32_list':
        ret = {'type': 'const std::vector<int32_t> &', 'name': elem['name']}
    else:
        ret = {'type': elem['cpp_type'], 'name': elem['name']}
    # If we make some input (e.g. api 29 new input) optional, the outputs (after inputs in arg list)
    # have to be optional too, so disable it.
    # if 'default' in elem:
        # ret['default'] = elem['default']
    return ret


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
                return make_unexpected("Unknown type " + typeToStr(weight_type));
            }
        } else {
            bias_idx_val = operand_indexes_.at(bias.value());
        }
        input_indexes.push_back(bias_idx_val);'''


def add_tensor_operand(operand):
    if operand['predefined'] == 'optional_bias':
        return add_optional_bias()
    if operand['cpp_type'] == 'str':
        return '''imm_blob_inputs_.insert({0});
const auto {0}_idx = operand_indexes_.at({0});
input_indexes.push_back({0}_idx);'''.format(operand['name'])
    elif operand['cpp_type'] == 'float':
        return '''const auto {0}_idx = FillOperand("input_{0}_of_" + output, {{Type::TENSOR_FLOAT32, {{1}}}}, {0}); 
input_indexes.push_back({0}_idx);'''.format(operand['name'])
    elif operand['cpp_type'] == 'int32_list':
        return '''const auto {0}_idx = AddTensorFromBuffer("input_{0}_of_" + output, &{0}[0], {{Type::TENSOR_INT32, Shape{{static_cast<uint32_t>({0}.size())}}}}); 
input_indexes.push_back({0}_idx);'''.format(operand['name'])
    elif operand['cpp_type'] == 'str_list':
        return '''for (const auto &x : {}) {{
imm_blob_inputs_.insert(x);
input_indexes.push_back(operand_indexes_.at(x));
}}'''.format(operand['name'])
    else:
        raise Exception('Unknown cpp_type {}'.format(operand['cpp_type']))


def has_fuse_code_attr(op: dict):
    return any([x['predefined'] == 'fuse_code' for x in op['input']])


def infer_cfg(cfg, target: Target):
    next_pos = 0
    for i, op in enumerate(cfg):
        if 'input' not in op:
            op['input'] = []
        if 'base_input_num' not in op or op['base_input_num'] == 1:
            op['input'].insert(0,
                               {'name': 'input', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'is_onnx_attr': False,
                                'needed_by_shaper': True})
        elif op['base_input_num'] == 2:
            op['input'] = [{'name': 'input1', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'is_onnx_attr': False,
                            'needed_by_shaper': True},
                           {'name': 'input2', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'is_onnx_attr': False,
                            'needed_by_shaper': True}] \
                          + op['input']
        elif op['base_input_num'] == 'n':
            op['input'].insert(0,
                               {'name': 'inputs', 'nnapi_type': 'tensor', 'cpp_type': 'str_list', 'is_onnx_attr': False,
                                'needed_by_shaper': True})
        elif op['base_input_num'] == 0:
            pass
        else:
            raise Exception()

        if 'pos' not in op:
            op['pos'] = next_pos
        next_pos = op['pos'] + 1

        if 'output' not in op:
            op['output'] = [{'name': 'output', 'nnapi_type': 'tensor', 'cpp_type': 'str', 'needed_by_shaper': True}]
        assert 'shaper' in op
        assert 'dnn' not in op
        assert 'name' not in op
        # if 'shaper' not in op:
        #     op['shaper'] = op['nnapi']
        # if 'nnapi' not in op:
        #     op['nnapi'] = op['name'].upper()
        # if 'dnn' not in op:
        #     op['dnn'] = op['name']
        if target == Target.ModelBuilder and 'nnapi_input' in op:
            op['input'].extend(op['nnapi_input'])
        elif target == Target.OnnxConverter and 'dnn_input' in op:
            op['input'].extend(op['dnn_input'])
        if 'support_quant_asymm' not in op:
            op['support_quant_asymm'] = False
        if 'converter_simple' not in op:
            op['converter_simple'] = True
        if 'builder_simple' not in op:
            op['builder_simple'] = True
        if 'output_tensor_type' not in op:
            op['output_tensor_type'] = 'auto'
        for ipt in op['input']:
            if 'predefined' not in ipt:
                ipt['predefined'] = ''
            if ipt['predefined'] == 'optional_bias':
                ipt['name'] = 'bias'
                ipt['nnapi_type'] = 'tensor'
                ipt['cpp_type'] = 'optional_str'
                ipt['is_onnx_attr'] = False
                ipt['convert_func'] = 'OnnxToNnapiIdentity'
            elif ipt['predefined'] == 'fuse_code':
                ipt['name'] = 'fuse_code'
                ipt['nnapi_type'] = 'scalar'
                ipt['cpp_type'] = 'FuseCode'
            if 'is_onnx_attr' not in ipt:
                ipt['is_onnx_attr'] = True
            if 'convert_func' not in ipt:
                ipt['convert_func'] = 'OnnxToNnapiAxes0231'
            if 'needed_by_shaper' not in ipt:
                ipt['needed_by_shaper'] = False


def update_code(file: str, label: str, reformat: bool=True) -> None:
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
        assert start in s and end in s
    with open(file, 'w') as f:
        new_s = s[:idx1] + str_io.getvalue() + s[idx2:]
        f.write(new_s)
    str_io = io.StringIO()
    if reformat:
        clang_format(file)


def generate_onnx_converter():
    with open('ops.yml') as f:
        cfg = yaml.load(f)
    infer_cfg(cfg, Target.OnnxConverter)
    for i, op in enumerate(cfg):
        ipt_opt = op['input'] + op['output']
        params = list(map(get_param, ipt_opt))
        params_str = ', '.join(map(param_to_string_in_definition, params))
        cogoutl(f"void OnnxConverter::WriteDaqLayer_{op['nnapi']}{'' if op['converter_simple'] else 'Impl'}({params_str}) {{")
        # if has_fuse_code_attr(op):
            # cogoutl(f"const auto activation = FindActivation(model_proto_, output);")
        for x in op['input']:
            if not x['is_onnx_attr']:
                if x['cpp_type'] == 'str':
                    cogoutl(f"""
                    {{
                        const auto name = {x['name']};""")
                elif x['cpp_type'] == 'optional_str':
                    cogoutl(f"""
                    if ({x['name']}.has_value()) {{
                        const auto name = {x['name']}.value();""")
                elif x['cpp_type'] == 'str_list':
                    cogoutl(f"""
                    for (const auto &name : {x['name']}) {{""")
                cogoutl(f"""
                        if (onnx_tensors_.has(name)) {{
                            const auto &onnx_tensor = onnx_tensors_.at(name);
                            const auto new_tensor = {x['convert_func']}(onnx_tensor);
                            shaper_.AddShape(name, new_tensor.shape); 
                            nnapi_tensors_[name] = new_tensor;
                            CreateTensorFb(name, new_tensor);
                        }}
                    }}
                """)
            if x['cpp_type'] == 'str_list':
                cogoutl(f"const auto {x['name']}_fb = FbStrVector({x['name']});")

        shaper_params = []
        for x in op['input']:
            if x['needed_by_shaper']:
                if x['cpp_type'] == 'str':
                    shaper_params.append(f"m({x['name']})")
                else:
                    shaper_params.append(f"{x['name']}")
        shaper_params += [x['name'] for x in op['output']]
        cogoutl(
            f"shaper_.{op['shaper']}({', '.join(shaper_params)});")

        def get_input_param(x):
            if x['cpp_type'] == 'str':
                return f"m({x['name']}).c_str()"
            elif x['cpp_type'] == 'optional_str':
                return f"{x['name']}.has_value() ? {x['name']}.value().c_str() : nullptr"
            elif x['cpp_type'] == 'str_list':
                return f"&{x['name']}_fb"
            elif x['cpp_type'] == 'int32_list':
                return f"&{x['name']}"
            elif x['predefined'] == 'fuse_code':
                return f"ConvertFuseCodeType({x['name']})"
            else:
                return x['name']

        cogout(f"const auto input_param = DNN::Create{op['nnapi']}_InputDirect(builder_, ")
        cogout(', '.join(list(map(get_input_param, op['input']))))
        cogoutl(');')
        # cogout(', ')
        cogout(f"const auto output_param = DNN::Create{op['nnapi']}_OutputDirect(builder_, ")
        cogout(', '.join(list(map(lambda x: f"{x['name']}.c_str()", op['output']))))
        cogoutl(');')
        cogout(f"const auto param = DNN::Create{op['nnapi']}(builder_, input_param, output_param);")
        cogout(f"const auto layer = DNN::CreateLayer(builder_, DNN::LayerType::{op['nnapi']}, ")
        cogout(''.join(['0, '] * (op['pos'])))
        cogoutl('param);')
        cogoutl('layers_.push_back(layer);')
        cogoutl('}')
        cogoutl('')
    update_code('tools/onnx2daq/OnnxConverterImpl.cpp', 'OnnxConverter auto generated methods')
    for i, op in enumerate(cfg):
        ipt_opt = op['input'] + op['output']
        params = list(map(get_param, ipt_opt))
        params_str = ', '.join(map(param_to_string_in_declaration, params))
        cogoutl(f"void WriteDaqLayer_{op['nnapi']}{'' if op['converter_simple'] else 'Impl'}({params_str});")
    update_code('include/tools/onnx2daq/OnnxConverter.h', 'OnnxConverter auto generated methods')


def generate_daq_reader():
    with open('ops.yml') as f:
        cfg = yaml.load(f)
    infer_cfg(cfg, Target.DaqReader)
    for i, op in enumerate(cfg):
        cogoutl(f'case DNN::LayerType::{op["nnapi"]}:')
        cogoutl(f'return "{op["nnapi"]}";')
    update_code('dnnlibrary/DaqReader.cpp', 'DaqReader auto generated layer_type_to_str')
    for i, op in enumerate(cfg):
        cogoutl(f"case DNN::LayerType::{op['nnapi']}: {{")

        arg_names = [x['name'] for x in op['input']]
        cogoutl(f"UNPACK_LAYER_QUANT({op['nnapi']}, {', '.join(arg_names)});")
        arg_names += [x['name'] for x in op['output']]
        for i, x in enumerate(op['input']):
            if x['cpp_type'] == 'optional_str':
                cogoutl(f"const dnn::optional<std::string> {x['name']}_right_type "
                        f"= ({x['name']} == \"\") ? dnn::nullopt : dnn::make_optional({x['name']});")
                arg_names[i] = f"{x['name']}_right_type"
        if op['support_quant_asymm']:
            arg_names += ['quant_info']
        cogoutl(f"""
                TRY(builder.AddLayer_{op['nnapi']}({', '.join(arg_names)}));
                break;
            }}""")
    update_code('dnnlibrary/DaqReader.cpp', 'auto generated layer reader')


def generate_fbs():
    with open('ops.yml') as f:
        cfg = yaml.load(f)
    # The target of fbs is the same as onnx converter
    infer_cfg(cfg, Target.OnnxConverter)

    d = {
        'int32_list': '[int]',
        'int32_t': 'int',
        'str': 'string',
        'optional_str': 'string',
        'str_list': '[string]',
        'float': 'float',
        'FuseCode': 'FuseCode',
        'bool': 'bool',
    }
    for i, op in enumerate(cfg):
        cogoutl(f"table {op['nnapi']}_Input {{")
        for x in op['input']:
            cogoutl(f"    {x['name']}: {d[x['cpp_type']]};")
        cogoutl('}')
        cogoutl('')

        cogoutl(f"table {op['nnapi']}_Output {{")
        for x in op['output']:
            cogoutl(f"    {x['name']}: {d[x['cpp_type']]};")
        cogoutl('}')
        cogoutl('')

        cogoutl(f"table {op['nnapi']} {{")
        cogoutl(f"    input: {op['nnapi']}_Input;")
        cogoutl(f"    output: {op['nnapi']}_Output;")
        cogoutl('}')
        cogoutl('')
    update_code('common/daq.fbs', 'Auto generated tables', reformat=False)

    for i, op in enumerate(cfg):
        cogoutl(f"    {op['nnapi']}_param:{op['nnapi']};")
    update_code('common/daq.fbs', 'Auto generated fields', reformat=False)
    for i, op in enumerate(cfg):
        cogoutl(f"    {op['nnapi']},")
    update_code('common/daq.fbs', 'Auto generated layer types', reformat=False)
    compile_fbs()


def generate_model_builder():
    with open('ops.yml') as f:
        cfg = yaml.load(f)
    infer_cfg(cfg, Target.ModelBuilder)
    for i, op in enumerate(cfg):
        if len(op['input']) == 0:
            continue
        ipt_opt = op['input'] + op['output']
        params = list(map(get_param, ipt_opt))
        if op['support_quant_asymm']:
            params.append({'type': 'const dnn::optional<QuantInfo> &', 'name': 'output_quant_info'})
        params_str = ', '.join(map(param_to_string_in_definition, params))
        cogoutl("expected<Unit, std::string> ModelBuilder::AddLayer_{}{}({}) {{".format(
            op['nnapi'], '' if op['builder_simple'] else '_Impl', params_str))
        cogoutl(f'if (nnapi_->android_sdk_version < {op["api"]}) {{'
                f'return make_unexpected("{op["nnapi"]} requires API {op["api"]}");'
                f'}}')
        for ipt in op['input']:
            if 'default' in ipt and 'api' in ipt:
                cogoutl(f'''
                    if ({ipt['name']} != {ipt['default']} && nnapi_->android_sdk_version < {ipt['api']}) {{
                        return make_unexpected("Input \\"{ipt['name']}\\" of {op["nnapi"]} requires API {ipt["api"]}");
                    }}
                ''')
        tensor_input = list(filter(lambda x: x['nnapi_type'] == 'tensor', op['input']))
        scalar_input = list(filter(lambda x: x['nnapi_type'] == 'scalar', op['input']))

        cogoutl('IndexSeq input_indexes;')
        for x in tensor_input:
            cogoutl(add_tensor_operand(x))
        # cogoutl('IndexSeq input_indexes{{{}}};'.format(', '.join([x['name'] + "_idx" for x in tensor_input])))
        for x in scalar_input:
            if 'api' in x:
                cogoutl(f"""
                    if (android_api_level() > {x['api']}) {{
                        AddScalarOperands(input_indexes, {x['name']});
                    }}""")
            else:
                cogoutl(f"AddScalarOperands(input_indexes, {x['name']});")
        cogoutl('shaper_.{}({});'.format(op['shaper'],
                                         ', '.join([x['name'] for x in ipt_opt if x['needed_by_shaper']])))
        if op['output_tensor_type'] != 'auto':
            op_type_params = ['Type::{}'.format(op['output_tensor_type']),
                              'shaper_[{}]'.format(op['output'][0]['name'])]
        elif op['input'][0]['cpp_type'] == 'str_list':
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
    imm_blob_outputs_.insert(output);
    return Unit();
    }
    
    '''
        )
    update_code('dnnlibrary/ModelBuilderImpl.cpp', 'ModelBuilder auto generated methods')
    for i, op in enumerate(cfg):
        if len(op['input']) == 0:
            continue
        ipt_opt = op['input'] + op['output']
        params = list(map(get_param, ipt_opt))
        if op['support_quant_asymm']:
            params.append({'type': 'const dnn::optional<QuantInfo> &', 'name': 'output_quant_info', 'default': 'dnn::nullopt'})
        params_str = ', '.join(map(param_to_string_in_declaration, params))
        cogoutl("expected<Unit, std::string> AddLayer_{}({});".format(
            op['nnapi'], params_str))

        # if op['builder_simple'] is not True, we generate both AddLayer_* and AddLayer_*_Impl declaration
        if not op['builder_simple']:
            cogoutl('private:')
            cogoutl("expected<Unit, std::string> AddLayer_{}_Impl({});".format(
                op['nnapi'], params_str))
            cogoutl('public:')
    update_code('include/dnnlibrary/ModelBuilder.h', 'ModelBuilder auto generated methods')


def main():
    generate_fbs()
    generate_model_builder()
    generate_onnx_converter()
    generate_daq_reader()


if __name__ == '__main__':
    main()
