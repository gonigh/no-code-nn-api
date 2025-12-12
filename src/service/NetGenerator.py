# 可执行python代码构造器
import os
import os.path
from typing import Any, Dict, Iterable, List, Union

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _as_id(v: Any) -> Any:
    """
    兼容前端传入的 string/int id。
    - "1" -> 1
    - 1 -> 1
    - 其它保持原样
    """
    try:
        return int(v)
    except Exception:
        return v


def _normalize_nodes(nodes: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    兼容两种输入:
    - list[dict] (前端常见)
    - dict[str, dict] (entity/cnn.json 当前格式)
    """
    if isinstance(nodes, dict):
        return list(nodes.values())
    return list(nodes)


def createNet(nodes, edges):
    output_dir = os.path.join(ROOT_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'net.py')

    nodes_list = _normalize_nodes(nodes)

    # 生成可执行的Python文件
    with open(file_path, 'w') as f:
        f.write('import torch\n')
        f.write('import torch.nn as nn\n')
        f.write('import torch.nn.functional as F\n\n\n')
        f.write('class Net(nn.Module):\n')
        f.write('\tdef __init__(self):\n')
        f.write('\t\tsuper(Net, self).__init__()\n\n')
        for i in nodes_list:
            f.write(add_layer(i))
        f.write('\n')

        f.write('\tdef forward(self, x):\n')
        order = topological_sort(nodes_list, edges)
        for node_id in order:
            f.write(add_forward(node_id, nodes_list))
        f.write('\t\tx = F.log_softmax(x, dim=1)\n')
        f.write('\t\treturn x\n')
    s = ''
    # 打印生成的Python文件内容
    with open(file_path, 'r') as f:
        s = f.read()
    return s


def add_layer(node):
    t = node['type']
    n = node['name']
    s = ''
    if t == 'layer':
        s = '\t\tself.layer_{} = '.format(node['id'])
        attr_list = []
        attr_str = ''
        if node.get('attr'):
            for a in node['attr']:
                attr_list.append('{}={}'.format(a, node['attr'][a]))
            attr_str = ', '.join(attr_list)

        if n == 'linear':
            s += 'nn.Linear({})\n'.format(attr_str)
        elif n == 'conv2d':
            s += 'nn.Conv2d({})\n'.format(attr_str)
        elif n == 'dropout':
            s += 'nn.Dropout({})\n'.format(attr_str)
        elif n == 'maxpool2d':
            s += 'nn.MaxPool2d({})\n'.format(attr_str)
    return s


def topological_sort(node, edge):
    order = []
    dict_in = {}    # 记录入度
    dict_out = {}   # 记录出度
    # 初始化
    for i in node:
        node_id = _as_id(i['id'])
        dict_in[node_id] = []
        dict_out[node_id] = []

    for e in edge:
        frm = _as_id(e.get('from'))
        to = _as_id(e.get('to'))
        if frm in dict_out:
            dict_out[frm].append(to)
        if to in dict_in:
            dict_in[to].append(frm)

    while len(order) != len(node):
        for key in list(dict_in.keys()):
            if len(dict_in.get(key)) == 0:
                order.append(key)
                for k in dict_out.get(key):
                    dict_in.get(k).remove(key)
                dict_in.pop(key)
                break
    return order


def add_forward(node_id, node_list):
    node = {}
    for i in node_list:
        if _as_id(i['id']) == node_id:
            node = i
    t = node['type']
    s = ''
    if t == 'layer':
        s = '\t\tx = self.layer_{}(x)\n'.format(node_id)
    elif t == 'option':
        n = node['name']
        if n == 'op_view':
            s = '\t\tx = x.view({}, {})\n'.format(node['attr']['h'], node['attr']['w'])
    elif t == 'activation':
        n = node['name']
        if n == 'relu':
            s = '\t\tx = F.relu(x)\n'
        elif n == 'sigmoid':
            s = '\t\tx = F.sigmoid(x)\n'
        elif n in ('tanh', 'tahn'):
            s = '\t\tx = torch.tanh(x)\n'
    return s
