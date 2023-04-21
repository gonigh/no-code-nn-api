# 可执行python代码构造器
import os.path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create(data):
    file_path = ROOT_DIR + '\\output\\cnn.py'

    # 生成可执行的Python文件
    with open(file_path, 'w') as f:
        f.write('import torch\n')
        f.write('import torch.nn as nn\n')
        f.write('import torch.nn.functional as F\n\n\n')
        f.write('class GraphNet(nn.Module):\n')
        f.write('\tdef __init__(self):\n')
        f.write('\t\tsuper(GraphNet, self).__init__()\n\n')
        for i in data.get('node'):
            f.write(add_layer(i))
        f.write('\n')

        f.write('\tdef forward(self, x):\n')
        list = topological_sort(data.get('node'), data.get('edge'))
        print(list)
        for i in list:
            f.write(add_forward(i, data.get('node')))
        f.write('\t\tx = F.log_softmax(x, dim=1)\n')
        f.write('\t\treturn x\n')
    s = ''
    # 打印生成的Python文件内容
    with open(file_path, 'r') as f:
        s = f.read()
    print(s)
    return s


def add_layer(node):
    t = node['type']
    n = node['name']
    print(n)
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
    list = []
    dict_in = {}    # 记录入度
    dict_out = {}   # 记录出度
    # 初始化
    print(node)
    for i in node:
        dict_in[i['id']] = []
        dict_out[i['id']] = []

    for e in edge:
        dict_out.get(e['from']).append(e['to'])
        dict_in.get(e['to']).append(e['from'])

    while len(list) != len(node):
        for key in dict_in.keys():
            if len(dict_in.get(key)) == 0:
                list.append(key)
                for k in dict_out.get(key):
                    dict_in.get(k).remove(key)
                dict_in.pop(key)
                break
    return list


def add_forward(node_id, node_list):
    node = {}
    for i in node_list:
        if i['id'] is node_id:
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
        elif n == 'tahn':
            s = '\t\tx = F.tahn(x)\n'
    return s
