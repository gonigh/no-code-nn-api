# 可执行python代码构造器
import os.path

import torch
import torch.nn as nn


# 定义神经网络模型
class GraphNet(nn.Module):
    def __init__(self, graph):
        super(GraphNet, self).__init__()

        # 创建节点对象
        self.nodes = nn.ModuleDict()
        for node_id in graph['nodes']:
            node = graph['nodes'][node_id]
            if node['type'] == 'input':
                self.nodes[str(node_id)] = nn.Identity()  # 输入节点
            elif node['type'] == 'output':
                self.nodes[str(node_id)] = nn.Identity()  # 输出节点
            elif node['type'] == 'linear':
                self.nodes[str(node_id)] = nn.Linear(node['input_size'], node['output_size'])
            elif node['type'] == 'relu':
                self.nodes[str(node_id)] = nn.ReLU()
            else:
                raise ValueError("Unknown node type: {}".format(node['type']))

        # 创建边对象
        for edge in graph['edges']:
            src_node_id, dst_node_id = edge
            src_node = self.nodes[str(src_node_id)]
            dst_node = self.nodes[str(dst_node_id)]
            self.add_module("edge_{}_{}".format(src_node_id, dst_node_id), nn.Sequential(src_node, dst_node))

    def forward(self, x):
        # 执行每个节点操作
        for node_id in self.nodes:
            x = self.nodes[node_id](x)
        return x


def demo():

    # 创建一个图数据结构
    graph = {
        'nodes': {
            'n0': {'type': 'input', 'input_size': 2},
            'n1': {'type': 'linear', 'input_size': 2, 'output_size': 10},
            'n2': {'type': 'relu'},
            'n3': {'type': 'linear', 'input_size': 10, 'output_size': 1},
            'n4': {'type': 'output'}
        },
        'edges': [(0, 1), (1, 2), (2, 3), (3, 4)]
    }

    file_path = os.path.join('/output/', 'model.py')

    # 生成可执行的Python文件
    with open(file_path, 'w') as f:
        f.write('import torch\n')
        f.write('import torch.nn as nn\n\n')
        f.write('class GraphNet(nn.Module):\n')
        f.write('    def __init__(self):\n')
        f.write('        super(GraphNet, self).__init__()\n\n')
        for node_id in graph['nodes']:
            node = graph['nodes'][node_id]
            if node['type'] == 'input':
                f.write('        self.{} = nn.Identity()\n'.format(node_id))
            elif node['type'] == 'output':
                f.write('        self.{} = nn.Identity()\n'.format(node_id))
            elif node['type'] == 'linear':
                f.write('        self.{} = nn.Linear({}, {})\n'.format(node_id, node['input_size'], node['output_size']))
            elif node['type'] == 'relu':
                f.write('        self.{} = nn.ReLU()\n'.format(node_id))
            else:
                raise ValueError("Unknown node type: {}".format(node['type']))
        f.write('\n')

        # 添加forward方法
        f.write('    def forward(self, x):\n')
        for edge in graph['edges']:
            src_node_id, dst_node_id = edge
            f.write('        x = self.{}(x)\n'.format("edge_{}_{}".format(src_node_id, dst_node_id)))
        f.write('        return x\n\n')

    s = ''
    # 打印生成的Python文件内容
    with open(file_path, 'r') as f:
        s = f.read()
    print(s)
    return s
