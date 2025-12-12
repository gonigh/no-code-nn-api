import json
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from service.NetGenerator import createNet


if __name__ == '__main__':

    file_path = '../entity/cnn.json'
    with open(file_path) as f:
        data = json.load(f)
        # 仅生成网络结构代码（net.py）；训练脚本 main.py 由 API /submit 生成（需要 loss/optimizer/hyperParameters）。
        createNet(data.get("node"), data.get("edge"))
