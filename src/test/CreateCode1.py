import json
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from service.Generator import create


if __name__ == '__main__':

    file_path = '../entity/cnn.json'
    with open(file_path) as f:
        data = json.load(f)
        create(data)
