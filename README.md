# NoCodeNN 运行指南

本项目提供一个基于 Flask 的后端服务，根据前端传入的“节点/边”描述动态生成 PyTorch 神经网络代码，并可直接下载 `net.py` 与 `main.py` 用于训练 MNIST。

## 环境依赖

- Python 3.9+
- pip / venv
- PyTorch 与 torchvision（CPU 版即可）
- Flask、Flask-Cors

可以通过下述方式安装依赖：

```
python -m venv .venv
.\.venv\Scripts\activate
pip install flask flask-cors torch torchvision
```

> 如果需要 GPU/CUDA，请参考 https://pytorch.org/get-started/locally/ 选择对应的安装命令。

## 启动后端服务

1. 激活虚拟环境并进入仓库根目录。
2. 运行 Flask 应用：

```
python src\app.py
```

应用默认监听 `http://127.0.0.1:8081`。

## 主要接口

- `POST /submit`
  - 请求体字段：`node`、`edge`（图结构）、`loss`、`optimizer`、`hyperParameters`
  - 返回：生成的 `net.py` 和 `main.py` 源码字符串
- `GET /download_net`、`GET /download_main`
  - 直接下载最近一次生成的文件

## 验证生成结果

1. 通过接口生成代码后，`src/output/` 会出现新的 `net.py` 与 `main.py`。
2. 也可以参考示例：
   - `src/entity/cnn.json`：样例拓扑描述
   - `src/output/cnn.py`、`src/test/main.py`：示例网络及训练脚本
3. 切换到 `src/test/`，使用内置的 MNIST 数据集运行：

```
python src\test\main.py
```

该脚本会加载 `output.cnn.GraphNet` 并在 `./src/test/data/MNIST` 数据上训练。

## 注意事项

- `src/api/*.py`、`src/service/Actuator.py` 仍为占位文件，按需扩展。
- `src/test/CreateCode1.py` 引用了未实现的 `service.Generator`，如需使用请补全对应逻辑或移除引用。
- 默认使用 Windows 路径分隔符（`\\`）；在 Linux 部署时可根据需要调整。
