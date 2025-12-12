## NoCodeNN（基于图结构的 PyTorch 代码生成）

本项目提供一个 **Flask API**：前端提交“网络拓扑（node/edge）+ 训练配置（loss/optimizer/hyperParameters）”，服务端会生成可运行的 **PyTorch** 代码文件：

- `src/output/net.py`：网络结构（`class Net`）
- `src/output/main.py`：MNIST 训练脚本（会 `import net` 并训练生成的 `Net`）

同时接口会直接返回两份源码字符串，并提供下载接口。

---

## 目录结构

- `src/app.py`：Flask 服务入口（`/submit` 生成代码，`/download_*` 下载生成文件）
- `src/service/NetGenerator.py`：根据 `node/edge` 生成 `net.py`
- `src/service/MainGenerator.py`：根据 `loss/optimizer/hyperParameters` 生成 `main.py`
- `src/entity/cnn.json`：一个示例网络（仅包含 `node/edge`）
- `src/test/CreateCode1.py`：本地脚本示例（从 `entity/cnn.json` 生成 `net.py`）
- `src/test/main.py`：项目内置的 MNIST 训练示例（使用 `src/output/cnn.py` 的 `GraphNet`）

> 说明：`src/api/*` 与 `src/controller/*` 当前基本为空壳/占位，不影响主流程。

---

## 环境依赖

- Python 3.x
- Flask / flask-cors
- torch / torchvision

### 安装依赖

**Windows（cmd/PowerShell）**：

```bat
python -m pip install -U pip
python -m pip install flask flask-cors torch torchvision
```

**Linux/macOS（可选）**：

```bash
python3 -m pip install -U pip
python3 -m pip install flask flask-cors torch torchvision
```

---

## 快速开始（启动 API 并生成代码）

### 1) 启动服务

**Windows（cmd/PowerShell）**：

```bat
python src\app.py
```

服务默认监听：`http://127.0.0.1:8081`

### 2) 调用 `/submit` 生成 `net.py` + `main.py`

请求体是 JSON，至少包含：`node`、`edge`、`loss`、`optimizer`、`hyperParameters`。

示例 `payload.json`（可直接用于调用）：

```json
{
  "node": [
    {"id": 1, "type": "layer", "name": "conv2d", "attr": {"in_channels": 1, "out_channels": 32, "kernel_size": 3}},
    {"id": 2, "type": "activation", "name": "relu"},
    {"id": 3, "type": "option", "name": "op_view", "attr": {"h": 64, "w": -1}},
    {"id": 4, "type": "layer", "name": "linear", "attr": {"in_features": 7744, "out_features": 10}}
  ],
  "edge": [
    {"from": 1, "to": 2},
    {"from": 2, "to": 3},
    {"from": 3, "to": 4}
  ],
  "loss": "NLLLoss",
  "optimizer": "Adadelta",
  "hyperParameters": {
    "batchSize": 64,
    "epochs": 1,
    "lr": 1.0,
    "gamma": 0.7,
    "seed": 1,
    "NoCUDA": true
  }
}
```

**Windows（cmd/PowerShell）**（使用 curl）：

```bat
curl -X POST "http://127.0.0.1:8081/submit" -H "Content-Type: application/json" -d @payload.json
```

返回：

- `net`：生成的 `net.py` 源码字符串
- `main`：生成的 `main.py` 源码字符串

并且会在磁盘落地：

- `src/output/net.py`
- `src/output/main.py`

### 3) 下载生成文件

- 下载网络文件：`GET /download_net`
- 下载训练脚本：`GET /download_main`

---

## 本地离线生成（仅生成 net.py）

仓库内置了一个脚本会读取 `src/entity/cnn.json`，并生成 `src/output/net.py`。

**Windows（cmd/PowerShell）**：

```bat
python src\test\CreateCode1.py
```

> 注意：`entity/cnn.json` 只包含 `node/edge`，不包含 `loss/optimizer/hyperParameters`，因此该脚本只生成网络文件；训练脚本建议通过 API 生成。

---

## 输入 JSON 规范（/submit）

### 顶层字段

- **node**：节点列表（也兼容 `{"1": {...}, "2": {...}}` 这种字典形式）
- **edge**：边列表（`from/to` 兼容 int 或数字字符串）
- **loss**：损失函数名称（见下方可选值）
- **optimizer**：优化器名称（torch.optim 里的类名，例如 `Adam` / `SGD` / `Adadelta` 等）
- **hyperParameters**：训练超参对象

### node 节点格式

每个节点：

- `id`：节点 id（建议用 int）
- `type`：`layer` / `activation` / `option`
- `name`：具体类型
- `attr`：可选参数字典（不同节点不同）

当前支持：

- `type=layer`：
  - `name=linear` → `nn.Linear(**attr)`
  - `name=conv2d` → `nn.Conv2d(**attr)`
  - `name=dropout` → `nn.Dropout(**attr)`
  - `name=maxpool2d` → `nn.MaxPool2d(**attr)`
- `type=activation`：
  - `name=relu`
  - `name=sigmoid`
  - `name=tanh`（也兼容旧拼写 `tahn`）
- `type=option`：
  - `name=op_view`：会生成 `x = x.view(h, w)`，需要 `attr: {"h": ..., "w": ...}`

### edge 边格式

- `from`：上游节点 id
- `to`：下游节点 id

> 当前生成逻辑本质是“拓扑排序后顺序执行”，更适合**单输入单输出的串行网络**；分支/残差/多输入等需要扩展生成逻辑。

### loss 可选值

`src/service/MainGenerator.py` 内置支持：

- `L1Loss`
- `CrossEntropy`
- `SmoothL1Loss`
- `MSELoss`
- `BCELoss`
- `BCEWithLogitsLoss`
- `NLLLoss`
- `KLDivLoss`
- `MarginRankingLoss`
- `MultiMarginLoss`
- `MultiLabelMarginLoss`
- `SoftMarginLoss`
- `MultiLabelSoftMarginLoss`
- `CosineEmbeddingLoss`

### hyperParameters 格式

生成的 `main.py` 读取字段：

- `batchSize`
- `epochs`
- `lr`
- `gamma`
- `seed`
- `NoCUDA`（true 表示禁用 CUDA）

---

## 运行生成的训练脚本

生成完成后，你可以直接运行：

**Windows（cmd/PowerShell）**：

```bat
python src\output\main.py
```

训练脚本会下载 MNIST 到 `./data`（相对于“运行命令时的当前目录”）。

---

## 常见问题

- **Q：为什么生成的文件不在 output 目录？**
  - A：已修复为跨平台路径拼接，生成文件固定输出到 `src/output/`。

- **Q：`entity/cnn.json` 和 `/submit` 的 node 格式不一样？**
  - A：现在生成器已兼容 `node` 为“列表”或“字典”；但 `/submit` 推荐使用列表格式，便于前端直接序列化。

- **Q：能支持残差/跳连/多分支吗？**
  - A：目前 `forward` 逻辑只有一个 `x` 变量顺序流动，属于串行网络；要支持分支需要在生成器中引入多变量/张量缓存与 merge 规则。
