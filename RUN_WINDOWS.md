# 在 Windows 上运行 DialectSense（conda）

这份项目主要用 `torch + transformers` 做音频嵌入（WavLM），并不需要 `torchvision`。你遇到的报错：

`RuntimeError: operator torchvision::nms does not exist`

通常是 **`torch` 与 `torchvision` 版本/安装来源不匹配**（conda/pip 混装最常见），导致 `transformers` 导入时连带导入 `torchvision` 失败。

## 1) 推荐做法：卸载 torchvision（本项目不需要）

在你的 conda 环境里执行其一即可：

```powershell
pip uninstall -y torchvision
```

或：

```powershell
conda remove -y torchvision
```

然后重新运行项目。

## 2) 如果你确实需要 torchvision：重装匹配的 torch/torchvision/torchaudio

关键点：
- **不要混用** conda 和 pip 来装 `torch/torchvision/torchaudio`
- 三者版本需要匹配（同一渠道、同一 CUDA/CPU 构建）

### 方案 A：全用 conda（示例）

CPU 版：

```powershell
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
```

CUDA 版（按你机器的 CUDA 版本调整 `pytorch-cuda`）：

```powershell
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 方案 B：全用 pip（示例）

CPU 版：

```powershell
pip install -U --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CUDA 版（按需改 `cu121` 等）：

```powershell
pip install -U --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 3) 安装本项目依赖

在项目根目录执行：

```powershell
pip install -r requirements.txt
```

如果你使用 `make` 不方便（Windows 常见），可以直接用 CLI：

```powershell
python -m dialectsense.cli smoke --config configs/smoke.json
python -m dialectsense.cli ui --config configs/smoke.json
```

## 4) 快速自检（定位 torch/torchvision 是否仍不匹配）

```powershell
python -c "import torch; print('torch', torch.__version__)"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
```

若第二条仍报 `torchvision::nms`，优先回到第 1/2 节处理安装问题。

