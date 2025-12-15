# Windows 运行指南（PowerShell）

本项目可以在 Windows 上**不依赖 `make`** 直接运行（推荐做法）。以下命令都在仓库根目录执行。

## 1) 创建并启用虚拟环境（`.venv`）

```powershell
cd <你的仓库根目录>
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

如果 PowerShell 阻止脚本执行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

如果你的环境缺少 `venv` 组件（提示 `ensurepip` 不可用），可用 `virtualenv` 兜底：

```powershell
py -3.13 -m pip install --user virtualenv
py -3.13 -m virtualenv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Smoke 模式端到端跑通（等价于 `make smoke-all`）

```powershell
python -m dialectsense.cli preprocess --config configs\smoke.json
python -m dialectsense.cli embed      --config configs\smoke.json
python -m dialectsense.cli split      --config configs\smoke.json
python -m dialectsense.cli coarsen    --config configs\smoke.json
python -m dialectsense.cli train      --config configs\smoke.json
python -m dialectsense.cli eval       --config configs\smoke.json
python -m dialectsense.cli report     --config configs\smoke.json
```

产物目录：`artifacts\smoke\`

## 3) 启动 Web UI（等价于 `make ui`）

```powershell
python -m dialectsense.cli ui --config configs\smoke.json
```

随后按终端输出的本地地址在浏览器打开。

提示：UI 会读取 `artifacts\<run_name>\` 下的训练/评估结果（图表、混淆矩阵、top confusions 等），若尚未生成，可在 UI 的 “Run” 页签直接点击运行各阶段。

## 4) Full 模式（服务器/高算力机器）

将 `--config` 换成 `configs\full.json`：

```powershell
python -m dialectsense.cli preprocess --config configs\full.json
python -m dialectsense.cli embed      --config configs\full.json
python -m dialectsense.cli split      --config configs\full.json
python -m dialectsense.cli coarsen    --config configs\full.json
python -m dialectsense.cli train      --config configs\full.json
python -m dialectsense.cli eval       --config configs\full.json
python -m dialectsense.cli report     --config configs\full.json
```

## 常见问题

### 1) `.ogg` 解码失败

如果 `soundfile` 无法解码 `.ogg`，代码会尝试回退到 `ffmpeg`。请安装 `ffmpeg` 并加入 PATH（例如使用 Chocolatey/Scoop 安装）。

### 2) 想使用 `make`

Windows 默认没有 `make`，需要自行安装（例如 `choco install make` 或 `scoop install make`）。不安装也不影响使用，上面的 Python 命令即可运行。
