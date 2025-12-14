# DialectSense：方言识别演示 + 实验流水线

本仓库以**数据为中心**：湘音（XiangYin）数据集位于 `datasets/`，代码生成的所有派生产物写入 `artifacts/`。

上游数据集文档：`datasets/README.txt`（外部链接）。

## 数据集假设（已确认）

- 元数据：`datasets/metadata.csv`
- 音频：`datasets/oggs/<id>.ogg`（扁平目录，数字文件名）
- 本项目使用的关键元数据字段：
  - 标签：`dialect.province`
  - 说话人/分组：`uploader_id`
  - 片段 id：`id`
  - 过滤字段：`sound_length`、`votes`

## 目标

1. **离线流水线**：预处理 → 嵌入提取 →（可选）聚类 → 训练 SVM → 评估。
2. **在线演示**：Web UI 录音（麦克风）或上传音频，提取嵌入，运行训练好的分类器，并展示：
   - Top-k 省份预测 + 置信度
   - 置信度随时间变化曲线
   - 可选的 2D 嵌入投影可视化（可用时）
3. **避免泄漏**：训练/验证/测试划分必须通过 `uploader_id` 做**说话人互斥**（speaker-disjoint）。
4. **Smoke 模式**：在笔记本上快速端到端跑通，并缓存嵌入以便复跑加速。

## 文件 / 目录结构

```
configs/                  # smoke / full 的 JSON 配置
dialectsense/             # Python 包 + CLI
artifacts/<run_name>/     # 生成的产物（划分、嵌入、模型、图表等）
```

## 嵌入（默认后端）

默认后端：`logmel_stats`（CPU 友好，无需神经网络权重）。

步骤：
- 解码音频 → 单声道 → 重采样到 16 kHz
- 计算 log-mel 频谱（64 个 mel bin）
- 按时间维做均值 + 标准差聚合 → 128 维 embedding

嵌入提取器通过 registry 可替换，后续可接入其他 backbone。

## 模型

- 分类器：SVM（默认 `LinearSVC`），并使用 `class_weight="balanced"` 处理类别不均衡。
- 概率/置信度：尽可能在**验证集**上做校准（`sigmoid_on_val`）。
- 可复现性：固定随机种子；将解析后的配置保存到 artifacts。

## 过滤与加权

- 使用 `min_sound_length_sec` 过滤过短音频。
- 可选按 `min_votes` 过滤低票样本。
- 可选在训练中按 votes 做样本加权（`sample_weight_by_votes`）。
- 流水线会记录按省份统计的样本数（过滤前/后，以及 smoke 子集之后）。

说明：在 Full 模式下，如果某些省份样本数/说话人数过少，无法满足说话人互斥划分的最小样本要求，预处理步骤会自动剔除这些“过稀有”类别，并在 `artifacts/<run_name>/stats.json` 中记录被剔除的类别列表。

## 聚类 / 探索分析

聚类仅用于探索/异常点检测，不作为监督学习的替代。

- 2D 投影：优先 UMAP（可用时），否则 t-SNE。
- 保存按省份着色的嵌入投影图。

## 命令

本项目使用 `.venv` 管理依赖。

### Smoke 模式（本地）

```bash
make smoke-all
make ui
```

Makefile 会优先尝试 `python3 -m venv .venv`，若不可用则回退到 `python3 -m virtualenv .venv`。

### Full 模式（服务器）

```bash
make full-all CONFIG=configs/full.json
```

也可以单独执行各步骤：

```bash
.venv/bin/python -m dialectsense.cli preprocess --config configs/smoke.json
.venv/bin/python -m dialectsense.cli embed --config configs/smoke.json
.venv/bin/python -m dialectsense.cli train --config configs/smoke.json
.venv/bin/python -m dialectsense.cli eval --config configs/smoke.json
.venv/bin/python -m dialectsense.cli cluster --config configs/smoke.json
.venv/bin/python -m dialectsense.cli ui --config configs/smoke.json
```

## Smoke 模式验收标准

执行 `make smoke-all` 后，应生成：

- `artifacts/smoke/figs/class_distribution.png`（过滤前/后分布）
- `artifacts/smoke/figs/embedding_2d.png`（UMAP/t-SNE 投影）
- `artifacts/smoke/figs/confusion_matrix.png`（混淆矩阵热力图）
- `artifacts/smoke/reports/metrics.json` 以及按类别的评估报告
- `artifacts/smoke/embeddings/` 下的嵌入缓存（用于加速复跑）

执行 `make ui` 后，UI 应展示：

- 当前 Top-3 省份预测 + 置信度
- 置信度随时间变化曲线
- Reset（重置）按钮可清空历史
