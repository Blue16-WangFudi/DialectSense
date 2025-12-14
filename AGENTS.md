# 仓库规范

## 项目结构与数据布局
- 仓库根目录包含数据集资源与一个最小化的 `.gitignore`。
- `datasets/metadata.csv` 存储音频片段的元数据；保持列名稳定，新增列需记录说明。
- `datasets/oggs/` 存放按方言组织的音频；使用一致的命名方式，例如 `<dialect>_<speaker>_<id>.ogg`。
- `datasets/LICENSE` 与上游许可保持一致；当数据内容变化时同步更新。
- `datasets/README.txt` 指向上游文档；如本地有偏离之处，在此补充摘要说明。

## 构建、测试与开发命令
- 本仓库没有构建流水线；以数据为主。
- 仅使用标准库做快速元数据一致性检查：
  ```powershell
  python - <<'PY'
import csv, pathlib
p = pathlib.Path('datasets/metadata.csv')
with p.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print('headers:', rows[0].keys())
print('rows:', len(rows))
print('empty fields:', sum(any(v=='' for v in r.values()) for r in rows))
PY
  ```
- 统计每个方言文件夹中的音频数量：`Get-ChildItem datasets/oggs -Directory | ForEach-Object { $_.Name, (Get-ChildItem $_ -Filter *.ogg).Count }`。

## 代码风格与命名规范
- CSV 表头使用 `lowercase_with_underscores`；避免空格与行尾空白。
- 文本文件尽量使用 UTF-8 编码与 LF 行结尾。
- 辅助脚本优先使用 Python 3、4 空格缩进、语义清晰的函数命名。

## 测试规范
- 无自动化测试套件。在脚本的 `if __name__ == '__main__':` 中加入自检，并记录关键假设。
- 提交前手动核验新增元数据行（路径存在、标签与文件夹名一致、时长合理等）。

## Commit 与 Pull Request 规范
- Commit 信息：使用简短、祈使句式的主题（<=72 字符）；说明数据变更原因及来源。
- PR 需列出数据来源、文件数量（新增/更新的音频）、`metadata.csv` 的 schema 变更，以及用于验证的命令。
- 若涉及重命名/移动文件，请附简要说明，便于审阅者追踪变更。

## 数据治理与安全
- 未获得再分发许可时不要加入音频；确保 `datasets/LICENSE` 准确无误。
- 大规模新增建议按方言/说话人分子目录组织；除非在同一 PR 同步更新元数据，否则避免重命名。
- 在添加敏感内容或非公开录音前先创建 issue 讨论。
