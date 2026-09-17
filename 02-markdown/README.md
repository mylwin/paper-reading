# 论文解析结果

`02-markdown/` 用于保存从 `01-raw/` 中的论文 PDF 解析得到的 Markdown 文本。

## 目录规则

```text
02-markdown/
├── README.md
├── index.md
└── YYYY-MM/
    ├── README.md
    └── <论文标题>.md
```

- 月份与文件名主干必须与对应 `01-raw/YYYY-MM/<论文标题>.pdf` 完全一致（月份 = 论文入库月份）。
- 内容应尽量保持论文原文结构，作为阅读和后续整理的内容来源。
- 不在解析文本中混入个人笔记、批注或 AI 分析；相关内容请放入 `03-notes/YYYY-MM/<论文标题>/`。
- 解析结果是机器产物（例如 MinerU 输出），其中的图片可能仍是解析服务的远程 URL；远程图片不算跨目录引用。
- 尚未解析的论文不创建空文件，而是在月份 `README.md` 与 `index.md` 中登记为 `未解析`；解析失败登记为 `解析失败` 并写明原因。

## 说明文件

- `index.md`：解析总索引，含状态、解析文件路径、入库日期与失败原因。
- `YYYY-MM/README.md`：该月论文的解析清单与未解析提醒。

状态由 `sync_indexes.py` 依据磁盘文件刷新，人工填写的失败原因会保留：

```bash
python .claude/skills/paper-daily/scripts/sync_indexes.py
```

完整规范见工作区 `.AGENT.md`。
