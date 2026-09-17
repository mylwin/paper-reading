# 原始论文

`01-raw/` 仅用于保存用户提供或下载的论文原始 PDF，是论文资料的原始来源，也是**入库时间的权威记录**。

## 目录规则

```text
01-raw/
├── README.md
├── index.md
└── YYYY-MM/
    ├── README.md
    └── <论文标题>.pdf
```

- 按论文**首次进入本目录的日期**分月份归档：`YYYY-MM/` 即入库日期的年月。
- 文件名使用论文标题主干 `<论文标题>`，并与 `02-markdown/YYYY-MM/` 中对应 Markdown、`03-notes/YYYY-MM/`、`06-translation/YYYY-MM/` 中的论文目录保持相同主干。
- 同一篇论文在四个阶段必须使用相同月份与相同主干；文件名中不写日期，日期记录在表格里。
- 保留 PDF 原始内容，不在此目录保存解析文本、笔记、图片或翻译结果。
- 不覆盖已有原始 PDF；目标文件已存在时停止下载。

## 说明文件

- `01-raw/YYYY-MM/README.md` 记录该月的**入库清单**（入库日期、论文标题、主干、来源、PDF 链接），是入库日期的权威来源。
- `index.md` 是跨阶段**统一论文登记表**：月份、入库日期、来源，以及解析/精读/翻译状态；由 `sync_indexes.py` 依据磁盘实际文件生成。

```bash
# 刷新四份索引与各月份 README 表格
python .claude/skills/paper-daily/scripts/sync_indexes.py

# 归档新论文到当月目录（写入月份 README 与统一登记表）
python .claude/skills/paper-daily/scripts/fetch_pdfs.py \
  --papers-json 08-daily/<日期>/search_result.json --date <日期>
```

完整规范见工作区 `.AGENT.md`。
