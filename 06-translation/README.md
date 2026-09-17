# 论文翻译结果

`06-translation/` 用于保存论文的翻译结果。翻译资料与原始 PDF、解析 Markdown 和阅读笔记相互独立。

## 目录规则

```text
06-translation/
├── README.md
├── index.md
└── YYYY-MM/
    ├── README.md
    └── <论文标题>/
        ├── 双栏对比.<扩展名>
        ├── 单独翻译.<扩展名>
        └── <其他翻译形式>.<扩展名>
```

- 月份为该论文的**入库月份**，论文目录名必须与 `01-raw/YYYY-MM/`、`02-markdown/YYYY-MM/`、`03-notes/YYYY-MM/` 中的 `<论文标题>` 一致。
- 可同时保存双栏对比、单独翻译等不同形式，文件名应体现翻译形式；翻译文件只放在对应论文目录内，不放在月份目录或根目录。
- **未翻译的论文不预建空目录**，在月份 `README.md` 与 `index.md` 中登记为 `未翻译`；进行中登记 `翻译中`，完成后登记 `已翻译` + 翻译形式与完成日期。
- 翻译结果不应覆盖或修改 `01-raw/` 中的原始 PDF 和 `02-markdown/` 中的解析结果。

## 说明文件

- `index.md`：翻译总索引，含状态、翻译形式、完成日期与目录路径。
- `YYYY-MM/README.md`：该月论文的翻译清单与待翻译提醒。

```bash
python .claude/skills/paper-daily/scripts/sync_indexes.py
```

完整规范见工作区 `.AGENT.md`。
