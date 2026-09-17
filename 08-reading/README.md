# 交互式精读过程材料

`08-reading/` 保存公式伴读 / 交互式精读的**过程材料**：逐单元问答记录、全局推理的中间草稿。

## 目录规则

```text
08-reading/
├── README.md
└── <论文标题>/
    ├── <论文标题>_精读笔记.md
    └── <论文标题>_全局推理_过程.md
```

- **本目录不按月份分层**（只有 `01-raw`、`02-markdown`、`03-notes`、`06-translation` 使用 `YYYY-MM/`），只按论文标题建立目录。
- 目录名主干与 `01-raw/YYYY-MM/<论文标题>.pdf` 等其他阶段保持一致。
- 过程材料留在本目录：问答记录写入 `<论文标题>_精读笔记.md`，全局推理草稿写入 `<论文标题>_全局推理_过程.md`。
- **定稿不放在这里**：可独立阅读的成品（`全局推理.md`、`公式伴读.md`、`附录完整推导.md`）写入 `04-equation_problem/<论文标题>/`。过程与定稿分离，便于 `paper-weekly` 判断卡点而不被草稿干扰。

落盘由 `paper-sgd-reading` 的 `scripts/outputs.py` 完成（目录从 `.claude/skills/config.yaml` 的 `reading_dir` 读取，不要手写路径）：

```bash
python .claude/skills/paper-sgd-reading/scripts/outputs.py --title "<论文标题>" --kind process --prepare
python .claude/skills/paper-sgd-reading/scripts/outputs.py --title "<论文标题>" --kind final --name 全局推理 --prepare
```

完整规范见工作区 `.AGENT.md`。
