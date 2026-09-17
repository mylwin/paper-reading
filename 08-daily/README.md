# 每日检索日报

`08-daily/` 用于保存每日论文检索与顶会推荐产物，按日期目录隔离。

## 目录规则

```text
08-daily/
├── README.md
└── YYYY-MM-DD/
    ├── 今日检索.md
    ├── 顶会论文推荐.md
    ├── daily-editorial.json
    ├── search_result.json
    └── _index.json
```

- 日期目录名使用 `YYYY-MM-DD`，是本次检索的运行日期；文件名本身不再带日期。
- `今日检索.md`：每日检索日报（默认文件名取配置 `daily_note_name`）。
- `顶会论文推荐.md`：`paper-conf` 的顶会推荐笔记，与日报同构放在运行日期目录下。
- `daily-editorial.json`：本次人工撰写的概览与前 3 篇分析，保留它就能在链接或目录结构变化后**重新渲染**日报而不丢失正文。
- `search_result.json`：`search_arxiv.py` 的全量检索结果（可回溯）。
- `_index.json`：本次日报与落盘论文的索引。

## 链接规则

日报中的本地链接必须是**从日报文件出发**的相对路径（本目录下的文件位于工作区根目录的第二层，需要 `../../`）：

```markdown
[PDF](../../01-raw/2026-09/<论文标题>.pdf)
[精读](../../03-notes/2026-09/<论文标题>/精读.md)
[相关日报](今日检索.md)
```

引用论文必须先核验真实资产是否存在；不存在的资料用远程链接或标注"暂无"，不得捏造本地路径。

生成与校验：

```bash
python .claude/skills/paper-daily/scripts/render_note.py --date <日期> \
  --papers-json 08-daily/<日期>/search_result.json > note.md
python .claude/skills/paper-daily/scripts/write_note.py --date <日期> \
  --papers-json 08-daily/<日期>/search_result.json --stdin-file note.md
python .claude/skills/paper-daily/scripts/sync_indexes.py --check-links
```

完整规范见工作区 `.AGENT.md`。
