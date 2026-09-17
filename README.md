# 论文工作区

本工作区用于保存论文原文、解析结果、精读笔记、公式解读、翻译结果，以及书籍、调研、日报与精读过程材料。

**目录结构与命名的事实标准是 `.AGENT.md`**；本文件只是导航入口，与其他目录的 `README.md` 一样服从 `.AGENT.md`。

## 目录一览

| 目录 | 内容 | 是否按 `YYYY-MM/` 分层 |
|---|---|---|
| `01-raw/` | 原始论文 PDF（入库时间的权威记录） | 是 |
| `02-markdown/` | PDF 解析结果 | 是 |
| `03-notes/` | 精读笔记（`<论文标题>/精读.md` 为固定入口） | 是 |
| `04-equation_problem/` | 公式解读、推导与附件 | 否（按论文目录） |
| `05-books/` | 学习参考书籍 | 否 |
| `06-translation/` | 翻译结果（双栏对比、单独翻译等） | 是 |
| `07-research/` | 专题调研报告（`YYYY-MM-DD-<主题>.md`） | 否 |
| `08-daily/` | 每日检索日报与顶会推荐（`YYYY-MM-DD/`） | 否（按日期目录） |
| `08-reading/` | 交互式精读的过程材料（问答记录、推理草稿） | 否（按论文目录） |
| `.claude/skills/` | paper-* 技能实体（其他工具目录通过 symlink 指向此处） | — |

## 核心规则（摘要，完整见 `.AGENT.md`）

1. **月份口径**：`01-raw`、`02-markdown`、`03-notes`、`06-translation` 按论文**首次进入 `01-raw` 的日期**所在月分层；同一篇论文在四个阶段使用**相同月份 + 相同 `<论文标题>` 主干**，文件名不带日期。
2. **说明文件**：每个编号根目录与每个 `YYYY-MM/` 月份目录都有 `README.md`；`01-raw`、`02-markdown`、`03-notes`、`06-translation` 各有根 `index.md`。
3. **状态清单**：未解析 / 未精读 / 未翻译的论文必须出现在对应阶段的 `index.md` 与月份 `README.md` 中（`未解析`、`未精读`、`未翻译`、`解析失败` 为固定词表）。
4. **引用**：跨目录引用一律使用**相对所在文件**的真实路径；不使用符号链接（唯一例外是 `.agents/skills`、`.dsh/skills` 等技能发现入口）。
5. **无规划文件**：项目内不保留 `task_plan.md`、`findings.md`、`progress.md` 等过程文件。

## 常用命令

```bash
# 刷新四份索引与所有月份 README 表格（幂等）
python .claude/skills/paper-daily/scripts/sync_indexes.py

# 校验链接与月份/主干一致性（0 问题即通过）
python .claude/skills/paper-daily/scripts/sync_indexes.py --check-links

# 运行技能脚本的单元测试
cd .claude/skills && python -m unittest discover -s tests -v
```

技能使用说明见 `.claude/skills/paper-skills-README.md`。
