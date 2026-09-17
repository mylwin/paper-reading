# paper-skills

论文推荐 + 周期分析 skill 集合（Claude Code / OpenAI Codex 通用），面向 **paper-reading 论文工作区**（目录规范见工作区 `.AGENT.md`）。

**不需要任何大模型 API key**——脚本只做 HTTP 检索、评分与落盘；写概览、总结、贡献点、周度分析这些语义工作由当前 agent（宿主模型）完成。

## 五个 skill

| Skill | 作用 | 产出位置 |
|---|---|---|
| `paper-daily` | 多源检索（arXiv + OpenReview）→ 四维评分 → 前 K 篇 PDF 归档 → 今日检索日报 | `01-raw/*.pdf`、`08-daily/<日期>/今日检索.md` |
| `paper-conf` | 按会议+年份检索顶会论文（DBLP + Semantic Scholar，三维评分） | `08-daily/<年份>-顶会论文推荐.md` |
| `paper-weekly` | 每 7 天一次：先宏观看研究主题发展（08-daily 题目全景 + 03-notes），再看你不理解的地方（04 疑问 + 08-reading 问答），给出优化方向与下一步 | `07-research/<周起始日>-第N周周报.md` |
| `paper-interests` | 对话式维护研究主题：询问 → 扩散关键词 → 确认 → 写配置 | `paper-skills/config.yaml` |
| `paper-sgd-reading` | 随机梯度类理论论文的**交互式精读家教**：逐个公式讲、不跳步、禁生活类比；问答过程落盘，定稿进公式目录 | 过程：`08-reading/<标题>/`；定稿：`04-equation_problem/<标题>/全局推理.md` 等 |

**不属于本套 skill 的**：`02-markdown` 的 PDF 解析、`03-notes` 的精读笔记 —— 由用户自己的流程负责，本套 skill 只读它们用于去重与周期分析。

`paper-sgd-reading` 是从项目内原有的 `.claude/skills/sgd-paper-reading` **新建的独立副本**（原件保持不动），改动只有三处：落盘路径改为配置驱动（不再写死 `/mnt/user-data/outputs/`）、过程材料与定稿分离、文件名主干与工作区对齐。**产出文件名保持不变**（`全局推理.md` / `公式伴读.md` / `附录完整推导.md`）。

## 与 paper-reading 工作区的对应关系

```text
paper-reading/
├── 01-raw/            <论文标题>.pdf        <- paper-daily 下载归档（已存在即停）
│   └── index.md                              <- 只记录题目 + 落盘日期
├── 02-markdown/       <论文标题>.md          <- 用户流程，只读
├── 03-notes/          <论文标题>/精读.md     <- 用户流程，只读
├── 04-equation_problem/<论文标题>/           <- 公式推理定稿；周报只读"疑问"文件与标题骨架
├── 05-books/                                  （不涉及）
├── 06-translation/    <论文标题>/            <- 翻译产物；周报只标注存在性
├── 07-research/       <周起始日>-第N周周报.md <- paper-weekly 输出
├── 08-daily/
│   └── 2026-09-16/
│       ├── 今日检索.md                       <- 日报
│       ├── search_result.json                <- 全量检索结果（可回溯）
│       └── _index.json                       <- 当天落盘索引
└── 08-reading/                                <- AI 思考过程与问答记录（周报只读）
    └── <论文标题>/
        ├── <论文标题>_精读笔记.md
        └── <论文标题>_全局推理_过程.md
```

`<论文标题>` 是跨目录稳定主干：`paper_config.sanitize_paper_title()` 把空格与连字符统一转下划线（如 `lp-norm` → `lp_norm`、`Memory-Efficient` → `Memory_Efficient`），与工作区现有 PDF 命名风格一致。**PDF 文件名与 `02-markdown` 主干必须一致**，否则跨目录引用会断。

**落盘分工（严格不重叠）**：

| 目录 | 放什么 | 谁维护 |
|---|---|---|
| `04-equation_problem/` | 公式推理**定稿**（`全局推理.md`、`公式伴读.md`、`附录完整推导.md`）+ 你的 `疑问.md` | `sgd-paper-reading` 定稿 + 用户 |
| `08-reading/` | AI **过程**材料：逐单元问答记录、全局推理中间草稿 | `sgd-paper-reading` 过程 |

## 目录结构

```
paper-skills/
├── config.yaml                      # 共享配置：研究领域、目录映射、周期设置
├── paper-daily/
│   ├── SKILL.md
│   ├── agents/openai.yaml
│   └── scripts/
│       ├── search_arxiv.py          # 多源检索 + 四维评分 + 去重标记
│       ├── fetch_pdfs.py            # PDF 归档到 01-raw（已存在即停）
│       ├── write_note.py            # 日报落盘到 08-daily/<日期>/
│       ├── scan_existing_notes.py   # 已有论文笔记关键词索引
│       ├── link_keywords.py         # 关键词 → wikilink（Obsidian 场景用）
│       ├── common_words.py
│       └── paper_config.py          # 配置查找 / 路径解析 / 主干规范化（共享）
├── paper-conf/
│   ├── SKILL.md
│   ├── config.yaml                  # 会议/年份/排除词（关键词默认回落到共享配置）
│   └── scripts/search_conf_papers.py
├── paper-weekly/
│   ├── SKILL.md
│   └── scripts/collect.py           # 周期判断 + 素材清单
├── paper-interests/
│   ├── SKILL.md
│   └── scripts/merge_interests.py   # 安全合并研究领域进 config.yaml
├── paper-sgd-reading/
│   ├── SKILL.md                     # 交互式/全局推理两种模式的教学法
│   ├── agents/openai.yaml
│   ├── assets/session-log-template.md
│   ├── references/                  # 公式/算法/假设讲解模板 + 术语表
│   └── scripts/outputs.py           # 路径解析 + 过程/定稿落盘（自包含，可独立安装）
└── tests/test_paper_skills.py
```

## 设计原则：解耦 —— 研究主题只有一处真相源

四个 skill 共用一份 `config.yaml`，**研究目标与主题一律从 `research_domains` 读取**，任何 skill、脚本、SKILL.md 都**不内置具体主题**：

| 谁 | 从配置读什么 |
|---|---|
| `paper-daily` | 检索关键词（各域 `keywords` 并集）、`arxiv_categories`、域 `priority`、数据源、规模 |
| `paper-conf` | 关键词默认回落到同一份 `research_domains`（`conf.keywords` 可覆盖） |
| `paper-weekly` | 报告的主题骨架 = `research_domains` 的领域名与关键词 |
| `paper-interests` | **唯一写入口**：把对话结果合并进 `research_domains` |
| `paper-sgd-reading` | 目录 `reading_dir` / `equation_dir`（默认 `08-reading` / `04-equation_problem`）——**路径不写死在 SKILL.md 里** |

因此：**换研究主题只需改 `config.yaml`（或用 `paper-interests`），不需要动任何 skill**。

配套约束：
- 脚本里**没有**默认研究领域；配置缺失或无 `research_domains` 时会明确告警并返回空结果，而不是塞一套内置主题
- SKILL.md 里出现的领域名都是**格式示例**，实际以配置为准
- 各 skill 的目录职责也不重叠（见上表"落盘分工"），`04`/`06`/`08-reading` 只读不写

## 配置

```yaml
language: zh
vault_path: "E:/paper-reading"      # 留空则读 OBSIDIAN_VAULT_PATH

papers_dir: "01-raw"                # PDF 落盘 + 已存在扫描
markdown_dir: "02-markdown"         # 只读
notes_dir: "03-notes"               # 只读（精读笔记）
equation_dir: "04-equation_problem" # paper-sgd-reading 定稿 + 周报只读骨架与疑问
translation_dir: "06-translation"   # 只读
reading_dir: "08-reading"           # 思考过程/问答记录，周报只读
daily_dir: "08-daily"               # 日报，再按日期分层
research_dir: "07-research"         # 周期报告
daily_note_name: "今日检索"

sources: [arxiv, openreview]
arxiv_categories: [math.OC, stat.ML, cs.LG, cs.NA, cs.AI]
top_n: 10
pdf_top_k: 3
exclude_known: true                 # 推荐时排除知识库已有论文

weekly:
  interval_days: 7
  window_days: 7
  start_date: ""                    # 脚本自动维护，勿手改
  last_run: ""                      # 脚本自动维护，勿手改
  note_pattern: "{week_start}-第{week_no}周周报.md"
  theme_first: true
  use_git_history: true
  inputs:                           # 第一阶段（宏观），按优先级读取
    - "08-daily"
    - "03-notes"
    - "02-markdown"
  context_dirs:                     # 第二阶段（针对性）
    - "04-equation_problem"
    - "08-reading"
    - "06-translation"

research_domains:                   # 用 paper-interests 维护
  随机梯度与优化理论:
    keywords: [stochastic gradient descent, momentum, polyak step size, ...]
    arxiv_categories: [math.OC, stat.ML, cs.LG]
    priority: 5
  大模型训练优化:
    keywords: [optimizer design, memory-efficient training, ZeRO, muP, ...]
    arxiv_categories: [cs.LG, cs.CL]
    priority: 4
```

`weekly.start_date` / `last_run` 由 `collect.py --mark-run` 写入（写前校验 YAML + 备份），**不要手动改**。

## 使用

```bash
VAULT="$OBSIDIAN_VAULT_PATH"   # 或配置 vault_path

# 1. 检索（全量 all_papers + 过滤后的 top_papers）
python paper-daily/scripts/search_arxiv.py --output "$VAULT/08-daily/2026-09-16/search_result.json" \
  --target-date 2026-09-16 --exclude-known

# 2. 前 3 篇 PDF 归档到 01-raw（已存在即停）
python paper-daily/scripts/fetch_pdfs.py --papers-json "$VAULT/08-daily/2026-09-16/search_result.json" --date 2026-09-16

# 3. 日报落盘到 08-daily/2026-09-16/今日检索.md
python paper-daily/scripts/write_note.py --date 2026-09-16 \
  --papers-json "$VAULT/08-daily/2026-09-16/search_result.json" --stdin-file note.md

# 4. 周期分析：先看是否到期 + 收集素材
python paper-weekly/scripts/collect.py --output weekly_manifest.json
#    ... 读完清单里的素材、写出 07-research/<日期>-<主题>.md 之后 ...
python paper-weekly/scripts/collect.py --mark-run

# 5. 改研究主题
python paper-interests/scripts/merge_interests.py --preview --payload payload.json
python paper-interests/scripts/merge_interests.py --apply   --payload payload.json

# 测试
python -m unittest discover -s tests -v
```

在 agent 里用自然语言触发即可，命令细节见各 `SKILL.md`。

## 安装

```powershell
# Codex
Copy-Item -Recurse paper-skills\paper-daily, paper-skills\paper-conf, paper-skills\paper-interests, paper-skills\paper-weekly, paper-skills\paper-sgd-reading "$env:USERPROFILE\.agents\skills\"
# Claude Code：同样五个目录复制到 $env:USERPROFILE\.claude\skills\
# 项目内安装（本仓库用法）：
Copy-Item -Recurse paper-skills\* "$env:USERPROFILE\..\paper-reading\.claude\skills\"   # 路径按实际调整
```

安装约束分两类：

- `paper-daily` / `paper-conf` / `paper-interests` / `paper-weekly` 必须放在**同一父目录**，共享的 `config.yaml` 放父目录（它们从 `../paper-daily/scripts` 复用配置解析与评分函数）
- **`paper-sgd-reading` 自包含**：`scripts/outputs.py` 不 import 其他 skill 的代码，可单独安装。配置按 `--config` → `$PAPER_SKILLS_CONFIG` → `$PAPER_READING_CONFIG` → `<skill>/config.yaml` → 由 `$OBSIDIAN_VAULT_PATH` 反推 `<vault>/../paper-skills/config.yaml` → 从工作目录向上查找 的顺序定位

## 依赖

```
pip install PyYAML requests
```

- Python 3.8+
- 网络：`export.arxiv.org`、`api2.openreview.net`、`api.semanticscholar.org`、`dblp.org`（见下）
- 可选：`semantic_scholar_api_key` 缓解 S2 匿名限流（429）

## 已知限制

**DBLP 反爬**：`paper-conf` 依赖 DBLP，而 DBLP 目前对非浏览器流量返回 Anubis JS 验证页（`Making sure you're not a bot`）。脚本会立即给出明确报错而不是空转重试。替代方案：用 `paper-daily --sources openreview` 检索 ICLR/NeurIPS/ICML 顶会论文。

**arXiv 响应慢**：高峰期单次请求可能超过 20 秒；可用 `--days` 收窄窗口，或先只跑 `openreview`。超过 `arxiv_request_timeout`（默认 120 秒）会重试。

**周期分析素材为空**：`02-markdown` / `03-notes` 按**文件修改时间**取窗口内文件，留了 ±1 天容差。若你的解析流程会批量改写旧文件时间戳，请把 `weekly.window_days` 调大，或用 `--force` 手动触发。
