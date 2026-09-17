# paper-skills

论文推荐 + 周期分析 skill 集合（Claude Code / OpenAI Codex / dsh / pencode / qoder / trae 通用），面向本论文工作区（目录规范见工作区 `.AGENT.md`，本文件中的一切路径都以它为准）。

**不需要任何大模型 API key**——脚本只做 HTTP 检索、候选预筛与落盘；写概览、研究评价、主张-证据审计和周度综合这些语义工作由当前 agent（宿主模型）完成。

所有论文评价共用 [`research-evaluation-rubric.md`](research-evaluation-rubric.md)：元数据阶段只给研究优先级，读过正文后才按论文类型做证据化学术评价。会议、引用和摘要宣传词都不能替代全文判断。

## 六个 skill

| Skill | 作用 | 产出位置 |
|---|---|---|
| `paper-analyze` | 对单篇论文进行深度分析，生成可在通用 Markdown 工具中阅读的标准笔记 | `03-notes/<YYYY-MM>/<论文标题>/精读.md` 及同目录 `images/` |
| `paper-daily` | 多源检索（arXiv + OpenReview）→ 四维预筛 + 研究人员复排 → 前 K 篇 PDF 归档 → 今日检索日报 | `01-raw/<YYYY-MM>/<论文标题>.pdf`、`08-daily/<日期>/今日检索.md` |
| `paper-conf` | 按会议+年份检索顶会论文（DBLP + Semantic Scholar，三维预筛 + 语义复排） | `08-daily/<运行日期>/顶会论文推荐.md` |
| `paper-weekly` | 每 7 天一次：建立问题与证据地图，综合共识/冲突/空白，再结合个人卡点形成可证伪的研究机会与计划 | `07-research/<周起始日>-第N周周报.md` |
| `paper-interests` | 对话式维护研究主题：询问 → 扩散关键词 → 确认 → 写配置 | `.claude/skills/config.yaml` |
| `paper-sgd-reading` | 随机梯度类理论论文的**交互式精读家教**：逐个公式讲、不跳步、禁生活类比；问答过程落盘，定稿进公式目录 | 过程：`08-reading/<标题>/`；定稿：`04-equation_problem/<标题>/全局推理.md` 等 |

**不属于本套 skill 的**：`02-markdown` 的 PDF 解析由用户自己的流程负责，本套 skill 只读它用于去重与周期分析。`03-notes` 的精读笔记可以由用户手写，也可以由 `paper-analyze` 生成——两者都写入同一个固定入口 `03-notes/<YYYY-MM>/<论文标题>/精读.md`。

`paper-analyze` 的落盘位置默认由工作区决定（自动识别工作区，再读 `.claude/skills/config.yaml` 的 `notes_dir`，默认 `03-notes`），可用 `--output-dir` / `--notes-dir` 覆盖；月份默认按 `01-raw` 中同名 PDF 所在月份推断，也可用 `--month` 指定。

`paper-sgd-reading` 是当前集合中的独立 skill，落盘路径使用配置驱动（不写死 `/mnt/user-data/outputs/`），过程材料与定稿分离，文件名主干与工作区对齐。**产出文件名保持不变**（`全局推理.md` / `公式伴读.md` / `附录完整推导.md`）。

## 与工作区的对应关系

```text
<workspace>/
├── 01-raw/                              <- paper-daily 下载归档（已存在即停）
│   ├── README.md  index.md                 <- 统一论文登记表（月份/入库日期/四阶段状态）
│   └── YYYY-MM/README.md  <论文标题>.pdf
├── 02-markdown/      YYYY-MM/<论文标题>.md    <- 解析结果，只读
├── 03-notes/         YYYY-MM/<论文标题>/精读.md <- paper-analyze 产出；周报只读
├── 04-equation_problem/<论文标题>/            <- 公式推理定稿；周报只读"疑问"文件与标题骨架（不月份化）
├── 05-books/                                  （不涉及）
├── 06-translation/   YYYY-MM/<论文标题>/      <- 翻译产物；周报只标注存在性
├── 07-research/      <周起始日>-第N周周报.md   <- paper-weekly 输出（不月份化）
├── 08-daily/
│   ├── README.md
│   └── 2026-09-16/
│       ├── 今日检索.md                        <- 日报
│       ├── 顶会论文推荐.md                    <- paper-conf 产出
│       ├── daily-editorial.json               <- 概览与前 3 篇分析（可重渲染）
│       ├── search_result.json                 <- 全量检索结果（可回溯）
│       └── _index.json                        <- 当天落盘索引
└── 08-reading/<论文标题>/                      <- AI 思考过程与问答记录（周报只读，不月份化）
    ├── <论文标题>_精读笔记.md
    └── <论文标题>_全局推理_过程.md
```

`<论文标题>` 是跨目录稳定主干，`paper_config.sanitize_paper_title()` 把空格与连字符统一转下划线（如 `lp-norm` → `lp_norm`、`Memory-Efficient` → `Memory_Efficient`）。**月份与主干在 `01-raw` / `02-markdown` / `03-notes` / `06-translation` 四个阶段必须完全一致**，否则跨目录引用会断；月份 = 论文首次进入 `01-raw` 的日期所在月。

**落盘分工（严格不重叠）**：

| 目录 | 放什么 | 谁维护 |
|---|---|---|
| `01-raw/` | 原始 PDF + 入库日期记录（`01-raw/YYYY-MM/README.md` 权威） | `paper-daily` 下载 + `sync_indexes.py` 维护索引 |
| `02-markdown/` | PDF 解析结果；未解析/失败状态登记在索引 | 用户流程 + `sync_indexes.py` 维护状态 |
| `03-notes/` | AI 精读（`精读.md`）+ 用户自己的笔记 | `paper-analyze` 或用户 |
| `04-equation_problem/` | 公式推理**定稿**（`全局推理.md`、`公式伴读.md`、`附录完整推导.md`）+ 你的 `疑问.md` | `paper-sgd-reading` 定稿 + 用户 |
| `06-translation/` | 翻译产物（双栏对比、单独翻译等） | 用户流程 + `sync_indexes.py` 维护状态 |
| `08-reading/` | AI **过程**材料：逐单元问答记录、全局推理中间草稿 | `paper-sgd-reading` 过程 |

## 目录结构

```
.claude/skills/
├── config.yaml                      # 共享配置：研究领域、目录映射、周期设置
├── research-evaluation-rubric.md    # 博士/博士后视角的统一证据化评审准则
├── paper-analyze/
│   ├── SKILL.md                     # 单篇论文分析与标准 Markdown 笔记
│   └── scripts/generate_note.py
├── paper-daily/
│   ├── SKILL.md
│   ├── agents/openai.yaml
│   └── scripts/
│       ├── search_arxiv.py          # 多源检索 + 四维研究优先级预筛 + 去重标记
│       ├── fetch_pdfs.py            # PDF 归档到 01-raw/<YYYY-MM>/（已存在即停）
│       ├── render_note.py           # 按主题分组渲染日报（链接相对日报文件）
│       ├── write_note.py            # 日报/顶会推荐落盘到 08-daily/<日期>/
│       ├── sync_indexes.py          # 四份根索引 + 月份 README 表格 + --check-links（共享）
│       ├── localize_markdown_images.py  # 解析插图本地化到 02-markdown/YYYY-MM/images/
│       ├── scan_existing_notes.py   # 已有论文笔记关键词索引（默认 03-notes）
│       ├── link_keywords.py         # 关键词 → 标准 Markdown 链接
│       ├── common_words.py
│       └── paper_config.py          # 配置查找 / 月份路径解析 / 主干规范化（共享）
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

五个工作区型 skill 共用一份 `config.yaml`，**研究目标与主题一律从 `research_domains` 读取**；`paper-analyze` 是独立的单篇分析 skill，不依赖这份共享配置：

| 谁 | 从配置读什么 |
|---|---|
| `paper-daily` | 检索关键词（各域 `keywords` 并集）、`arxiv_categories`、域 `priority`、数据源、规模 |
| `paper-conf` | 关键词默认回落到同一份 `research_domains`（`conf.keywords` 可覆盖） |
| `paper-weekly` | 报告的主题骨架 = `research_domains` 的领域名与关键词 |
| `paper-interests` | **唯一写入口**：把对话结果合并进 `research_domains` |
| `paper-sgd-reading` | 目录 `reading_dir` / `equation_dir`（默认 `08-reading` / `04-equation_problem`，均**不月份化**）——**路径不写死在 SKILL.md 里** |
| `paper-analyze` | 工作区内默认写 `<notes_dir>/<YYYY-MM>/<论文标题>/精读.md`（`notes_dir` 读共享配置，默认 `03-notes`）；优先级：`--output-dir`/`--notes-dir` > 配置 `notes_dir` > `03-notes` |

因此：**换研究主题只需改 `config.yaml`（或用 `paper-interests`），不需要动任何 skill**。

配套约束：
- 脚本里**没有**默认研究领域；配置缺失或无 `research_domains` 时会明确告警并返回空结果，而不是塞一套内置主题
- SKILL.md 里出现的领域名都是**格式示例**，实际以配置为准
- 各 skill 的目录职责也不重叠（见上表"落盘分工"）；`paper-weekly` 对 `02-markdown`/`03-notes`/`06-translation` 一律只读
- **月份与索引**：月份规则、`README.md`/`index.md` 要求、状态词表都由工作区 `.AGENT.md` 规定；脚本只从 `config.yaml` 的 `layout.month_dirs` 读“哪些目录按月分层”，不重复实现规范

## 配置

```yaml
language: zh
workspace_path: ""                     # 留空则自动识别当前项目根目录

layout:                             # 月份分层与索引（见工作区 .AGENT.md）
  month_dirs:                        # 这些目录使用 YYYY-MM/ 分目录
    - "01-raw"
    - "02-markdown"
    - "03-notes"
    - "06-translation"
  month_pattern: "YYYY-MM"
  readme_name: "README.md"
  index_name: "index.md"
  index_markers: ["<!-- INDEX:BEGIN -->", "<!-- INDEX:END -->"]

papers_dir: "01-raw"                # PDF 落盘（写入其下 YYYY-MM/）+ 已存在扫描
markdown_dir: "02-markdown"         # 解析结果（只读）
notes_dir: "03-notes"               # 精读笔记（paper-analyze 写入 / 周报只读）
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
    subdomains:                         # 日报按命中关键词归类的细分领域
      随机梯度与收敛性: {keywords: [SGD, convergence analysis, ...]}
    arxiv_categories: [math.OC, stat.ML, cs.LG]
    priority: 5
  大模型训练优化:
    keywords: [optimizer design, memory-efficient training, ZeRO, muP, ...]
    arxiv_categories: [cs.LG, cs.CL]
    priority: 4
```

工作区路径按以下优先级解析：命令行 `--workspace` → 环境变量
`PAPER_WORKSPACE_PATH` → 配置项 `workspace_path` → 自动识别当前项目根目录。
因此同一套 skills 可以在不同电脑或不同项目目录中复用，不需要修改盘符。

`weekly.start_date` / `last_run` 由 `collect.py --mark-run` 写入（写前校验 YAML + 备份），**不要手动改**。

## 使用

```bash
# 默认自动识别当前项目根目录；也可显式覆盖
WORKSPACE="${PAPER_WORKSPACE_PATH:-$PWD}"

# 1. 检索（全量 all_papers + 过滤后的 top_papers）
python .claude/skills/paper-daily/scripts/search_arxiv.py --output "$WORKSPACE/08-daily/2026-09-16/search_result.json" \
  --target-date 2026-09-16 --exclude-known

# 2. 前 3 篇 PDF 归档到 01-raw/2026-09/（入库月份 = --date 所在月；已存在即停）
#    归档后自动更新 01-raw/2026-09/README.md 与四份根索引
python .claude/skills/paper-daily/scripts/fetch_pdfs.py --papers-json "$WORKSPACE/08-daily/2026-09-16/search_result.json" --date 2026-09-16

# 3. 生成按主题分组的日报 Markdown（本地链接为 ../../01-raw/2026-09/…）
python .claude/skills/paper-daily/scripts/render_note.py \
  --date 2026-09-16 \
  --papers-json "$WORKSPACE/08-daily/2026-09-16/search_result.json" \
  --editorial-json "$WORKSPACE/08-daily/2026-09-16/daily-editorial.json" > note.md

# 4. 日报落盘到 08-daily/2026-09-16/今日检索.md
python .claude/skills/paper-daily/scripts/write_note.py --date 2026-09-16 \
  --papers-json "$WORKSPACE/08-daily/2026-09-16/search_result.json" --stdin-file note.md

# 4a. 解析插图本地化（MinerU CDN -> 02-markdown/YYYY-MM/images/<论文标题>/）
python .claude/skills/paper-daily/scripts/localize_markdown_images.py --replace-failed

# 4b. 刷新索引与状态清单，并校验链接（0 问题通过）
python .claude/skills/paper-daily/scripts/sync_indexes.py
python .claude/skills/paper-daily/scripts/sync_indexes.py --check-links

# 4c. 顶会推荐：写到 08-daily/<运行日期>/顶会论文推荐.md
python .claude/skills/paper-daily/scripts/write_note.py --date "<运行日期>" \
  --filename 顶会论文推荐.md --stdin-file conf_note.md

# 5. 周期分析：先看是否到期 + 收集素材
python .claude/skills/paper-weekly/scripts/collect.py --output weekly_manifest.json
#    ... 读完清单里的素材、写出 07-research/<日期>-<主题>.md 之后 ...
python .claude/skills/paper-weekly/scripts/collect.py --mark-run

# 6. 改研究主题
python .claude/skills/paper-interests/scripts/merge_interests.py --preview --payload payload.json
python .claude/skills/paper-interests/scripts/merge_interests.py --apply   --payload payload.json

# 测试（从 .claude/skills 目录运行）
cd .claude/skills && python -m unittest discover -s tests -v
```

在 agent 里用自然语言触发即可，命令细节见各 `SKILL.md`。

## 安装

本项目把全部论文 skills 的实体文件统一放在 `.claude/skills/`。为同时兼容各 agent 工具，额外提供以下入口（**这是工作区唯一允许的符号链接形式**，见 `.AGENT.md`）：

```text
.agents/skills   -> ../.claude/skills  # Codex 项目级发现入口
.dsh/skills      -> ../.claude/skills  # dsh 项目级发现入口
.pencode/skills  -> ../.claude/skills  # pencode 项目级发现入口
.qoder/skills    -> ../.claude/skills  # qoder 项目级发现入口
.trae/skills     -> ../.claude/skills  # trae 项目级发现入口
```

论文资料目录（`01-raw` ~ `08-reading`）内不得出现任何符号链接。

`.claude/skills/` 下的每个 skill 仍然是直接子目录，并直接包含 `SKILL.md`；不要再增加一层集合目录，否则 Codex 不会递归发现这些 skill。

```powershell
# Codex
Copy-Item -Recurse .claude\skills\paper-analyze, .claude\skills\paper-daily, .claude\skills\paper-conf, .claude\skills\paper-interests, .claude\skills\paper-weekly, .claude\skills\paper-sgd-reading "$env:USERPROFILE\.agents\skills\"
Copy-Item .claude\skills\config.yaml, .claude\skills\research-evaluation-rubric.md "$env:USERPROFILE\.agents\skills\"
# Claude Code：同样六个目录、config.yaml 与研究评价准则复制到 $env:USERPROFILE\.claude\skills\
# 项目内安装（本仓库用法）：
Copy-Item -Recurse .claude\skills\* "<论文工作区>\.claude\skills\"   # 路径按实际调整
```

安装约束分两类：

- 所有涉及评价的 skill 都读取父目录的 `research-evaluation-rubric.md`；复制 skill 时必须一并复制该文件
- `paper-daily` / `paper-conf` / `paper-interests` / `paper-weekly` 必须放在**同一父目录**，共享的 `config.yaml` 放父目录（它们从 `../paper-daily/scripts` 复用配置解析、月份路径解析、索引刷新与预筛函数）
- **`paper-analyze` 代码自包含**：`scripts/generate_note.py` 不 import 其他 skill 的代码，自行向上识别工作区、从共享配置读取 `notes_dir`（读不到就回落 `03-notes`）；提示词评价仍需父目录的共享准则
- **`paper-sgd-reading` 代码自包含**：`scripts/outputs.py` 不 import 其他 skill 的代码，可单独安装；提示词评价仍需父目录的共享准则。配置按 `--config` → `$PAPER_SKILLS_CONFIG` → `$PAPER_READING_CONFIG` → `<skill 同级>/config.yaml` → 由 `$PAPER_WORKSPACE_PATH` 查找 `<workspace>/.claude/skills/config.yaml` → 从工作目录向上查找 的顺序定位

## 环境搭建（Conda）

以下命令在论文工作区根目录执行。环境名可按需要修改；后续运行各 skill 前先激活该环境。

```bash
# 创建 Python 3.11 环境，并安装脚本依赖
conda create -n paper-skills python=3.11 pyyaml requests -c conda-forge

# 激活环境
conda activate paper-skills

# 验证 Python 与依赖
python --version
python -c "import yaml, requests; print('PyYAML', yaml.__version__); print('requests', requests.__version__)"
```

如果环境已经存在，只需执行：

```bash
conda activate paper-skills
```

例如，从项目根目录运行每日检索：

```bash
python .claude/skills/paper-daily/scripts/search_arxiv.py \
  --output "$PWD/08-daily/$(date +%F)/search_result.json" \
  --target-date "$(date +%F)" \
  --exclude-known
```

- Python 3.8+
- 网络：`export.arxiv.org`、`api2.openreview.net`、`api.semanticscholar.org`、`dblp.org`（见下）
- 可选：`semantic_scholar_api_key` 缓解 S2 匿名限流（429）

## 已知限制

**DBLP 反爬**：`paper-conf` 依赖 DBLP，而 DBLP 目前对非浏览器流量返回 Anubis JS 验证页（`Making sure you're not a bot`）。脚本会立即给出明确报错而不是空转重试。替代方案：用 `paper-daily --sources openreview` 检索 ICLR/NeurIPS/ICML 顶会论文。

**arXiv 响应慢**：高峰期单次请求可能超过 20 秒；可用 `--days` 收窄窗口，或先只跑 `openreview`。超过 `arxiv_request_timeout`（默认 120 秒）会重试。

**周期分析素材为空**：`02-markdown` / `03-notes` 按**文件修改时间**取窗口内文件（自动递归 `YYYY-MM/` 月份目录），留了 ±1 天容差。若你的解析流程会批量改写旧文件时间戳，请把 `weekly.window_days` 调大，或用 `--force` 手动触发。

**索引提示“悬空链接”**：说明某个 Markdown 里的相对路径指向了不存在的文件。月份迁移后最容易漏的是把 `03-notes/YYYY-MM/<标题>/精读.md` 里的路径仍写成 `../../`（该文件实际在工作区根目录下的第三层，应为 `../../../`）。执行 `sync_indexes.py --check-links`，按报告的文件与行号修正即可。
