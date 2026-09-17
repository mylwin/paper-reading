---
name: paper-daily
description: 多源检索每日论文推荐，把前几篇原文 PDF 归档到 01-raw，并在 08-daily/<日期>/ 写出今日检索日报。Use when the user asks to start the day, create today's paper brief, or generate paper recommendations for a specified date.
---

# 目标

每天为 paper-reading 论文工作区完成三件事：

1. **检索**：多源（arXiv + OpenReview）拉取最近一个月的新论文 + 过去一年高影响力论文
2. **推荐**：四维评分排序，排除知识库里已有的论文
3. **落盘**：前 K 篇（默认 3）原始 PDF 存到 `01-raw/`，日报写到 `08-daily/<日期>/今日检索.md`

**不负责**：`02-markdown` 解析结果与 `03-notes` 精读笔记由用户自己的流程负责，本 skill 只读它们用于去重。

# 工作区结构（paper-reading）

```text
01-raw/            原始 PDF              <- 本 skill 落盘
02-markdown/       PDF 解析结果          <- 用户流程
03-notes/<标题>/   精读等笔记            <- 用户流程
07-research/       周期调研报告          <- paper-weekly
08-daily/<日期>/   每日检索日报          <- 本 skill
```

论文稳定主干 `<论文标题>` 的规则见工作区 `.AGENT.md`：不带扩展名、可安全用作文件名与文件夹名。
**同一篇论文在 `01-raw`、`02-markdown`、`03-notes` 必须使用相同主干**——本 skill 的 PDF 文件名由 `paper_config.sanitize_paper_title()` 生成，替换 `[ /\\:*?"<>|]` 与逗号分隔符为 `_`、压缩连续空白、长度上限 120 字符。

# 配置

配置查找顺序：`--config` → `$PAPER_SKILLS_CONFIG` → `paper-daily/config.yaml` → `.claude/skills/config.yaml`。

workspace 路径：命令行 `--workspace` → `PAPER_WORKSPACE_PATH` → 配置的 `workspace_path` → 自动识别当前项目根目录。

## 研究主题的唯一真相源

**检索什么主题只由 `config.yaml` 的 `research_domains` 决定，本 skill 不内置任何具体研究方向。**

- 关键词 = 所有研究域 `keywords` 的并集
- arXiv 分类 = 配置的 `arxiv_categories`
- 域权重 = 配置的 `priority`（1-10）
- 想换主题 → 用 `paper-interests` skill 或直接改 `config.yaml`，**不要改本 SKILL.md 或脚本**
- 配置缺失或无 `research_domains` 时：脚本**不会**塞入默认主题，会明确告警并返回 0 篇；此时先引导用户配置

本文件示例里出现的领域名（如"随机梯度与优化理论"）**只是格式示例**，实际以配置为准。

关键项（共享 `config.yaml`）：

| 配置 | 默认 | 作用 |
|---|---|---|
| `papers_dir` | `01-raw` | PDF 落盘 + 已存在扫描 |
| `notes_dir` | `03-notes` | 已精读论文扫描 |
| `daily_dir` | `08-daily` | 日报目录，按日期再分层 |
| `daily_note_name` | `今日检索` | 日报文件名（日期已在文件夹名上，故文件名不再带日期） |
| `pdf_top_k` | `3` | 前几篇下载 PDF |
| `top_n` | `10` | 推荐篇数 |
| `sources` | `arxiv,openreview` | 数据源 |
| `exclude_known` | `true` | 推荐时排除知识库已有的论文 |

# 数据源

| 源 | 说明 | key |
|---|---|---|
| `arxiv` | 预印本，日期区间 + 分类过滤 | 不需要 |
| `openreview` | ICLR/NeurIPS/ICML 等计算机顶会 | 不需要 |

Semantic Scholar 作为补充源始终参与（引用数、高影响力论文），匿名访问可能 429；填 `semantic_scholar_api_key` 可缓解。

新增源只需在 `paper-daily/scripts/search_arxiv.py` 写一个 `fetcher(keywords, start_date, end_date, max_results)` 并用 `@register_source('名字','说明')` 装饰，评分/去重/输出自动复用。

# 工作流程

## 步骤1：检索

```bash
python scripts/search_arxiv.py \
  --output 08-daily/<日期>/search_result.json \
  --target-date "<日期>" \
  --exclude-known
```

参数省略时取配置值。常用覆盖：`--sources`、`--focus "kw1,kw2"`、`--days`、`--top-n`、`--skip-hot-papers`、`--output -`。

**去重（A 方案）**：脚本扫描三处判断论文是否已在知识库：

1. `papers_dir`（`01-raw/*.pdf`）
2. `notes_dir`（`03-notes/*/` 文件夹名）
3. `daily_dir` 历史（`08-daily/*/search_result.json` 里的推荐记录）

结果分两层：

- `all_papers`：**全量检索结果**，含 `already_known` 标记 → 日报的「本次检索列表」用它
- `top_papers`：**排除历史命中后的推荐**，按评分排序取 `top_n` → 日报的「前 3 篇 / 其余论文」用它

输出 JSON 关键字段：`sources_searched`、`source_counts`、`papers_by_source`、`total_unique`、`total_known`、`total_candidates`、`all_papers`、`top_papers`。每篇含 `paper_stem`（稳定主干）、`scores`、`matched_domain`、`already_known`。

## 步骤2：PDF 归档到 01-raw

```bash
python scripts/fetch_pdfs.py --papers-json 08-daily/<日期>/search_result.json --date "<日期>"
```

**硬性规则**：

- **目标文件已存在 → 立即停止**：不下载、不覆盖、不改名顶替，状态记 `exists`
- 只接受真 PDF（校验 `%PDF` 魔术字）；不合法就删掉临时文件、记 `failed`
- 先写 `.part` 再原子改名，不留半截文件
- 下载失败的论文**照常出现在日报里**并标注失败原因，不静默丢弃
- 顺手维护 `01-raw/index.md`（只记题目 + 落盘日期，已有条目跳过）

输出 JSON：`downloaded` / `skipped_existing` / `failed` / `archived[]`（每项含 `status`、`detail`、`path`）。

`--dry-run` 只报告不下载。

## 步骤3：写今日检索日报

路径 **`<daily_dir>/<日期>/<daily_note_name>.md`**，即 `08-daily/2026-09-16/今日检索.md`。
**用脚本落盘**（会一并复制 `search_result.json` 并写 `_index.json`）：

```bash
python scripts/write_note.py --date "<日期>" --papers-json 08-daily/<日期>/search_result.json --stdin-file note.md
```

`--dry-run` 只解析路径。

日报的固定布局也可以用渲染脚本生成，避免手工漏掉主题分组或全量列表：

```bash
python scripts/render_note.py \
  --date "<日期>" \
  --papers-json 08-daily/<日期>/search_result.json \
  --editorial-json <本次概览与前 3 篇分析>.json > note.md
python scripts/write_note.py \
  --date "<日期>" \
  --papers-json 08-daily/<日期>/search_result.json \
  --stdin-file note.md
```

`--editorial-json` 为可选项；结构如下，键名使用论文标题：

```json
{
  "overview": {
    "trend": "总体趋势",
    "hotspots": [{"title": "热点名称", "text": "热点说明"}]
  },
  "domain_summaries": {
    "研究大方向": {
      "summary": "该大方向的今日汇总",
      "subfields": {"具体细分领域": "该细分领域的汇总"}
    }
  },
  "top_papers": {
    "论文标题": {
      "title_zh": "题目中文翻译",
      "summary": "一句话总结",
      "abstract_zh": "中文摘要",
      "contributions": ["核心贡献"],
      "method": "方法思路",
      "result": "主要结果"
    }
  }
}
```

渲染脚本读取 `config.yaml` 的 `research_domains` 决定主题节顺序，
读取 `all_papers` 生成完整列表，读取 `top_papers` 生成前 3 篇和速览；
缺少编辑内容时会保留字段并以 `--` 或摘要作为占位，不会编造分析。

### 当前日报结构

执行时严格按以下顺序组织日报：

1. `今日概览`：先写用户在 `research_domains` 定义的大方向，再逐个给出该方向的论文数量、推荐数量、方向总结和具体细分领域汇总。细分领域由共享配置中对应主题的 `subdomains` 按命中关键词归类。
2. `今日推荐论文`：按全局评分取前 3 篇。标题格式为“英文题目 — 评分”，正文依次包含来源、主题（大方向）、细分领域、题目中文翻译、作者、英文摘要、中文摘要、一句话总结、核心贡献、方法思路、主要的实验成果、相关论文、访问链接。
3. `其余 7 篇推荐`：给出题目、评分、主题、细分领域、一句话总结和访问链接。
4. `今日落盘`：列出检索结果、PDF、日报和索引的真实状态。
5. `附录：本次检索列表`：放在全文最后，按大方向分节，包含题目、来源、评分、状态、研究大方向、具体细分领域、同脉络和访问链接。

相关论文必须先核验真实资产：优先链接 `03-notes/<论文主干>/精读.md`，其次链接真实存在的 `01-raw/<论文主干>.pdf`，再次链接实际存在且包含该论文的历史日报。三处都不存在时写“暂无”，不得根据标题相似度捏造文献或链接。

### 渲染器使用的编辑 JSON

`render_note.py` 负责固定布局和数据字段；人工撰写的概览、细分领域总结、标题翻译、中文摘要和前三篇分析放在 `--editorial-json` 中。除 `overview`、`domain_summaries` 和 `top_papers` 外，其他内容不应由渲染器自行编造。

旧版示例保留在下方，仅用于识别历史格式，不再作为执行标准。

### 历史日报结构（不再使用）

```markdown
## 今日概览

今日检索 N 篇（arXiv X / OpenReview Y），落盘 PDF 3 篇，按历史跳过 M 篇。

- **方向聚焦**：**{方向1}**、**{方向2}**、**{方向3}**
- **总体趋势**：{本周该领域在往哪走}
- **研究热点**：
  - **{热点1}**：…
  - **{热点2}**：…

## 本次检索列表

按 `config.yaml` 中 `research_domains` 的顺序分节；每个主题内按推荐评分从高到低排列。
未命中主题的论文统一放到最后的「未分类」节。`all_papers` 中的每篇论文必须且只能
出现一次，序号跨主题连续，便于回到全量检索结果核对。

### 随机梯度与优化理论（N 篇）

| # | 题目 | 来源 | 领域 | 评分 | 状态 | 同脉络 |
|---|---|---|---|---|---|---|
| 1 | … | arXiv | 随机梯度与优化理论 | 8.85 | 推荐 | Clipped Gradient Descent |
| 2 | … | arXiv | 随机梯度与优化理论 | 8.13 | 已推荐过 | -- |

### 大模型训练优化（N 篇）

| # | 题目 | 来源 | 领域 | 评分 | 状态 | 同脉络 |
|---|---|---|---|---|---|---|
| 3 | … | OpenReview | 大模型训练优化 | 8.02 | 推荐 | -- |

### 未分类（N 篇，如有）

| # | 题目 | 来源 | 领域 | 评分 | 状态 | 同脉络 |
|---|---|---|---|---|---|---|
| 4 | … | arXiv | -- | 7.57 | 推荐 | -- |

（用 `all_papers`，全量列出；`matched_domain` 为空时归入「未分类」。
`already_known` 为 true 的标「已在库」，并在状态里注明 `kb_source`：`pdf`=已有 PDF、
`note`=已精读、`daily`=历史推荐过。`related_papers` 取第一条标题填「同脉络」。
主题节可以省略空节，但不能省略包含论文的主题。）

## 前 3 篇

### 1. {论文标题} — 8.85

- **来源**：arXiv `2609.xxxxx` | [PDF](01-raw/{论文标题}.pdf) | [原文](https://arxiv.org/abs/…)
- **作者**：… | **领域**：…
- **一句话总结**：…
- **核心贡献**：
  - …
- **方法思路**：…
- **主要结果**：…
- **相关已有文献**（来自 `related_papers`；`--` 表示库里还没有同脉络的论文）：
  - [已有论文标题](03-notes/<已有论文标题>/精读.md) — 共同点：`shared_terms`；与本文的关系（继承 / 改进 / 互补 / 冲突）
  - …

## 其余论文速览

4. **{题目}** — 8.13 ｜ 同脉络：{`related_papers` 第一条标题或 `--`} ｜ {一句话}
5. **{题目}** — 7.57 ｜ 同脉络：`--` ｜ {一句话}

## 今日落盘文件

| 类型 | 路径 | 状态 |
|---|---|---|
| PDF | 01-raw/{标题}.pdf | 新下载 / 已存在跳过 / 下载失败 |
| 日报 | 08-daily/<日期>/今日检索.md | 本次生成 |
```

前 3 篇写**中等深度**（一句话 + 核心贡献 + 方法思路 + 主要结果），其余写一句话。

### 格式规则（paper-reading 使用标准 Markdown）

- **链接一律用标准 Markdown 相对路径**：`[PDF](01-raw/{标题}.pdf)`、`[精读](03-notes/{标题}/精读.md)`
- **只使用标准 Markdown**：图片用 `![说明](相对路径)`，链接用 `[文字](相对路径)`，不依赖特定笔记软件语法。
- 图片用标准语法 `![说明](相对路径)`
- 无数据用 `--`，不要用 `---`（会被当分隔线）
- 不伪造 arXiv 字段：OpenReview 来源没有 arXiv ID 就不写 arXiv 链接

## 步骤4（可选）：关键词索引

需要对已有论文笔记建「关键词 → 笔记」索引时：

```bash
python scripts/scan_existing_notes.py --workspace "$PAPER_WORKSPACE_PATH" --output existing_notes_index.json
python scripts/link_keywords.py --index existing_notes_index.json --input <笔记> --output <笔记>
```

会跳过 `index.md` 与 `images/` 下的图片索引。`link_keywords.py` 会把唯一匹配的关键词转换为相对于输出文件的标准 Markdown 链接。

# 重要规则

- **日期隔离**：日报只写在 `08-daily/<YYYY-MM-DD>/` 下
- **不碰用户资产**：`02-markdown` / `03-notes` / 已有 PDF 只读，绝不覆盖
- **已存在即停**：任何写入前先判存在，存在就跳过并如实报告
- **按主题组织日报**：`本次检索列表` 按 `research_domains` 顺序分节，每节内按 `score`
  从高到低；`top_papers` 的前 3 篇和其余速览仍沿用全局推荐分数顺序
- **相关论文必须可核验**：只引用实际存在于 `03-notes/<主干>/精读.md`、
  `01-raw/<主干>.pdf` 或历史日报中的论文；本地精读链接必须指向 `精读.md`，
  三处都不存在时写 `--`，不得根据标题相似度捏造文献
- **访问链接不得断链**：PDF 已落盘才使用 `01-raw` 相对链接，否则使用真实的远程
  `pdf_url`；原文链接使用记录中的 `url`
- **不需要大模型 API key**：脚本只做 HTTP 检索、评分与落盘；概览、总结、贡献点由当前 agent 撰写

# 依赖项

- Python 3.8+，`PyYAML`、`requests`（可选，缺失回退 urllib）
- 网络：`export.arxiv.org`、`api2.openreview.net`、`api.semanticscholar.org`
- workspace 中存在 `01-raw` / `02-markdown` / `03-notes` / `08-daily` 目录

# 常见问题

**搜不到新论文？** 可能都已在知识库里（看 `total_known`）。日报的「本次检索列表」仍会体现全量，推荐为空时如实说明。

**PDF 下载失败？** 常见于 OpenReview 无 `pdf` 字段、或网络受限。日报里标注失败原因即可，不必中止流程。

**arXiv 很慢？** 高峰期单次响应可能超过 20 秒，关键词多时整体更慢；可减小 `--days`、先只跑 `--sources openreview`，或调 `arxiv_request_timeout`。
