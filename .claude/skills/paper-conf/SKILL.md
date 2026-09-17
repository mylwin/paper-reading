---
name: paper-conf
description: 通过 DBLP 与 Semantic Scholar 搜索 CVPR、ICCV、ECCV、ICLR、AAAI、NeurIPS、ICML 等顶会论文，经候选预筛和研究人员语义复排后生成年度推荐笔记。Use when the user asks for a conference/year paper list, top-conference recommendations, or papers filtered by venue.
---

# 目标

帮助用户搜索顶级学术会议（CVPR/ICCV/ECCV/ICLR/AAAI/NeurIPS/ICML/MICCAI/ACL/EMNLP）中与研究兴趣相关的论文，按年份筛选，生成推荐笔记到 Markdown workspace。

# 工作流程

## 工作流程概述

使用 DBLP API 搜索指定会议和年份的论文列表，用 Semantic Scholar 补充引用数与摘要，再按相关性、影响力信号、摘要证据充分度做候选预筛，并由研究人员视角进行语义复排，生成推荐笔记。

开始工作前读取 `../research-evaluation-rubric.md`。会议名和引用量只能帮助发现候选，不能代替对内容的判断。

## 配置说明

配置位于**本 skill 目录**下的 `config.yaml`：

```yaml
conf:
  # 覆盖关键词（留空 = 使用共享 config.yaml 的 research_domains 关键词）
  # keywords:
  #   - "large language model"

  excluded_keywords:      # 标题命中即丢弃
    - "workshop"

  default_year: 2025              # 默认年份（--year 可覆盖）
  default_conferences:            # 默认会议（--conferences 可覆盖）
    - ICLR
    - NeurIPS
    - ICML
    - CVPR
  top_n: 10                       # 返回论文数量
```

**关键词来源优先级**：

1. 本 skill `config.yaml` 的 `conf.keywords`
2. `.claude/skills/config.yaml` 的 `research_domains` 里所有关键词（默认走这条，与 `paper-daily` 同源）
3. 配置顶层 `keywords`（旧版 `conf-papers.yaml` 格式，向后兼容）

也就是说：**默认情况下顶会检索与每日推荐用同一套研究兴趣**。只有当顶会想用不同关键词时，才在 `conf.keywords` 里覆盖。修改研究兴趣请用 `paper-interests` skill。

命令行参数（`--year`、`--conferences`、`--top-n`）优先于配置。

## 步骤1：解析参数

1. **年份**（可选）：从用户输入提取，如 `2025`；未指定时用 `conf.default_year`
2. **会议**（可选）：如 `ICLR,CVPR`；未指定时用 `conf.default_conferences`
   - 注意双年会议：ICCV 偶数年、ECCV 奇数年可能无结果，正常跳过

## 步骤2：扫描已有笔记构建索引（可选）

复用 `paper-daily` 的扫描脚本，用于识别哪些论文已有详细笔记：

```bash
python "../paper-daily/scripts/scan_existing_notes.py" \
  --workspace "$PAPER_WORKSPACE_PATH" \
  --output existing_notes_index.json
```

（路径是相对本 skill 目录的写法；也可传绝对路径。脚本默认扫描 `notes_dir`（`03-notes`，自动递归 `YYYY-MM/` 月份目录），会跳过 `README.md`、`index.md` 与 `images/` 下的图片索引。）

## 步骤3：搜索顶会论文

```bash
python scripts/search_conf_papers.py \
  --year {年份} \
  --conferences "{会议列表，逗号分隔}" \
  --output conf_papers_result.json
```

`--config` 默认指向本 skill 的 `config.yaml`，通常不需要手动指定。

**脚本工作流**：

1. **DBLP 搜索**：调用 DBLP API 获取指定会议和年份的论文（toc 查询，ECCV/EMNLP/MICCAI 走 venue+year 备选查询）
2. **轻量过滤**：凭标题关键词匹配研究兴趣，大幅缩小范围
3. **S2 补充**：仅对过滤后的论文查询 Semantic Scholar，获取摘要、引用数与 arXiv ID
4. **三维预筛**：相关性 55% + 影响力信号 15% + 摘要证据充分度 30%，排序取 top N

**预筛说明**（与 `paper-daily` 的区别：无新近性维度，年份由用户指定）：

```yaml
研究优先级 =
  相关性: 55%          # 与当前研究问题的匹配程度
  影响力信号: 15%      # 引用的滞后信号，不代表正确性或质量
  摘要证据充分度: 30%  # 是否报告可核验的问题、方法、比较和结果
```

该分数不能跨年份直接比较，也不能解释成录用质量或学术价值。`novel`、`SOTA`、`first` 等宣传词不构成创新证据。新论文引用少不扣“质量分”，因为这里根本不做全文质量评分。

## 步骤4：读取筛选结果

读取 `conf_papers_result.json`：

- `year`、`conferences_searched`
- `total_found`（DBLP 搜索总数）、`total_filtered`（关键词过滤后）、`total_enriched`（S2 补充成功数）
- `top_papers`，每篇包含：
  - `title` / `authors` / `conference` / `year`
  - `dblp_url`、`arxiv_id`（如有）
  - `abstract`、`citationCount`、`influentialCitationCount`
  - `scores`（relevance / impact / evidence / recommendation；旧字段 popularity / quality 仅为兼容别名）
  - `matched_domain` / `matched_keywords`
  - `note_filename`：**生成 Markdown 相对链接时用这个字段**

## 步骤5：生成推荐笔记

### 5.1 笔记文件

- 路径：**`<daily_dir>/<运行日期>/顶会论文推荐.md`**
  - `daily_dir` 取 `.claude/skills/config.yaml` 的 `daily_dir`（默认 `08-daily`）
  - 例如 `08-daily/2026-09-18/顶会论文推荐.md`；`<运行日期>` 是本次检索的日期（`YYYY-MM-DD`）
  - 与 `今日检索.md` 同构放在日期目录下，便于复用 `write_note.py` 落盘，并让文件相对链接（`../../01-raw/YYYY-MM/…`）保持有效
  - 落盘命令：

```bash
python "../paper-daily/scripts/write_note.py" \
  --date "<运行日期>" \
  --filename 顶会论文推荐.md \
  --stdin-file conf_note.md
```
- frontmatter：

```yaml
---
keywords: [关键词1, 关键词2, ...]
tags: ["llm-generated", "conf-paper-recommend"]
---
```

**排版约定**：本工作区使用标准 Markdown（见工作区 `.AGENT.md`）。
链接使用**相对本笔记文件**的路径，笔记位于 `08-daily/<运行日期>/` 下，因此本地资产需要 `../../`：

```markdown
[PDF](../../01-raw/2026-09/<论文标题>.pdf)
[精读](../../03-notes/2026-09/<论文标题>/精读.md)
```

**不要**使用特定笔记软件的双中括号链接或图片语法；无数据用 `--` 而不是 `---`。

### 5.2 概览部分

```markdown
## {年份} 顶会论文推荐概览

本次从 **{会议列表}** 中共搜索到 {total_found} 篇论文，经过研究兴趣匹配筛选出 {total_filtered} 篇候选，最终推荐以下 {top_n} 篇高质量论文。

- **总体趋势**：{总结论文的整体研究趋势}

- **研究热点**：
  - **{热点1}**：{简要描述}
  - **{热点2}**：{简要描述}
  - **{热点3}**：{简要描述}

- **阅读建议**：{给出阅读顺序建议}
```

### 5.3 研究人员语义复排与论文列表

先对脚本返回候选逐篇读题目与摘要，再按“与当前问题的直接性、相对已有知识的新增信息、可核验性、阅读成本与组合多样性”调整顺序。至少包含一篇直接竞争工作或最强近邻，不要让高引用综述或热门大模型论文挤占全部名额。

只有摘要时，贡献与结果一律标“摘要声称”，并列出待全文核验项。复排时保留 `screening_rank` 和原分数，给最终候选补 `semantic_rank`，不要篡改脚本分数。最终列表按复排后的研究优先级组织，而不是机械照抄引用排序。

```markdown
### {论文标题}
- **作者**：[作者列表]
- **机构**：[机构名称] 或 --
- **会议**：{CVPR/ICLR/...} {年份}
- **研究优先级**：{recommendation}/10（元数据预筛，不是质量分）
- **引用信号**：{citationCount} (influential: {influentialCitationCount})
- **链接**：[DBLP](链接) | [arXiv](链接) | [PDF](../../01-raw/YYYY-MM/{论文标题}.pdf)
- **笔记**：[精读](../../03-notes/YYYY-MM/{论文标题}/精读.md) 或 --

**摘要声称的核心贡献**：[一句话概括；待全文核验]

**核心贡献/观点**：
- [贡献点1]
- [贡献点2]
- [贡献点3]

**关键结果**：[从摘要中提取的最重要结果]

**选择理由**：[它对当前研究问题提供什么新增信息]

**待核验**：[最关键的理论/实验/复现问题]

**阅读决策**：精读 / 选读 / 跟踪

---
```

英文报告（`language: "en"`）用对应的英文标签（Authors / Conference / Citations / Links / Notes / One-line Summary / Core Contributions / Key Results）。

**链接规则**：
- 有 arXiv ID：给 arXiv 和 PDF 链接
- 无 arXiv ID：只给 DBLP（或 DOI）链接，并标注"无 arXiv 版本"
- 有 DOI：额外给 DOI 链接
- 本地 PDF / 精读笔记用**相对本笔记文件**的标准 Markdown 相对路径指向 `01-raw/<YYYY-MM>/`、`03-notes/<YYYY-MM>/`

**格式规则**：
- 用标准 Markdown 相对链接（`[文字](相对路径)`），不依赖特定笔记软件语法。
- 图片用标准语法 `![说明](相对路径)`
- 无数据用 `--`，不要用 `---`（会被解析成分隔线）

### 5.4 前 3 篇特殊处理

对语义复排后优先级最高的 3 篇：

1. **查重**：按论文主干（`paper_stem` / `<论文标题>`）在 `papers_dir`（含 `YYYY-MM/`）、`notes_dir`（含 `YYYY-MM/`）、`08-daily` 历史中匹配。已有笔记则引用，**不重复生成**。
2. **有 arXiv ID 且未归档**：用 `paper-daily` 的归档脚本下载，**不要手写目标路径**（脚本会按入库月份落到 `01-raw/<YYYY-MM>/` 并刷新索引）：

```bash
python "../paper-daily/scripts/fetch_pdfs.py" \
  --papers-json <本次检索结果>.json --date "<运行日期>"
```

   随后在列表里给出 `[PDF](../../01-raw/<YYYY-MM>/<论文标题>.pdf)` 相对链接。
3. **无 arXiv ID 且未归档**：标注"无 arXiv 版本，无法自动获取 PDF 与图片"，只给 DBLP/DOI 链接供手动查阅，**不伪造 arXiv 字段**。

精读笔记（`03-notes/<YYYY-MM>/<论文标题>/精读.md`）由用户自己的流程生成，本 skill 不代做。

写完后刷新索引与状态清单：

```bash
python "../paper-daily/scripts/sync_indexes.py"
python "../paper-daily/scripts/sync_indexes.py" --check-links
```

# 重要规则

- **年份必填**：命令行 `--year` 或配置 `conf.default_year`，都没有则报错退出
- **预筛后复排**：先按 `scores.recommendation` 形成候选池，再按研究问题、新增信息和证据可核验性做语义复排
- **前 3 篇**：有 arXiv ID 的才做图片与详细报告；其余论文只写基本信息
- **不需要大模型 API key**：脚本只做 HTTP 检索、补充与候选预筛；概览、研究判断与复排理由由当前 agent 撰写
- **DBLP 限流**：脚本内置重试；一次搜多个会议会明显变慢，建议先指定 1-2 个会议

# 依赖项

- Python 3.8+，`PyYAML`、`requests`（缺失时回退 urllib）
- 网络：`dblp.org`、`api.semanticscholar.org`
- 与 `paper-daily` 同目录层级（脚本会从 `../paper-daily/scripts` 复用评分函数与配置解析）

# 常见问题

**某会议没有结果？**
双年会议（ICCV/ECCV）在非举办年份无数据；另外 DBLP 的 toc 命名对 ECCV/EMNLP/MICCAI 不稳定，脚本已走备选查询，仍可能失败。

**S2 补充全失败？**
匿名访问容易 429。在 `.claude/skills/config.yaml` 里填 `semantic_scholar_api_key`，或加 `--skip-enrichment` 先只看 DBLP 标题与研究相关性预筛。

**关键词太少导致过滤后没有论文？**
说明 `research_domains` 的英文关键词偏窄。用 `paper-interests` 补充该方向的关键词。
