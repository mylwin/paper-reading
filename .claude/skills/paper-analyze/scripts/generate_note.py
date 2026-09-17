#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文分析笔记生成脚本 - 生成标准 Markdown（不依赖任何笔记软件）
支持中英文报告生成

与旧版本的主要区别：
- 不依赖任何工作区路径或环境变量，改用普通的 --output-dir
- 相关论文引用改用纯文本加粗，而不是特定笔记软件的链接语法
- 图片引用改用标准 Markdown 语法 ![alt](images/xxx.png)，而不是 ![[file|800]]
- 提示/警告改用标准引用块 + 加粗，而不是 > [!tip] 这类 callout 语法
- 公式统一使用 $...$ / $$...$$，块级公式独立成行并前后空行，兼容更多渲染器
"""

import sys
import os
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def find_workspace(cli_workspace=None):
    """定位论文工作区根目录：--workspace → $PAPER_WORKSPACE_PATH → 向上查找标记目录。"""
    candidates = []
    if cli_workspace:
        candidates.append(Path(cli_workspace).expanduser())
    env_path = os.environ.get('PAPER_WORKSPACE_PATH')
    if env_path:
        candidates.append(Path(env_path).expanduser())
    here = Path.cwd().resolve()
    candidates.extend([here, *here.parents])
    markers = ('01-raw', '02-markdown', '03-notes')
    for candidate in candidates:
        try:
            if sum((candidate / marker).is_dir() for marker in markers) >= 2:
                return candidate.resolve()
        except OSError:
            continue
    return here


def read_workspace_config_value(workspace, key):
    """从 `<workspace>/.claude/skills/config.yaml` 读取单个目录配置（yaml 缺失时用正则兜底）。"""
    config_path = Path(workspace) / '.claude' / 'skills' / 'config.yaml'
    if not config_path.is_file():
        return None
    try:
        text = config_path.read_text(encoding='utf-8-sig')
    except OSError:
        return None
    try:
        import yaml
        data = yaml.safe_load(text) or {}
        value = data.get(key)
        if value:
            return str(value).strip()
    except Exception:
        pass
    match = re.search(r'^%s:\s*["\']?([^"\'\n#]+)' % re.escape(key), text, re.MULTILINE)
    return match.group(1).strip() if match else None


def resolve_notes_root(workspace, output_dir=None):
    """精读笔记根目录：--output-dir 覆盖 → 配置 notes_dir（默认 03-notes）。"""
    if output_dir:
        path = Path(output_dir).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path)
    configured = read_workspace_config_value(workspace, 'notes_dir') or '03-notes'
    path = Path(configured)
    return path if path.is_absolute() else Path(workspace) / path


def normalize_stem(text):
    return re.sub(r'[^a-z0-9\u4e00-\u9fff]+', '', str(text or '').lower())


def month_of(value):
    """把日期字符串归一为 `YYYY-MM`。"""
    match = re.match(r'^(\d{4})[-/.](\d{1,2})', str(value or '').strip())
    if match:
        month = int(match.group(2))
        return '%04d-%02d' % (int(match.group(1)), month) if 1 <= month <= 12 else ''
    return ''


def infer_archive_month(workspace, title, fallback_date):
    """推断论文的入库月份：01-raw 中同名 PDF 所在月份 → 兜底用给定日期所在月。

    月份口径见工作区 .AGENT.md：论文首次进入 01-raw 的日期所在月。
    """
    papers_dir_name = read_workspace_config_value(workspace, 'papers_dir') or '01-raw'
    papers_dir = Path(papers_dir_name)
    if not papers_dir.is_absolute():
        papers_dir = Path(workspace) / papers_dir
    target = normalize_stem(title)
    if papers_dir.is_dir() and target:
        month_dirs = sorted(
            (p for p in papers_dir.iterdir()
             if p.is_dir() and re.match(r'^\d{4}-(0[1-9]|1[0-2])$', p.name)),
            key=lambda p: p.name, reverse=True)
        for month_dir in month_dirs:
            for pdf in sorted(month_dir.glob('*.pdf')):
                if normalize_stem(pdf.stem) == target:
                    return month_dir.name
    return month_of(fallback_date) or datetime.now().strftime('%Y-%m')


def generate_note_content(paper_id, title, authors, domain, date, language="zh", archive_month=""):
    """生成笔记的标准 Markdown 内容（含 frontmatter，可选，仅作普通元数据）"""

    if language == "zh":
        domain_tags = {
            "LLM": ["大模型", "LLM"],
            "Multimodal": ["多模态", "Vision-Language"],
            "Agent": ["智能体", "Agent"],
            "RL": ["强化学习", "RL"],
        }
        tags = ["论文笔记"] + domain_tags.get(domain, [domain])
        tags_yaml = ", ".join(f'"{t}"' for t in tags)

        return f'''---
date: "{date}"
archive_month: "{archive_month}"
paper_id: "{paper_id}"
title: "{title}"
authors: "{authors}"
domain: "{domain}"
tags: [{tags_yaml}]
research_value_score: "[SCORE]/10"
assessment_confidence: "[高/中/低]"
status: "analyzed"
---

# {title}

## 核心信息
- **论文ID**：{paper_id}
- **作者**：{authors}
- **机构**：[从作者信息或论文正文推断]
- **发布时间**：{date}
- **会议/期刊**：[如可从分类推断]
- **链接**：[arXiv](https://arxiv.org/abs/{paper_id}) | [PDF](https://arxiv.org/pdf/{paper_id})

## 摘要翻译

### 英文摘要
[论文英文摘要原文]

### 中文翻译
[流畅准确的中文翻译，保持学术术语准确]

### 核心要点提炼
- **研究背景**：[领域现状与存在的问题]
- **研究动机**：[为什么要做这项研究]
- **核心方法**：[一句话概括]
- **主要结果**：[最重要的实验结果]
- **研究意义**：[对领域的贡献]

## 研究背景与动机

### 领域现状
[该研究领域当前的发展状况]

### 现有方法的局限性
[现有方法存在的问题]

### 研究动机
[为什么需要这项研究]

## 研究问题

[清晰、准确地描述论文要解决的核心问题]

## 方法概述

### 核心思想
[用通俗易懂的语言解释方法的核心思想]

### 方法框架

架构图优先使用论文原图（提取后保存在 `images/` 目录，用标准 Markdown 语法插入）：

![图1：整体架构说明](images/fig1.png)

> 图1：[架构中各部分含义及关系的说明]

如果论文没有合适的架构图，可用 mermaid 代码块画流程图：

```mermaid
flowchart LR
    A[输入] --> B[模块1]
    B --> C[模块2]
    C --> D[输出]
```

### 数学公式
- 行内公式使用 `$...$`，例如：目标函数为 $L(\\theta)$。`$` 内侧不留空格，公式与前后中文之间各留一个半角空格，`$...$` 必须同一行闭合，禁用 `\\( ... \\)`。
- 表格单元格内的公式禁止裸 `|`，条件符号用 `\\mid`，范数写 `\\Vert x \\Vert`。
- 只用基础 LaTeX 命令；禁用 `\\tag`、`\\big` 系列、`\\lesssim`、间距命令、`\\boxed`、`\\underbrace`、`\\|`（必须写成 `\\Vert`）。
- 块级公式使用 `$$...$$`，独立成行，前后各空一行；公式编号写在正文文字里，不用 `\\tag{{}}`：

$$
\\theta^* = \\arg\\min_\\theta L(\\theta)
$$

### 各模块详细说明

**模块1：[模块名称]**
- **功能**：[主要功能]
- **输入 / 输出**：[输入数据] → [输出数据]
- **处理流程**：
  1. [步骤1]
  2. [步骤2]
- **关键技术**：[使用的关键技术或算法]

### 关键创新

1. [创新点1] — [为什么重要]
2. [创新点2] — [为什么重要]
3. [创新点3] — [为什么重要]

## 实验结果

### 数据集

| 数据集 | 规模 | 特点 |
|---|---|---|
| [数据集1] | ... | ... |
| [数据集2] | ... | ... |

### 实验设置
- **基线方法**：[列出对比方法]
- **评估指标**：[列出指标]
- **实验环境**：[硬件、超参数]

### 主要结果

| 方法 | 指标1 | 指标2 |
|---|---|---|
| 基线1 | ... | ... |
| 基线2 | ... | ... |
| **本文方法** | **...** | **...** |

[主要结果图，若有]

![图2：实验结果](images/fig2.png)

### 结果分析
[对实验结果的详细分析]

## 深度分析

### 研究价值
- **理论贡献**：[理论上的贡献]
- **实际应用**：[实际应用价值]
- **领域影响**：[对研究领域的潜在影响]

### 优势
- [优势1]
- [优势2]
- [优势3]

### 局限性
- [局限1]
- [局限2]
- [局限3]

### 致命问题与非致命局限
- **致命问题**：[若成立会推翻核心主张的问题；没有明确证据则写 --]
- **非致命局限**：[限制外推范围但不推翻核心结论的问题]

### 适用场景
- [场景1]
- [场景2]

## 主张-证据-限制矩阵

| 主要主张 | 证据位置 | 证据是否充分 | 替代解释/风险 | 置信度 |
|---|---|---|---|---|
| [主张1] | [定理/表/图/附录] | [判断] | [风险] | [高/中/低] |

## 与最接近工作对比

### **[相关论文1标题]**（[作者], [年份]）
- **关系类型**：[改进 / 扩展 / 对比 / 跟随]
- **差异**：[本文方法的不同之处]
- **改进**：[相比该论文的改进点]
- **性能对比**：[如果可用]

### **[相关论文2标题]**（[作者], [年份]）
[类似格式]

### **[相关论文3标题]**（[作者], [年份]）
[类似格式]

## 技术路线定位

本文属于[技术路线]，主要关注[具体子方向]。

## 可证伪的后续研究机会

| 假设 | 最小验证 | 指标 | 资源成本 | 停止条件 |
|---|---|---|---|---|
| [具体假设] | [对照/证明/实验] | [判据] | [低/中/高] | [何时放弃] |

## 我的综合评价

| 维度 | 分数（1-5/N/A） | 证据位置 | 理由 | 置信度 |
|---|---:|---|---|---|
| 问题重要性与界定 | [X] | [...] | [...] | [...] |
| 原创性与净增量 | [X] | [...] | [...] | [...] |
| 技术正确性与严谨性 | [X] | [...] | [...] | [...] |
| 证据强度与替代解释 | [X] | [...] | [...] | [...] |
| 可复现性与透明度 | [X] | [...] | [...] | [...] |
| 外推边界与稳健性 | [X] | [...] | [...] | [...] |
| 研究生成力 | [X] | [...] | [...] | [...] |
| 当前课题契合度 | [X] | [...] | [...] | [...] |

**学术价值分**：[X.X]/10 — [说明权重与证据]

**当前课题优先级**：[必读/精读/选读/跟踪/略读] — [具体用途]

**评价置信度**：[高/中/低]；**待核验**：[缺失信息]

### 突出亮点
- [亮点1]
- [亮点2]
- [亮点3]

### 可借鉴点
- [可以学习借鉴的技术或思路]

### 批判性思考
- [潜在问题]
- [可改进之处]
- [质疑点]

> **关键启示：** [论文最重要的启示，一句话总结核心思想]

> **注意事项：**
> - [注意事项1]
> - [注意事项2]

## 我的笔记

[用户阅读后手动补充的内容]

## 相关论文
- **[相关论文1标题]**（[作者], [年份]） — [关系描述]
- **[相关论文2标题]**（[作者], [年份]） — [关系描述]

## 外部资源
- [论文链接]
- [代码链接]（如果有）
- [项目主页]（如果有）
'''
    else:
        # English template
        domain_tags_en = {
            "LLM": ["LLM", "Large Language Model"],
            "Multimodal": ["Multimodal", "Vision-Language"],
            "Agent": ["Agent", "Multi-Agent"],
            "RL": ["Reinforcement Learning", "RL"],
        }
        tags = ["paper-notes"] + domain_tags_en.get(domain, [domain])
        tags_yaml = ", ".join(f'"{t}"' for t in tags)

        return f'''---
date: "{date}"
archive_month: "{archive_month}"
paper_id: "{paper_id}"
title: "{title}"
authors: "{authors}"
domain: "{domain}"
tags: [{tags_yaml}]
research_value_score: "[SCORE]/10"
assessment_confidence: "[high/medium/low]"
status: "analyzed"
---

# {title}

## Core Information
- **Paper ID**: {paper_id}
- **Authors**: {authors}
- **Affiliation**: [Infer from authors or paper text]
- **Publication Date**: {date}
- **Conference/Journal**: [Infer from categories, if possible]
- **Links**: [arXiv](https://arxiv.org/abs/{paper_id}) | [PDF](https://arxiv.org/pdf/{paper_id})

## Abstract & Translation

### Original Abstract
[Paper's abstract]

### Key Takeaways
- **Background**: [State of the field, existing gaps]
- **Motivation**: [Why this research matters]
- **Core Method**: [One-sentence summary]
- **Main Results**: [Most important findings]
- **Significance**: [Contribution to the field]

## Research Background & Motivation

### Current State of the Field
[Description of the field's current development]

### Limitations of Existing Methods
[Problems with existing approaches]

### Motivation
[Why this research is needed]

## Research Problem

[Clear, precise description of the core problem this paper addresses]

## Method Overview

### Core Idea
[Explain the core idea in plain language]

### Method Framework

Prefer the paper's original figure when available (extracted and saved under `images/`, referenced with standard Markdown syntax):

![Figure 1: overall architecture](images/fig1.png)

> Figure 1: [Explanation of each component and how they relate]

If the paper has no suitable architecture figure, use a mermaid diagram instead:

```mermaid
flowchart LR
    A[Input] --> B[Module 1]
    B --> C[Module 2]
    C --> D[Output]
```

### Mathematical Formulas
- Inline formulas: `$...$`, e.g. the objective is $L(\\theta)$. No spaces inside the `$` delimiters, keep `$...$` on a single line, never use `\\( ... \\)`.
- No bare `|` inside table-cell formulas; use `\\mid` for conditionals and `\\Vert x \\Vert` for norms.
- Stick to basic LaTeX commands; never use `\\tag`, `\\big` variants, `\\lesssim`, spacing commands, `\\boxed`, `\\underbrace`, or `\\|` (write `\\Vert` instead).
- Block formulas: `$$...$$`, on their own line with a blank line before and after; put equation numbers in the surrounding text, not `\\tag{{}}`:

$$
\\theta^* = \\arg\\min_\\theta L(\\theta)
$$

### Module Details

**Module 1: [Name]**
- **Function**: [Main purpose]
- **Input / Output**: [Input] → [Output]
- **Pipeline**:
  1. [Step 1]
  2. [Step 2]
- **Key Techniques**: [Algorithms/techniques used]

### Key Innovations

1. [Innovation 1] — [Why it matters]
2. [Innovation 2] — [Why it matters]
3. [Innovation 3] — [Why it matters]

## Experimental Results

### Datasets

| Dataset | Scale | Characteristics |
|---|---|---|
| [Dataset 1] | ... | ... |
| [Dataset 2] | ... | ... |

### Experimental Settings
- **Baselines**: [List of comparison methods]
- **Metrics**: [Evaluation metrics]
- **Environment**: [Hardware, hyperparameters]

### Main Results

| Method | Metric 1 | Metric 2 |
|---|---|---|
| Baseline 1 | ... | ... |
| Baseline 2 | ... | ... |
| **This paper** | **...** | **...** |

[Main results figure, if available]

![Figure 2: experimental results](images/fig2.png)

### Analysis
[Detailed analysis of the results]

## Deep Analysis

### Research Value
- **Theoretical Contribution**: [...]
- **Practical Applications**: [...]
- **Field Impact**: [...]

### Advantages
- [Advantage 1]
- [Advantage 2]
- [Advantage 3]

### Limitations
- [Limitation 1]
- [Limitation 2]
- [Limitation 3]

### Fatal Issues vs Non-fatal Limitations
- **Fatal issues**: [Issues that would invalidate the central claim, or --]
- **Non-fatal limitations**: [Issues that bound generalization]

### Applicable Scenarios
- [Scenario 1]
- [Scenario 2]

## Claim-Evidence-Limitation Matrix

| Main claim | Evidence location | Sufficiency | Alternative explanation/risk | Confidence |
|---|---|---|---|---|
| [Claim 1] | [Theorem/table/figure/appendix] | [Assessment] | [Risk] | [High/medium/low] |

## Comparison with Closest Work

### **[Related Paper 1 Title]** ([Authors], [Year])
- **Relationship**: [Improves / Extends / Compares / Follows]
- **Difference**: [How this method differs]
- **Improvement**: [Improvements over this paper]
- **Performance Comparison**: [If available]

### **[Related Paper 2 Title]** ([Authors], [Year])
[Similar format]

### **[Related Paper 3 Title]** ([Authors], [Year])
[Similar format]

## Technical Roadmap

This paper belongs to [technical track], focusing on [specific sub-direction].

## Falsifiable Research Opportunities

| Hypothesis | Minimum validation | Metric | Resource cost | Stop condition |
|---|---|---|---|---|
| [Specific hypothesis] | [Control/proof/experiment] | [Criterion] | [Low/medium/high] | [When to stop] |

## Assessment

| Dimension | Score (1-5/N/A) | Evidence location | Rationale | Confidence |
|---|---:|---|---|---|
| Problem importance and framing | [X] | [...] | [...] | [...] |
| Originality and net contribution | [X] | [...] | [...] | [...] |
| Technical soundness and rigor | [X] | [...] | [...] | [...] |
| Evidence and alternative explanations | [X] | [...] | [...] | [...] |
| Reproducibility and transparency | [X] | [...] | [...] | [...] |
| Generalization boundaries and robustness | [X] | [...] | [...] | [...] |
| Research generativity | [X] | [...] | [...] | [...] |
| Fit to current research agenda | [X] | [...] | [...] | [...] |

**Academic value**: [X.X]/10 — [Weights and evidence]

**Current-project priority**: [Must read/deep read/selective read/watch/skip] — [Specific use]

**Assessment confidence**: [High/medium/low]; **To verify**: [Missing information]

### Highlights
- [Highlight 1]
- [Highlight 2]
- [Highlight 3]

### Learnings
- [Techniques/ideas worth learning from]

### Critical Thinking
- [Potential issues]
- [Areas for improvement]
- [Points of contention]

> **Key Takeaway:** [The single most important insight, in one sentence]

> **Caveats:**
> - [Caveat 1]
> - [Caveat 2]

## My Notes

[Content to be added manually after reading]

## Related Papers
- **[Related Paper 1 Title]** ([Authors], [Year]) — [Relationship]
- **[Related Paper 2 Title]** ([Authors], [Year]) — [Relationship]

## External Resources
- [Paper link]
- [Code link] (if available)
- [Project homepage] (if available)
'''


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description='生成论文分析笔记（标准 Markdown） / Generate a paper analysis note (standard Markdown)')
    parser.add_argument('--paper-id', type=str, default='[PAPER_ID]', help='论文 arXiv ID / Paper arXiv ID')
    parser.add_argument('--title', type=str, default='[论文标题]', help='论文标题 / Paper title')
    parser.add_argument('--authors', type=str, default='[Authors]', help='论文作者 / Paper authors')
    parser.add_argument('--domain', type=str, default='Other', help='论文领域，仅写入 frontmatter / Paper domain (frontmatter only)')
    parser.add_argument('--workspace', type=str, default=None,
                        help='论文工作区根目录（默认自动向上识别，也可用 PAPER_WORKSPACE_PATH）')
    parser.add_argument('--notes-dir', type=str, default=None,
                        help='精读笔记根目录（默认取工作区配置 notes_dir = 03-notes）')
    parser.add_argument('--month', type=str, default=None,
                        help='入库月份 YYYY-MM（默认按 01-raw 中同名 PDF 所在月份推断）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='覆盖笔记根目录（等价于 --notes-dir，供独立使用）')
    parser.add_argument('--language', type=str, default='zh', choices=['zh', 'en'], help='语言 / Language: zh (中文) or en (English)')
    parser.add_argument('--force', action='store_true',
                        help='已存在 精读.md 时覆盖（默认不覆盖用户已有笔记）')
    args = parser.parse_args()

    workspace = find_workspace(args.workspace)
    output_root = resolve_notes_root(workspace, args.output_dir or args.notes_dir)
    date = datetime.now().strftime("%Y-%m-%d")

    # 论文主干：与 01-raw / 02-markdown 等其他阶段的命名规则一致
    paper_title_safe = re.sub(r'[ /\\:*?"<>|\t]+', '_', args.title)
    paper_title_safe = re.sub(r'[,\u3001;；]+', '_', paper_title_safe)
    paper_title_safe = re.sub(r'[\s\-]+', '_', paper_title_safe)
    paper_title_safe = re.sub(r'_{2,}', '_', paper_title_safe).strip(' ._')

    # 校验域名，防止路径穿越（仅用于 frontmatter）
    domain = args.domain.strip('/\\').replace('..', '')
    if not domain:
        domain = 'Other'

    month = (args.month or '').strip() or infer_archive_month(workspace, args.title, date)

    # 落盘位置：<notes_dir>/<YYYY-MM>/<论文主干>/{精读.md, images/}
    note_dir = os.path.join(str(output_root), month, paper_title_safe)
    images_dir = os.path.join(note_dir, "images")
    note_path = os.path.join(note_dir, "精读.md")

    if os.path.exists(note_path) and not args.force:
        logger.info("已存在精读笔记，未覆盖：%s", note_path)
        print(f"已存在精读笔记，未覆盖：{note_path}（如需重写请加 --force）")
        print(f"图片目录：{images_dir}")
        return

    os.makedirs(images_dir, exist_ok=True)
    content = generate_note_content(args.paper_id, args.title, args.authors, domain, date,
                                    args.language, archive_month=month)

    try:
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except IOError as e:
        logger.error("写入笔记失败: %s", e)
        sys.exit(1)

    if args.language == 'zh':
        print(f"工作区：{workspace}")
        print(f"入库月份：{month}")
        print(f"笔记已生成: {note_path}")
        print(f"图片目录: {images_dir}（把提取到的论文插图放到这里，命名为 fig1.png, fig2.png ...）")
        print("请手动编辑笔记内容，替换占位符为实际分析结果")
    else:
        print(f"Workspace: {workspace}")
        print(f"Archive month: {month}")
        print(f"Note generated: {note_path}")
        print(f"Images dir: {images_dir} (place extracted figures here as fig1.png, fig2.png, ...)")
        print("Please manually edit the note content and replace placeholders with actual analysis.")


if __name__ == '__main__':
    main()
