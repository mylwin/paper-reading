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

logger = logging.getLogger(__name__)


def get_output_dir(cli_output_dir=None):
    """获取笔记输出根目录，默认当前目录下的 paper-notes/"""
    if cli_output_dir:
        return cli_output_dir
    env_path = os.environ.get('PAPER_NOTES_DIR')
    if env_path:
        return env_path
    return os.path.join(os.getcwd(), "paper-notes")


def generate_note_content(paper_id, title, authors, domain, date, language="zh"):
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
paper_id: "{paper_id}"
title: "{title}"
authors: "{authors}"
domain: "{domain}"
tags: [{tags_yaml}]
quality_score: "[SCORE]/10"
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

### 适用场景
- [场景1]
- [场景2]

## 与相关论文对比

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

## 未来工作建议

1. [作者建议1]
2. [作者建议2]
3. [基于分析的延伸建议]

## 我的综合评价

| 维度 | 分数 | 理由 |
|---|---|---|
| 创新性 | [X]/10 | [理由] |
| 技术质量 | [X]/10 | [理由] |
| 实验充分性 | [X]/10 | [理由] |
| 写作质量 | [X]/10 | [理由] |
| 实用性 | [X]/10 | [理由] |

**总体评分**：[X.X]/10 — [评分理由简述]

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
paper_id: "{paper_id}"
title: "{title}"
authors: "{authors}"
domain: "{domain}"
tags: [{tags_yaml}]
quality_score: "[SCORE]/10"
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

### Applicable Scenarios
- [Scenario 1]
- [Scenario 2]

## Comparison with Related Work

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

## Future Work

1. [Author's suggestion 1]
2. [Author's suggestion 2]
3. [Extension based on this analysis]

## Assessment

| Dimension | Score | Rationale |
|---|---|---|
| Innovation | [X]/10 | [Rationale] |
| Technical Quality | [X]/10 | [Rationale] |
| Experimental Thoroughness | [X]/10 | [Rationale] |
| Writing Quality | [X]/10 | [Rationale] |
| Practicality | [X]/10 | [Rationale] |

**Overall Score**: [X.X]/10 — [Brief rationale]

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
    parser.add_argument('--domain', type=str, default='Other', help='论文领域，用于本地分目录 / Paper domain, used for local folder grouping')
    parser.add_argument('--output-dir', type=str, default=None, help='笔记输出根目录，默认 ./paper-notes / Output root dir, defaults to ./paper-notes')
    parser.add_argument('--language', type=str, default='zh', choices=['zh', 'en'], help='语言 / Language: zh (中文) or en (English)')
    args = parser.parse_args()

    output_root = get_output_dir(args.output_dir)
    date = datetime.now().strftime("%Y-%m-%d")

    # 清理文件名中的非法字符
    paper_title_safe = re.sub(r'[ /\\:*?"<>|]+', '_', args.title).strip('_')

    # 校验域名，防止路径穿越
    domain = args.domain.strip('/\\').replace('..', '')
    if not domain:
        domain = 'Other'

    note_dir = os.path.join(output_root, domain)
    images_dir = os.path.join(note_dir, paper_title_safe, "images")
    os.makedirs(note_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    note_path = os.path.join(note_dir, f"{paper_title_safe}.md")
    content = generate_note_content(args.paper_id, args.title, args.authors, domain, date, args.language)

    try:
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except IOError as e:
        logger.error("写入笔记失败: %s", e)
        sys.exit(1)

    if args.language == 'zh':
        print(f"笔记已生成: {note_path}")
        print(f"图片目录: {images_dir}（把提取到的论文插图放到这里，命名为 fig1.png, fig2.png ...）")
        print("请手动编辑笔记内容，替换占位符为实际分析结果")
    else:
        print(f"Note generated: {note_path}")
        print(f"Images dir: {images_dir} (place extracted figures here as fig1.png, fig2.png, ...)")
        print("Please manually edit the note content and replace placeholders with actual analysis.")


if __name__ == '__main__':
    main()
