---
description: 按 AAAI、IJCAI、ICML、NeurIPS、ICLR、COLT、JMLR、TMLR、TPAMI、TNNLS 或 TAI 的投稿规范与有来源的写作档案起草论文、重组论证、撰写章节，生成可编译 LaTeX 工程。用于新内容起草；已有稿件润色使用 paper-polishing。
metadata:
  targets:
    - codex
name: paper-writing
---

# 顶会顶刊论文写作

本技能和 `paper-polishing` 分工对应 Nature 的写作/润色技能；两者共同使用相邻的 `../paper-writing-shared/` 支持目录。复制或安装时必须保留完整支持目录；无需全局 Nature 技能。

## 按需加载

1. 读取 [核心原则](../paper-writing-shared/core/evidence.md)、[写作流程](../paper-writing-shared/core/writing.md) 和 [资源索引](../paper-writing-shared/registry.json)。
2. 根据已知上下文识别任务、venue、目标年份、文章类型、章节、源语言和投稿阶段；简短告知选择，不把这一步变成常规审批。
3. 读取 `../paper-writing-shared/venues/<venue>.md`。必须先检查该档案的研究完成度，区分官方规则、语料观察、编辑建议，不把未完成语义审阅的样本当作已经验证的 venue 风格。
4. 只加载 [章节规则](../paper-writing-shared/sections/manuscript.md) 中相关部分；理论稿加载 [理论论证](../paper-writing-shared/core/paper-types.md)，实证或系统稿读取同一文件对应部分；中文材料加载 [语言转换](../paper-writing-shared/core/language.md)。
5. 生成或修改 TeX 时读取 [LaTeX 工程规则](../paper-writing-shared/core/latex.md)，以及资源索引所指向的目标版本 `spec.json`。

venue 是风格主轴；研究方向只影响术语、样本适用性与证据需求。不得自动把“计算机论文”限定为优化器、大模型或某一种方法。

## 完成任务

- 起草前建立简短的主张—证据—边界表及术语/符号表。整体重写先构建段落论证；局部章节直接复用已确认的内容，不重复收集整篇信息。
- 按官方约束和目标读者安排问题、方法、证据与局限；章节标题、贡献列表和叙事顺序依据稿件需要，不照搬所有样本的形式。
- 用户指定的 venue/年份优先；未知 venue 先请求必要信息，同时继续整理材料。没有年份时使用资源索引中已核验的最新可用版本，并说明它不是未来届次保证。
- 全文默认交付 LaTeX 工程、PDF、中文说明及尚待作者补充的事实。单章节优先接入用户工程；纯文本请求按用户要求交付文本。
- 缺少事实可继续写独立内容并显式标记待补。不得补造实验、基线、显著性、引用、创新性或证明。

## 工程入口

从支持目录执行（脚本相对位置不依赖机器上的绝对路径）：

```bash
python3 scripts/project.py list
python3 scripts/project.py init --venue icml --mode submission --output /absolute/path/new-project
python3 scripts/project.py compile --project /absolute/path/new-project
python3 scripts/project.py check --project /absolute/path/new-project
```

`init` 不覆盖已存在目标。目标模板不可用或年份不匹配时，不冒用相近会议/期刊模板；标明具体缺口，完成仍可完成的正文。

## 交付说明

优先展示稿件或 PDF；另列模板版本、投稿阶段、已完成验证及事实/格式缺口。编译、静态格式检查与官方投稿系统验证分别报告。未做视觉检查、未完成语料验证或尚有占位内容，不能称为“可直接投稿”。

本技能不处理 rebuttal、审稿模拟或绘图；引用核实和图件制作按实际任务使用专门工具，不自动扩展用户范围。
