---
description: 按具体顶会顶刊的写作档案与官方 LaTeX 规范润色、翻译、压缩或优化已有论文，支持全文结构调整、投稿 venue 转换和排版维护，同时保留事实、公式、引用与工程结构。新章节起草使用 paper-writing。
metadata:
  targets:
    - codex
name: paper-polishing
---

# 顶会顶刊论文润色与优化

与 `paper-writing` 共用相邻的 `../paper-writing-shared/`；安装时保留支持目录，无需依赖全局 Nature 技能。

## 按需加载

1. 读取 [核心原则](../paper-writing-shared/core/evidence.md) 和 [润色流程](../paper-writing-shared/core/polishing.md)。
2. 从当前稿件和上下文识别 task、venue、年份、文章类型、章节、语言、投稿阶段和修改强度。没有特定目标的局部润色可直接应用通用原则，不强迫用户先选会议。
3. 指定 venue 时读取 [资源索引](../paper-writing-shared/registry.json) 和 `../paper-writing-shared/venues/<venue>.md`；检查实际语义审阅/留出验证状态，不以待审语料推断刊物惯例。
4. 按需要读取 [章节规则](../paper-writing-shared/sections/manuscript.md)、[论文类型](../paper-writing-shared/core/paper-types.md)、[语言规则](../paper-writing-shared/core/language.md)。
5. 涉及 TeX、页数、模板迁移或浮动体时读取 [LaTeX 规则](../paper-writing-shared/core/latex.md) 及对应版本 `spec.json`。

## 修改策略

- 默认按用户请求范围润色，保持学术含义。语言润色、论证重组、篇幅压缩和模板迁移是不同模式；不用一次轻量润色自动重写论文。
- 诊断顺序：章节任务 → 主张证据边界 → 证据位置 → 段落必要性 → 术语一致 → 句子表达。
- 结构问题需补充证据才能修复时，指出缺口，不能用流畅文字掩盖缺失。把实质性建议与已执行的语言修改分别列出。
- 保留公式、定理假设、数据、引用键、交叉引用、自定义宏及文件组织。必须改动这些内容时明确解释，与用户已有授权范围对应。
- 压缩首先删除重复、冗余背景和重复报数；把附录可承担的细节移出正文时，保留评估贡献必须看到的证据和依赖。
- venue 转换调整读者假设、信息顺序和格式；不会借此扩大结论、增加实验或把理论结果改写成未经验证的实证结果。

## LaTeX 处理

沿用用户工程，先检查入口、宏、参考文献及编译状态；在新输出工程或明确指定文件中修改，保留原件。纯文本请求不强迫创建完整工程。

模板迁移在新目标目录进行：采用目标官方样式，迁移正文与资产，记录冲突宏、不可直接迁移的环境和未解决项；编译并检查渲染后交付。不得覆盖官方样式或调整字号/页边距来规避篇幅限制。

```bash
# 从 paper-writing-shared 执行
python3 scripts/project.py compare --before /absolute/path/original --after /absolute/path/revised --report /absolute/path/comparison.json
python3 scripts/project.py compile --project /absolute/path/revised
python3 scripts/project.py check --project /absolute/path/revised
```

`compare` 是提示事实漂移的静态辅助，不能替代对数学与研究含义的语义核对。新增或删减的数字必须结合上下文检查。

## 输出

交付修改稿、必要时 PDF、中文修改说明及待解决问题。轻量段落修改提供稿件和少量关键理由；全文修改附术语/数值/结论核对与格式验证结果。缺失数据、无法获得模板或不能编译时给出真实状态，不把中间产物称为已通过投稿检查。

不处理 rebuttal、审稿模拟或绘图；不改变用户已选择的研究方向。
