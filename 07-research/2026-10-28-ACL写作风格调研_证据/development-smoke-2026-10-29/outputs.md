# ACL 润色技能开发期冒烟评测：实际输出

2026-10-29。生成者独立执行三个局部润色请求。本文件只保存实际产出，不打分；不涉及正式留出样本，也不代表ACL最终技能有效性评测。

## 案例 1

请求：按ACL风格轻度润色，原数据已核准

原稿：

```text
On an A100 GPU at batch size 8, Method A achieves a 1.94× wall-time speedup over greedy decoding. The method retains the full KV cache in GPU memory.
```

改写：

```text
On an A100 GPU with a batch size of 8, Method A achieves a 1.94× wall-time speedup over greedy decoding. It retains the full KV cache in GPU memory.
```

说明：轻调搭配并减少重复主语；保留硬件、批大小、比较对象、1.94倍实测时间收益及完整KV缓存驻留事实。

## 案例 2

请求：润色并压缩

原稿：

```text
Our experiments may perhaps suggest that Method B could potentially improve accuracy. On Dataset X, accuracy increases from 80.0% to 82.0% across the same 500 test examples. No significance test was conducted.
```

改写：

```text
On the same 500 test examples from Dataset X, accuracy increases from 80.0% to 82.0% with Method B. No significance test was conducted.
```

说明：删去重复弱化的概括句，直接报告已给出的准确率变化；保留同一测试样本和未做显著性检验的信息。

## 案例 3

请求：只做语言润色，不改变数学含义或TeX

原稿：

```text
Under Assumptions~\ref{ass:smooth} and~\ref{ass:noise}, Theorem~\ref{thm:rate} guarantees $\mathbb{E}\|\nabla f(x_R)\|^2 \le C/\sqrt{T}$; the constant $C$ depends on $L$ and $\sigma$. This bound concerns expected stationarity, rather than wall-clock time.
```

改写：

```text
Under Assumptions~\ref{ass:smooth} and~\ref{ass:noise}, Theorem~\ref{thm:rate} guarantees $\mathbb{E}\|\nabla f(x_R)\|^2 \le C/\sqrt{T}$; the constant $C$ depends on $L$ and $\sigma$. This bound concerns expected stationarity, rather than wall-clock time.
```

说明：原句已准确清楚，保留原文；假设引用、公式、常数依赖及期望平稳性与实际运行时间的区别均保持。

## 实际读取资源

- `.claude/skills/paper-polishing/SKILL.md`
- `.claude/skills/paper-writing-shared/core/evidence.md`
- `.claude/skills/paper-writing-shared/core/polishing.md`
- `.claude/skills/paper-writing-shared/core/language.md`
- `.claude/skills/paper-writing-shared/core/latex.md`
- `.claude/skills/paper-writing-shared/registry.json`
- `.claude/skills/paper-writing-shared/venues/acl.md`

资源校验值见 outputs.json。未读取正式留出、未新建LaTeX工程、未修改技能或知识库、未执行Git写操作。案例3仅为TeX文本维护，不涉及模板版式检查或PDF编译。
