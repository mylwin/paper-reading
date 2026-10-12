# Confidence Gate 合成写作评测交付

本工程为合成技能评测的英文初稿，不是真实研究论文，不能直接投稿。已用本地 ICLR 2027 官方样式生成双匿名投稿格式的 3 页 PDF；未填入作者姓名、单位、联系方式、OpenReview ID 或其他身份信息。正文和 PDF 元数据已检查，未发现作者身份字段。

## 产物

- `build/main.pdf`：真实编译的 3 页英文短稿。
- `main.tex`、`sections/body.tex`：入口及可继续编辑的正文。
- `references.bib`：有意保留的空引用数据库。未生成伪造参考文献，不保留模板示例引用。
- `project.json`：模板年份、阶段、约束、原始官方文件校验和及草稿状态。
- `records/input-material-zh.md`：唯一材料。
- `records/claim-evidence-boundaries.md`：起草前主张—证据—边界表、符号表和篇章安排。
- `records/execution.json`：技能资源、来源、操作和决策记录。
- `records/validation.json`：编译、静态检查、视觉检查及未进行的官方检查分别记录。
- `records/pdf-text.txt`：实际 PDF 提取文本。
- `build/rendered/page-1.png` 至 `page-3.png`：逐页审阅的渲染图。
- `build/compile.json`、`build/format-check.json`、`build/main.log`：原始编译和检查记录。

## 内容边界

完成了规则的数学形式：固定分类器的最大类别概率至少 0.8 时，保留分类器原有预测；否则拒绝。明确 0.8 边界包含等号，不添加原材料未给出的模型名称、训练过程或 tie-break 规则。

所有应用、数据集、分类器细节、代码、种子、实验方案和结果、相关工作、理论结论均标为作者待补。未声称性能提升、校准保证、创新性、鲁棒性或部署收益。接受频率与接受样本错误率只是注明尚未执行和未经作者确认的可选描述统计定义，不是实测值、定理或已经批准的实验协议；零接受情形的错误率标为未定义。

AI use statement 单独成节，真实披露 AI 用于本次初稿组织和英文表达，并明确作者尚未复核。作者后续应核实正文和完整实际 AI 使用记录。模板文献、演示定理、图表和占位实验已删除，未假装生成研究证据。

## 模板与验证

使用本地 `paper-writing-shared/assets/templates/iclr/2027` 的 submission 模式，官方资产 SHA-256 与本地 spec 一致，未改官方字号、栏宽、页边距、行距或样式。模板 spec 核验日期为 2026-10-11。官方模板来源记录为 https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip ，官方作者指南及 AI 政策来源在 project.json 中记录；此次未进行在线复核，未使用外部写作语料作为事实来源。

- 编译：现有 TeX Live 的 latexmk/pdflatex，退出码 0；禁用 shell escape 与 latexmkrc。
- 引用与版面诊断：无未定义引用、缺字体、LaTeX 错误、overfull 或 underfull 诊断。
- 静态检查：主文结束标记在第 3 页，总页数 3，低于本地 spec 的主文 9 页上限；必需 AI use statement 存在；官方资产未改。检查未判完全通过，唯一 issue 为 `author-input-pending`，因为本次要求保留实际材料缺口。
- 视觉检查：已逐页检查 3 页实际渲染，标题、匿名作者、摘要、公式、缺口表和 AI 声明可读；页面未截断或重叠。
- 匿名检查：作者区、正文、链接、致谢和 PDF 身份元数据未包含作者信息；这是本稿审阅结果，非官方匿名审查。
- 官方投稿系统检查：未进行，未上传稿件。
- 作者事实及语义复核：未完成，不能把编译成功视为可投稿。

ICLR 档案明确完整语义语料审阅仅 1/50，本稿未把暂定编辑建议描述为已验证的 venue 通用风格。

## 本地重编译

在本工程目录执行：

```sh
latexmk -norc -pdf -interaction=nonstopmode -halt-on-error -file-line-error -latexoption=-no-shell-escape -outdir=build main.tex
```

工程没有在线依赖、外部图片或机器特定的正文路径；需要现有 LaTeX 工具链的常见数学和链接包。引用数据库当前未调用，待作者添加真实可核验文献后再接入 bibliography。没有自行安装 TeX，也没有修改仓库技能或资料。
