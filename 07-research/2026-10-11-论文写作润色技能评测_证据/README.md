# 本轮评测证据

入口：[评测报告](../2026-10-11-论文写作润色技能评测.md)、[逐项结果](results.json)、[冻结判据](rubric.json)。

- `inputs/`：中文合成材料；`inputs-sha256.json`记录最初输入校验值。
- `outputs/`：实际稿件、多文件工程、PDF和交付说明；E01、E05、E10为独立上下文代理产物。E03-original为输入工程，根代理与独立代理都保留原件。
- `runs/`：真实CLI stdout/stderr、修复前后探针、16项回归、入口校验、语义复核、当前文件版本比较与最终链接检查。
- `renders/`：当前实际PDF渲染；`render-manifest.json`绑定PDF与逐页图校验值，正向输出共11页。
- `renders-before-content-corrections/`：正文栏目及附录修正前渲染，对应历史PDF保存于相应before目录。
- `baseline-skills-sha256.json`、`final-skills-sha256.json`、`skills-delta.json`：评测前后资源版本；本轮包含修复，不能当作固定版本模型排名。

E08及E09的故意损坏工程是负向测试输入，并非合规稿。根代理初次稿件遗漏与修正前结果也保留，详见报告。JSON中的原临时绝对路径是执行溯源；可继续编辑的产物位于本目录outputs，不依赖临时目录存续。

独立代理说明是第一层执行记录；根代理另外核对了实际文本、源文件保护对象、归档后编译及PDF页面。评测结果不替代真实论文研究验证、550篇风格语料研究、全venue长稿覆盖或官方投稿系统检查。
