# ACL官方模板与格式核验（2026-10-28）

官方ACL2027 CFP已公布，按当前可核验届次配置长稿与短稿，语料仍由主调研使用2024–2025年论文。投稿上限为长稿8页、短稿4页；终稿各增加1页至9/5页。来源：[ACL2027主会CFP](https://2027.aclweb.org/calls/main/)。

已采用官方指定的通用`acl.sty`和`acl_natbib.bst`，固定仓库commit为`d5adc823ff0f80f98c80405ca0ab66c68e684409`，没有虚构年度专用样式。两个官方文件在四个工程均保持原字节。模板来源、官方页面快照位置及SHA见[official-sources.json](official-sources.json)。仓库README明确供作者使用且禁止改样式；该固定归档无独立LICENSE，不推定MIT/LPPL。来源：[官方样式仓库](https://github.com/acl-org/acl-style-files)。

`2027-long`与`2027-short`两个版本键，各有submission/final阶段；这样可直接使用现有CLI，无需新增论文类型参数。投稿使用review匿名模式，终稿使用final实名模式。骨架提供分文件正文、公式、算法、定理、图表、文献和双栏附录；所有研究数据和作者信息明确待补。集成条目与命令见[integration.json](integration.json)，子代理未修改registry、venue档案或两个技能入口。

四个工程全部实际编译成功，逐页查看12页，A4、字体嵌入、正文/附录双栏、引用及交叉引用均核验。长短稿当前仅占位内容相同，页数限制分别记录。投稿有匿名作者、行号和页码；终稿显示作者占位并去除行号、页码。静态检查只报告预期的`author-input-pending`，不能称可直接投稿。[render-results.json](render-results.json)保存逐配置PDF和页图校验值、字体报告及视觉观察。

首次检查发现末个跨栏表延至正文第2页，结论处的结束label却在第1页。已将本骨架的label放进末个正文浮动体并复编译，当前计数为2。该位置仅适用于当前骨架；实际稿件仍需按最后正文内容/图表的真实页面核对。稀疏占位骨架的跨栏表当前单独占一页，填写真实内容后应重新检查浮动位置，未以改字号、边距或官方文件规避篇幅。[首次观察](initial-float-boundary-observation.json)保留原记录。

必需的Limitations位于结论后、参考文献前；不能装入新增结果。伦理讨论按实际研究需要，Responsible NLP checklist按投稿流程真实填写，AI写作/代码帮助按实际使用及官方披露政策处理。附录在参考文献后，并保留双栏；关键论证不能只放附录。来源：[ARR官方要求](https://aclrollingreview.org/cfp)。A4、摘要不超过200词、字体与图表题注等格式依据[ACLPUB](https://acl-org.github.io/ACLPUB/formatting.html)。

未核验到PDF大小上限，spec中为null而不是宣称无限制；没有执行官方投稿系统检查，也没有对真实长稿、匿名代码或所有最大限页边界做验证。编译、视觉核验与投稿合规分别记录。

已清理本子任务的编译缓存、12张页图、重复解包目录和临时构建脚本；保留精简结果、官方来源归档及4个本地忽略的预览PDF。没有删除论文原件或完整解析，没有Git暂存、提交或推送。见[cleanup-result.json](cleanup-result.json)。
