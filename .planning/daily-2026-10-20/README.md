# 2026-10-20 每日检索过程

日报及交付文件位于 `../../08-daily/2026-10-20/`。按工作区共享配置检索大模型优化器设计，近期窗口为2026-09-20至2026-10-20，年度补充窗口为2025-10-20至2026-09-19。

## 检索与日期核验

- 原技能脚本已实际调用 arXiv、OpenReview 与 Semantic Scholar。arXiv API返回503/429/超时，S2两次429；未取得年度高影响力补充。
- `supplement_openreview.py`逐个配置词检索；14词成功，AdamW词429。各词最多首页100条，窗口内并集35条。原始元数据及覆盖记录分别存于 `openreview-raw.json`、`openreview-coverage.json`。
- `search_html.py`使用全部15配置词查询arXiv官方页面；8词成功、7词SSL失败，页面上限200。原始并集1025条，普通页显示的提交日期可能属于修订版，不能直接视为首发日期。
- 完整15词高级查询502后，`search_advanced.py`以Muon、SignSGD和gradient clipping短查询按首次提交日期取得143条，记录于 `arxiv-first-date-filtered.json` 与 `arxiv-advanced-coverage.json`。
- 普通页并集只保留首次提交过滤集合内的记录，或2026-10编号的记录，得到864条；无法确认窗口的其他9月结果保守排除。`replay_dedup.py`再按全部15关键词在题目/摘要中作字面匹配，交由技能原评分筛选并限200，实际输入134条。该恢复方式非穷尽检索。
- OpenReview详情接口返回403挑战，未绕过。前三均取arXiv来源，原始页面与PDF首页已独立核验。
- OPT和DynaFront的Workshop类别依据主办方页面：<https://opt-ml.org/index.html>、<https://sites.google.com/view/dynafrontneurips26>。按配置排除，不以会场推断质量。

## 筛选、归档与校验

156条唯一记录，70条历史命中，86条候选。前十逐篇阅读题目与摘要，保留原分数与screening_rank，复排理由存于日报编辑JSON。前十排除2条粒子物理误命中、4条Workshop，保留4项正式阅读建议；前三为Ori、NBCLT、AURA。三篇PDF归档于2026-10，均新下载。

`initial-screening.json`保留复排前数据；`arxiv-source-verification.json`记录前三原始页面摘要、署名和许可。完整中文摘要已逐句对照方法、数字和限定条件；仅作摘要初评。`validate.py`核对分数保留、附录完整性、摘要及翻译锚点、PDF身份、本地链接和提交索引范围。

全库检查的9处既有研究文档工作区外链接问题与运行前一致，本次新增问题0，缺失README和月份一致性问题0。

`prepare_commit_indexes.py`以任务开始时HEAD构造临时索引快照，仅加入本次三篇PDF。工作区中已有MUON2精读及其索引改动保持未提交。
