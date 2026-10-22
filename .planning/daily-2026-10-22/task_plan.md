# 2026-10-22 每日检索

- 已读取工作区规范、paper-daily技能、共享研究评价准则及配置。
- 按配置15关键词检索arXiv与OpenReview；年度补充Semantic Scholar限流。
- 默认OpenReview持续翻页后改用逐关键词单页有界检索；覆盖限制见日报source-coverage.json。
- 前十逐篇摘要审阅，保留脚本分数与screening_rank；语义顺序记录于daily-editorial.json。
- 优先归档HyMUON、CAST、AlgoPerf三篇原始PDF，写入2026-10月份。
- 日报由render_note.py与本任务render_daily.py生成，write_note.py负责落盘。
- sync_indexes.py统一维护八份索引与月份README；二次刷新无变化。
- PDF首页身份、39/14/31页及49个日报本地链接校验通过。
- 全库9处旧调研断链与执行前相同；本次新增断链0、月份错位0。
- 提交范围仅本任务日报、PDF、索引及过程审计；工作区原有其他调研改动不纳入。

## 审计材料

- initial-screening.json：语义复排前的完整输出，便于复核评分未改写。
- arxiv-raw.json / openreview-raw.json：本轮源记录。
- search_bounded.py：复用既有评分、去重与输出，仅为OpenReview设定查询边界。
- editor.py：本轮编辑内容；权威成品为日报目录的daily-editorial.json。
- render_daily.py：保留标准布局，明确候选、正式建议与排除状态。
- validate.py：本轮产物核验；依赖同目录基线链接快照。
- 三个first-page.txt：身份校验片段，无论文解析资产写入02-markdown。
