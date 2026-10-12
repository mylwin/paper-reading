# 实际读取资源

本次仅使用指定技能及其直接引用的按需资源，未读取其他技能、venue 档案、模板、registry 或 provenance。

| 顺序 | 绝对路径 | 使用目的 |
| --- | --- | --- |
| 1 | /Users/myl/Documents/paper/.claude/skills/paper-polishing/SKILL.md | 确认轻量修改范围、按需加载策略、无 venue 局部润色及纯文本输出规则 |
| 2 | /Users/myl/Documents/paper/.claude/skills/paper-writing-shared/core/evidence.md | 保留数字、比较条件与结论范围，禁止补造统计和实验 |
| 3 | /Users/myl/Documents/paper/.claude/skills/paper-writing-shared/core/polishing.md | 识别过强主张，区分准确率、显著性与运行时间证据 |
| 4 | /Users/myl/Documents/paper/.claude/skills/paper-writing-shared/core/language.md | 中文转英文、术语保持、移除无证据的显著性和普适表述 |
| 5 | /Users/myl/Documents/paper/.claude/skills/paper-writing-shared/sections/manuscript.md | 只应用“标题／摘要”相关规则，保留证据及适用范围 |

第 2–4 项通过同一批工具调用读取；表中编号表示记录顺序，不表示相互依赖。
