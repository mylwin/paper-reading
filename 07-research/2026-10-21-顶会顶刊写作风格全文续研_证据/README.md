# 2026-10-21 全文续研证据

本批完成三个原锁定IJCAI归纳样本。完整写作风格审阅累计由23增至26，550篇验收尚未完成；无新留出、无新增主PDF、无新官方规范核验或模板编译。

- `selected-batch.json`：锁定ID、PDF路径与SHA256、选择顺序。`extracted_text_path`是当时临时抽取位置，已在核验后清理，可从原PDF重新生成。
- `paused-review-notes.json`：用户暂停时保存的中间阅读观察，保留历史；正式结果以三个canonical semantic-reviews和本批账本为准，不把暂停时的疑点视为已确认错误。
- `batch-ledger.json`、`progress-before.json`、`progress-after.json`：增量与累计计数，避免重复计入。
- `temporary-artifact-manifest.json`：14页图及3份重复抽取文本的路径、字节、SHA256和复现命令；内容仅曾放技能tmp，核验后已删。
- `cleanup-result.json`：准确删除17个可重建文件，共6656172字节；没有删除论文原件、主要解析或旧评测。
- `validation.json`：18项范围、划分、计数、字节保护、Git隔离与链接检查全部通过，不代表证明正确性或实验复现。
- `archive-sha256.json`：本批记录及三个正式审阅JSON的内容校验值，不包含论文原文、全文解析或页面图片。

PDF、原始主要解析和迁移语料继续仅存在共享技能tmp，受Git忽略。本批没有暂存、提交或推送；用户知识库、原有六个技能与既有评测证据未变。
