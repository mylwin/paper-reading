# ICML2025理论与实证样本验证证据

本轮连续读取两个预锁定归纳样本全部27页及A/B附录，直接检查14个关键原页。ICML由4增至6，全任务由26增至28；不含新留出，不代表独立数学/统计证明、实验复现或生成行为评测。

- `selected-batch.json`：样本、官方来源元数据及原文SHA；临时抽取路径是生成时位置，内容已删除。
- `batch-ledger.json`、`progress-before/after.json`：累计及增量，避免重复计数。
- `rule-applicability.json`：2篇×10维归纳规则的支持与适用限定，不是20项盲测通过率。
- `temporary-artifact-manifest.json`、`cleanup-result.json`：14张页图和2份重复抽取文本的哈希、复现方式及准确清理结果。16项共4735017字节；原论文和主解析不删。
- `validation-before-concurrent-index-audit.json`：首轮20/21，知识库索引变化检查不通过的原始结果。
- `concurrent-knowledge-index-changes.json`：8份索引在本轮期间新增今日其他论文登记的只读差异记录；没有技能语料登记，不回退，不声称识别了外部写入者。
- `validation.json`：21项通过，1项并行索引变动带限定保留；原论文资产、锁定划分、旧技能、此前审阅/评测和官方模板依赖均完整，无阻塞失败。没有重新抓取规范或重编译模板。
- `archive-sha256.json`：本目录证据和两个正式审阅JSON校验值；不包含原论文、完整解析和页图内容。

全部论文原文与主解析只在共享技能tmp，受Git忽略；没有执行Git暂存、提交或推送。本轮清理范围只含自身可重建中间产物，其他任务的资料不动。
