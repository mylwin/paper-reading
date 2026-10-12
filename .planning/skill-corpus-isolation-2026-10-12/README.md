# 技能语料隔离

用户明确要求论文不提交，技能调研论文仅存放在skill的tmp，不污染个人论文知识库。

- [迁移结果](migration-result.json)：453个主PDF、453个解析文件及32个研究副本/全文缓存，总计938个资产；原有356个知识库资产未变。
- [候选识别依据](candidate-assets.json)：已归档样本、官方来源框、原提交前的文件集合和用户笔记/翻译交叉检查，不按日期或目录批量猜测所有权。
- [并行技能修改](parallel-skill-modifications.json)：保留其他任务的技能元数据修改。
- [知识库刷新](knowledge-registry-after.json)：研究行移除后的真实用户资产登记。
- 正式旧新路径和校验值在共享技能`corpus-locations.json`。原始资料和历史记录原字节快照在受Git忽略的`tmp`中。

仅重建尚未推送的本地提交，保留工作区文件。原含论文提交不在当前分支的待推送祖先中，不执行force push。原六个技能、知识库资产和科研评测内容保留。

最终核验结果和提交范围由本目录的validation.json、unit-tests.txt和staged-scope.json记录。知识库索引中的其他用户变动、两篇日报原文和旧技能元数据的并行变动不纳入此次技能提交。
