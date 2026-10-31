# ACL-005 全文写作审阅记录

- 样本：acl-2025.acl-long.1181，2025 ACL long，Evaluating the Evaluation of Diversity in Commonsense Generation。manifest 已核为 induction，未读取正式留出。
- 实际阅读：18/18 PDF页逐页语义阅读（正文、局限/伦理、参考文献、全部A–H科学附录）；阅读行为不是下载、解析或摘要替代。正文科学内容p1–10；参考文献p10–12；科学附录p12–18。
- 原页视觉：p1、6、7、8、9、12、13、14、15、16、17、18，共12页；覆盖全部Figure 1–14、Table 1–10及附录公式1–3。页图SHA与核查对象保存在技能tmp的review-cache/acl-005/visual-review-result.json；可重建页图已清理。
- PDF SHA256：37f2212d8d54de3e89aa050fae0b41bae7598ae3ddfd648ac1e991dbfa648505。
- 全页Markdown SHA256：86c70ba597469ae0c32ab2ba3577d6f7d248f7a174cad800d2e62e661235cb59。
- 官方XML SHA256：bd95ce2de89dbc2c001ca63c73585a2145b338df3b0ef48833a6e3d8ecb245fe；论文revision hash 904d6eed；无revision条目、无attachment。
- 2026-10-28读取官方landing及全部链接；仅论文PDF，无独立科学补充附件。A–H已整合进PDF并读全。landing SHA256：4ffb21d80863c5201550a499f9a138f6ccb64548774587388529a1f2f2f7423e。正文p2另有代码/数据仓库，非独立科学附录。
- web-access检查：Node可用，Chrome CDP未连接；公共静态页面用技能允许的curl读取成功。受限网络首次DNS失败，获准只读网络后landing成功；raw官方XML再拉取连接重置，因此版本核对使用本地已归档官方XML及实时landing，未伪称实时XML重新下载。

可迁移的写作观察：

1. p1–2以具体句集排序冲突解释任务需要，再将评测设计、标注资源和发现连成贡献；两项列表是实质发现，不强制凑新算法。
2. p2–3以最近工作的任务、候选变化、标注方式三个差别定位贡献；背景服务研究问题。
3. p4–6按构念、评分与排序偏差、候选质量变化、人类核查和同质量配对展开方法；区分设计预期与测量结果。
4. p7–9按总体、跨数据集、质量分层、指标一致性与分布诊断组织结果，每组证据回答一个问题。
5. p12–18附录补公式、完整模板、质量测量与压力测试；其压力测试会限制结论时，应将必要条件反馈到正文。
6. 已测结果直接表达；推测解释用suggest等。局限集中且针对语言/模型/标注者范围，勿复制“无法覆盖所有模型”等套式自辩。

研究审计与编辑反例（不迁入用户稿件作普遍免责声明）：

- 人类核查为70对高质量集合、5名受训人员，平均与GPT-4o一致80.6%，人类间Fleiss kappa 0.45；主要评测accuracy参照GPT-4o，勿直接合并为全部数据经人类认可。
- p8 Table4部分指标非Default→Para3单调下降；p7 Table3也有形式指标高于部分内容指标的格子，all/consistently需与表格粒度对应。
- p17 Table9 Nonsensical的Overall印18.1，而78.6×2.7/100约2.12；留待作者核查。
- Table1/2称bootstrap CI，G称Clopper–Pearson；H低质量ComVE内容指标最高38.8%，相对更好不自动代表绝对可靠。未做代码复现、显著性认证或文献表核验。

十维观察、适用边界与原创新句存于该样本唯一semantic review JSON。完整原文、landing、XML片段、页级视觉核查结果均只在技能tmp中。未改其他样本、聚合进度、技能、venue、registry，未Git提交。
