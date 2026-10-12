from pathlib import Path
import json,hashlib
out=Path('07-research/2026-10-12-顶会顶刊写作风格调研_证据/aaai')
rs=[json.loads((out/f'{i}.json').read_text()) for i in ['aaai-2025-12','aaai-2024-22']]
labels={'reader_argument':'读者与论证目标','title_abstract':'标题和摘要','introduction':'引言与贡献','related_work':'相关工作','method_theory':'方法、算法与理论','evidence':'证据链','discussion_limits':'讨论与边界','paragraph_language':'段落与措辞','main_appendix':'正文和附录','actionable':'可执行建议'}
lines=['# AAAI 跨年度全文写作风格审阅（2026-10-12）','','本批连续语义审阅两篇预先锁定的归纳样本，没有改动留出划分、共享计数、manifest、records、技能档案或旧评测档案。两篇都是AAA​​I Press正式发表版本；当前投稿模板与这些发表稿的排版必须分别管理。以下规则属于样本观察和编辑建议，不是官方要求，也不足以估计AAAI全会风格频率。','','| 样本 | 主PDF全读 | 实际相关文本附录 | 视觉检查 | 计数建议 |','|---|---|---|---|---|']
lines+=['| aaai-2025-12 在线联盟学习 | 9页，含参考文献 | 作者补充A–F共7页全读；1949索引行无缺行 | 主文4、5、6、7页 | 语义全文完成；补充二进制归档与视觉核对未完成 |','| aaai-2024-22 KD-Club | 8页，含参考文献 | 正文无引用外部文本附录；官网仅列PDF，作者链接为代码 | 主文3、6、7页 | 语义全文完成 |','','“完成”仅指实际全部文本的写作风格语义阅读。没有执行代码、复现实验、独立核验下界或验证全部证明正确性。补充二进制PDF下载先出现DNS失败，升级联网后仍解析超时；网页PDF解析内容可读，截图请求未返回可用的可见图片。因此保留真实文本完成状态，同时登记档案和视觉限制，不宣称补充公式排版已验收。','','## 来源与版本','','- 在线联盟学习：[AAAI官方发表页](https://ojs.aaai.org/index.php/AAAI/article/view/33498)，DOI 10.1609/aaai.v39i13.33498，2025-04-11，13709–13717页；[正文明确引用的作者补充](https://u.cs.biu.ac.il/~agmon/CohenAAAI25Sup.pdf)，同标题、作者及AAAI2025标识。','- KD-Club：[AAAI官方发表页](https://ojs.aaai.org/index.php/AAAI/article/view/30061)，DOI 10.1609/aaai.v38i18.30061，2024-03-24，20735–20742页；[作者代码仓库](https://github.com/JHL-HUST/KD-Club)为实现产物，不是遗漏的文本附录。','','本地PDF SHA-256与锁定记录一致，正文标题、作者、会议信息与发表页相符。本次未重新下载并逐字节比较出版方PDF，不能把缓存身份核验表述成新的远端字节一致性证明。']
for r in rs:
 lines+=['',f"## {r['id']}：{r['title']}",'',f"[逐篇结构化证据]({r['id']}.json)。",'', '| 维度 | 样本观察 | 证据位置与适用边界 |','|---|---|---|']
 for key,value in r['ten_dimensions'].items():lines.append('| '+labels[key]+' | '+value['observation']+' | '+'；'.join(value['positions'])+'。'+value['applicability']+' |')
 lines+=['','可执行建议：','']
 for k in ['introduction','method_theory','evidence','main_appendix']:lines.append('- '+r['ten_dimensions'][k]['editorial_advice'])
 lines+=['','反例与需保留的编辑审计：','']
 for c in r['counterexamples']:lines.append('- **'+c['position']+'**：'+c['finding']+' '+c['editorial_action'])
lines+=['','## 两文比较与下一步适配','','两文都以明确机制支撑贡献，但结构不能强行统一。在线联盟学习采用独立相关工作、较长模型定义、分反馈形式的定理和外部证明；KD-Club将相关工作置于引言及baseline复述，正文直接保留短界证明与手算图例，再给两阶段集成和分层实验。二者区别首先由论文类型解释，尚不能归因于2024与2025的年度风格变化。','','写作技能可按稿件类型选择论证骨架；润色技能应优先检查主张中的量词、反馈/信息条件、指标量纲与附录引用。不能从发表事实推导所有原稿表述都正确：KD-Club表格的两处局部败例和k=3消融曲线应保留；在线联盟学习的主附录定理编号及范围/近似等式需要作者确认，不能当作可复刻的写作模板。','','置信划分：逐篇结构及主文视觉确认高；样本内反例高；AAAI普遍频率未估计；官方要求本批未新增。正式留出验证没有执行。','','本批原始网页文本和7张主文渲染位于本目录；sha256-manifest.json保存附件完整性记录。']
(out/'README.md').write_text('\n'.join(lines)+'\n')
manifest=[]
for p in sorted(out.rglob('*')):
 if p.is_file() and p.name!='sha256-manifest.json':manifest.append({'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(out/'sha256-manifest.json').write_text(json.dumps({'created':'2026-10-12','files':manifest},ensure_ascii=False,indent=2)+'\n')
planning=Path('.planning/venue-writing-aaai-2026-10-12')
(planning/'progress.md').write_text('# 审阅完成\n\n2025-12主刊9页、作者补充7页全部连续语义读完；补充1949行覆盖。2024-22主刊8页全部读完；无引用外部文本附录。主刊7页完成实际视觉核对。\n\n补充二进制下载失败、视觉截图未成功，是明确保留的归档/排版限制；不等于未读取补充实际文本。\n\n结果保存到07-research/2026-10-12-顶会顶刊写作风格调研_证据/aaai；共享清单/计数/技能未改。\n')
(planning/'task_plan.md').write_text('# AAAI 连续全文风格审阅\n\n- [x] 2025-12 主文9页及补充7页连续语义审阅\n- [x] 2024-22 主文8页与附录检索\n- [x] 身份、版本及主文7页可视核验\n- [x] 保存逐篇十维证据；不改共享计数\n- [ ] 补充二进制PDF归档与视觉复核：DNS超时及截图不可用\n')
print('Saved README and',len(manifest),'attachment checksums')
