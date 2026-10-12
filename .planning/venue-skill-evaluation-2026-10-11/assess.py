"""Combine retained evidence and explicit root semantic judgments.

Qualitative judgments here are reviewed statements, not automatic model scoring.
"""
from pathlib import Path
import hashlib,json,re
from datetime import datetime, timezone
W=Path(__file__).resolve().parents[2]
E=W/'07-research/2026-10-11-论文写作润色技能评测_证据'
def read(p):return json.loads((E/p).read_text())
def source_tree(p):return {str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and 'build' not in f.relative_to(p).parts}
baseline=read('source-tree-sha256.json')
counts={name:len(re.findall(r'\b[A-Za-z]+(?:[-\x27][A-Za-z]+)*\b',(E/'outputs'/name/'sections/body.tex').read_text())) for name in ['E03-original','E03-revised']}
matrix=[r for r in read('runs/baseline.json') if r['case'].startswith('E07-')]
original_unchanged=baseline['E03-original']==source_tree(E/'outputs/E03-original')
body_same=(E/'outputs/E03-revised/sections/body.tex').read_bytes()==(E/'outputs/E04-tai-migrated/sections/body.tex').read_bytes()
bib_same=(E/'outputs/E03-revised/references.bib').read_bytes()==(E/'outputs/E04-tai-migrated/references.bib').read_bytes()
impact=re.search(r'\\begin\{IEEEImpStatement\}(.*?)\\end\{IEEEImpStatement\}',(E/'outputs/E04-tai-migrated/main.tex').read_text(),re.S).group(1)
impact_words=len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",impact))
verdicts={
 'E01':{'status':'passed-limited-independent-case','review':'英文原文逐项核对：80.0→82.0及76.0→75.5、三种子、相同预算均保留；不添加统计或速度结论，没有venue追问或工程。','evidence':['outputs/E01-independent-polishing/revised.md','outputs/E01-independent-polishing/changes-zh.md']},
 'E02':{'status':'passed-root-assisted-draft-workflow','review':'方法、条件准确率/覆盖率、两个合成数据集的相反变化、有界损失与C>0及完整范围界证明忠实。未补造论文引用。可编译，静态检查正确拒绝待补稿；该输出由根代理知晓标准后辅助执行，不能作为盲测成功率。','evidence':['outputs/E02-icml-writing/main.tex','outputs/E02-icml-writing/claim-evidence.md','outputs/E02-icml-writing/build/format-check.json']},
 'E03':{'status':'initial-format-omission-corrected','review':'删除重复与不受支持宣传，事实和保护对象不漂移。初稿遗漏源工程原本缺失的ICML必需Impact Statement，静态规则也曾未列入；发现后保存原输出，补齐栏目并编译。','evidence':['runs/E03-compare.json','outputs/E03-before-impact-correction/article.tex','outputs/E03-revised/article.tex','runs/content-corrections.json']},
 'E04':{'status':'initial-macro-and-appendix-errors-corrected','review':'根代理首轮迁移遗漏theHalgorithm格式宏，被比较工具发现；恢复后，只剩模板排版数字0.3变化。实际PDF又发现IEEE单附录命令抑制标题，改为appendices后标题及Appendix A引用一致。正文、参考文献与作者资产保留，影响声明区分潜在用途与实证结果。','evidence':['runs/E04-compare.json','runs/E04-compare-final.json','outputs/E04-before-macro-correction/main.tex','outputs/E04-before-appendix-correction/main.tex','outputs/E04-tai-migrated/main.tex']},
 'E05':{'status':'passed-limited-independent-draft-case','review':'独立代理没有实验与引用仍完成真实方法定义，0.8边界包含等号；明确最大类别概率不等于校准正确率。新增评估量明确是作者待决的建议定义，不冒充数据或定理。AI use statement真实披露草拟任务及作者未复核。3页PDF可编译，待补检查如期拒绝。','evidence':['outputs/E05-independent-iclr/records/validation.json','outputs/E05-independent-iclr/sections/body.tex','runs/E05-archive-verification.json']},
 'E06':{'status':'passed-CLI-integration','review':'未知venue与2099年份均退出2且未创建目标；已有目录拒绝覆盖，源文件树不变。','evidence':['runs/baseline.json']},
 'E07':{'status':'passed-stage-assembly-matrix','review':'11 venue × 2阶段生成22工程，全部官方资产哈希一致。未知的TPAMI匿名要求及部分终稿页数维持unknown；该矩阵是组装测试，不计作22次新编译或新风格验证。','evidence':['runs/baseline.json']},
 'E08':{'status':'failures-retained-and-bounded-fixes-verified','review':'默认入口、旧PDF、同键BibTeX变化、直接可见作者漏检已修复；含宏标题改为显式要求渲染字数核查，不能自动宣称合规。18词宏标题实际超过TAI 15词限制；负向样例故意保留，不修成合规稿。一般匿名泄露与任意TeX展开仍未完整自动验证。','evidence':['runs/baseline.json','runs/after-fix.json','runs/final-boundary-checks.json','renders/E08-macro-title/page-1.png']},
 'E09':{'status':'passed-new-regression-variants','review':'新入口camera.tex编译成功；文献年份改变且人为保留mtime仍被编译指纹拒绝；引用内容比较也检出变化；缺ICML Impact Statement的新拷贝被明确拒绝。','evidence':['runs/after-fix.json','runs/E09-bib-compare.json','runs/content-corrections.json']},
}
if (E/'runs/E10-root-review.json').exists():verdicts['E10']=read('runs/E10-root-review.json')
else:verdicts['E10']={'status':'pending-independent-migration-output'}
result={'evaluated_at':datetime.now(timezone.utc).isoformat(),'grading_method':'Root semantic review of actual artifacts, CLI observations, and isolated-context agent forward tests; no automatic linguistic-quality score or aggregate pass rate','fixture_scope':'Synthetic selective classification, empirical mixed-results paragraph, elementary bounded-risk proof. No original research validity, statistical correctness or 550-paper venue-style validation implied.','verdicts':verdicts,'metrics':{'unit_tests':{'count':16,'passed':read('runs/unittest-final-16.json')['exit_code']==0},'template_assembly':{'count':len(matrix),'passed':all(r['exit_code']==0 and r['official_hashes_match'] for r in matrix)},'original_source_tree_unchanged':original_unchanged,'root_compression_source_tokens':counts,'root_compression_reduction':1-counts['E03-revised']/counts['E03-original'],'compression_metric_limit':'Heuristic English tokens in body.tex, includes TeX argument words; not official or rendered prose count','migration_body_byte_identical':body_same,'migration_bibliography_byte_identical':bib_same,'root_tai_impact_words':impact_words,'official_system_validation':'not performed','corpus_research_completed_count':14,'corpus_research_target':550},'versions':'Exploratory fix-and-retest session; baseline and final skill hashes retained. Not a benchmark holding all resources fixed throughout execution.'}
(E/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
assert original_unchanged and body_same and bib_same
assert all(r['exit_code']==0 and r['official_hashes_match'] for r in matrix)
assert 100<=impact_words<=150
print(json.dumps(result['metrics'],ensure_ascii=False,indent=2))
