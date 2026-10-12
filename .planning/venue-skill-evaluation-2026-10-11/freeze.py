"""Freeze evaluation materials before execution; refuse to overwrite an archive."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

WORK = Path(__file__).resolve().parents[2]
OUT = WORK / '07-research/2026-10-11-论文写作润色技能评测_证据'
if OUT.exists():
    raise SystemExit('Evaluation archive already exists; do not overwrite frozen materials')
OUT.mkdir()
(OUT / 'inputs').mkdir()
(OUT / 'outputs').mkdir()
(OUT / 'runs').mkdir()

notes = '''本材料专用于技能评测。所有经验数据均为合成测试数据，不是已完成的研究成果。
研究对象：固定阈值的选择性分类（Selective Classification）。方法 Gate 接受预测置信度至少 0.8 的样本，否则拒绝预测；基线 Always 总是预测。置信度定义为模型输出最大类别概率，模型固定，不重新训练。
两个人工数据集 Alpha、Beta，各有 1000 个样本。使用同一模型和 5 个评估种子；以下是种子均值，不提供方差或置信区间。
Alpha：Always 准确率 80.0%，Gate 在已接受样本上的准确率 84.0%，覆盖率 60.0%。Beta：Always 准确率 76.0%，Gate 已接受样本准确率 75.5%，覆盖率 58.0%。不得将条件准确率当总体准确率。
未做统计显著性检验，未测量延迟、训练成本或能耗，也未评估分布偏移。只比较一个基线，不能声称 SOTA。数据生成器细节缺失，标记作者待补。
理论定义：固定分类器 h 与接受函数 a(x)∈{0,1}；损失 0≤ℓ(h(x),y)≤1；C=E[a(X)]>0；R_sel=E[a(X)ℓ(h(X),Y)]/C。可直接得出 0≤R_sel≤1，不提供泛化界、不保证跨分布成立，不代表方法降低风险。
完整证明：由 0≤a(X)ℓ(h(X),Y)≤a(X)，取期望，再除以 C>0。
作者、机构、资助、近邻论文、代码链接均未提供。不能伪造 BibTeX，可用明确待补项而不填真实研究引用。
'''
(OUT / 'inputs/research-notes-zh.txt').write_text(notes)
paragraph = '''我们在两个公开分类数据集上比较了方法 A 与基线 B，使用相同的训练预算和 3 个随机种子。数据集一的平均准确率由 80.0% 升至 82.0%；数据集二由 76.0% 降至 75.5%。未记录标准差，未做显著性检验，也没有测量运行时间。方法只在固定训练预算下测试，尚未评估分布偏移。
原稿：Our revolutionary method consistently outperforms all existing methods and is significantly faster.
'''
(OUT / 'inputs/local-polishing-zh.txt').write_text(paragraph)
cases = [
    {'id':'E01','executor':'independent-agent','skill':'paper-polishing','request':'轻量润色并将 local-polishing-zh.txt 改成英文摘要段落；未指定 venue；交付中文修改说明，不创建工程。','gates':['两组结果与3种子不变','负结果保留','不增加显著性、速度或SOTA结论','区分百分点与相对变化','无不必要venue询问或LaTeX工程']},
    {'id':'E02','executor':'root-assisted','skill':'paper-writing','request':'根据 research-notes-zh.txt 起草 ICML 2026 投稿英文短稿及完整 LaTeX 工程，明确材料是合成评测。','gates':['条件准确率与总体准确率不混同','方法定义可识别','理论条件C>0及完整证明忠实','负结果和统计信息缺口保留','无伪造引用作者','ICML官方资产不变且编译成功','待补稿不称可投稿']},
    {'id':'E03','executor':'root-assisted','skill':'paper-polishing','request':'润色并压缩现有 ICML 多文件工程，入口 article.tex；保留宏、公式、数据、引用、标签及结构，原件保留。','gates':['有效事实与条件保留','至少压缩正文词数20%','保护对象静态比较不漂移','原件树哈希不变','真实入口编译成功','对不支持的普遍性与速度措辞说明修改']},
    {'id':'E04','executor':'root-assisted','skill':'paper-polishing','request':'将 E03 结果迁移到 TAI 当前核验的 Regular 双匿名投稿工程，在新目录保存；提供符合要求的影响声明，不增加实验结论。','gates':['使用IEEEtai.cls官方文件','原工程不变','正文数学数据引用宏不漂移','Impact Statement为100–150词','标题摘要关键词符合当前规则','匿名身份未泄露','编译并视觉核查','不虚称通过官方投稿系统']},
    {'id':'E05','executor':'root-assisted','skill':'paper-writing','request':'只有方法草案，无任何实验与引用，请生成 ICLR 2027 投稿短稿；缺失项显式待补。','gates':['不造数据引用证明','真实AI使用披露不写不存在的作者确认','AI use statement存在','可编译但静态检查应拒绝待补稿','明确尚未完成']},
    {'id':'E06','executor':'CLI-integration','skill':'shared-helper','request':'尝试未知venue、未知年份及已有目标目录；不得悄悄替代模板或覆盖。','gates':['每项拒绝','未知目标没有创建','已有文件哈希不变']},
    {'id':'E07','executor':'CLI-integration','skill':'shared-helper','request':'对所有11个venue的submission和final生成独立工程，核查官方资产、阶段配置与占位状态；沿用上一轮编译基线，不重复计作新编译。','gates':['22工程正常生成','官方资产哈希一致','阶段配置可追溯','共享目录没有第三个SKILL.md']},
    {'id':'E08','executor':'failure-probes','skill':'shared-helper','request':'探测自定义入口编译、源文件修改后的旧PDF、同键BibTeX漂移、宏展开标题超词数和真实作者匿名泄露，保存失败与修复后证据。','gates':['危险漏检与明确静态边界分别报告','保留修复前结果','修复只针对实测问题','新变体回归']},
]
rubric={'version':1,'created_at':datetime.now(timezone.utc).isoformat(),'fixture_scope':'synthetic research facts, not empirical research or proof validation','cases':cases,'grading':'Each gate is pass/fail/unknown with observable artifact evidence. Critical fidelity or compilation failures are not averaged away. No score claims venue-style corpus validation.','independence':'Only an agent that receives a request and raw input, not this rubric or expected output, counts as independent.'}
(OUT / 'rubric.json').write_text(json.dumps(rubric,ensure_ascii=False,indent=2)+'\n')
hashes={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['paper-writing','paper-polishing','paper-writing-shared'] for p in (WORK/'.claude/skills'/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(OUT/'baseline-skills-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
(OUT/'inputs-sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'inputs').rglob('*') if p.is_file()},indent=2)+'\n')
print(OUT)
