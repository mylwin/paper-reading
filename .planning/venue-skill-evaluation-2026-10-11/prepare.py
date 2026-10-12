"""Assemble root-assisted scenario artifacts; manuscript text authored during evaluation.

This is NOT an automated model runner. Text below is the actual root-agent output.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys

WORK=Path(__file__).resolve().parents[2]
E=WORK/'07-research/2026-10-11-论文写作润色技能评测_证据'
SCRIPT=WORK/'.claude/skills/paper-writing-shared/scripts/project.py'
def init(name,venue):
    target=E/'outputs'/name
    if target.exists(): raise SystemExit(f'Refusing overwrite: {target}')
    subprocess.run([sys.executable,str(SCRIPT),'init','--venue',venue,'--mode','submission','--output',str(target)],check=True)
    return target
body=r'''\section{Introduction}
Selective classification allows a model to abstain from uncertain predictions.
We study a fixed confidence gate, rather than a learned rejection policy.
This synthetic evaluation asks whether a fixed threshold changes accuracy among
accepted predictions. Coverage is reported alongside conditional accuracy because
the gate changes which predictions are evaluated.
\section{Method}\label{sec:method}
Gate uses a fixed classifier and accepts a prediction when its maximum class
probability is at least 0.8; otherwise, it abstains. Always uses the same classifier
and accepts every prediction. Neither method retrains the model.
For a fixed classifier $h$ and acceptance function $a(x)\in\{0,1\}$, assume
$0\leq\ell(h(x),y)\leq1$ and positive coverage $C=\mathbb{E}[a(X)]>0$.
Selective risk is
\begin{equation}\label{eq:risk}
R_{\mathrm{sel}}=\frac{\mathbb{E}[a(X)\ell(h(X),Y)]}{C}.
\end{equation}
\begin{PWTheorem}\label{thm:bounded}
Under these assumptions, $0\leq R_{\mathrm{sel}}\leq1$.
\end{PWTheorem}
The result is a range bound, not a generalization guarantee or a comparison
between Gate and Always. Appendix~\ref{app:proof} gives the full proof.
\section{Synthetic Evaluation}\label{sec:results}
Alpha and Beta each contain 1000 synthetic samples. Both methods use the same
model and 5 evaluation seeds; results are seed means, with no variance estimates.
\begin{table}[htbp]\centering
\caption{Synthetic fixture values. Always accuracy uses all samples;
Gate accuracy is conditional on acceptance. These values are not research findings.}
\label{tab:results}
\begin{tabular}{lrrr}\hline
Dataset & Always (\%) & Gate (\%) & Coverage (\%)\\\hline
Alpha & 80.0 & 84.0 & 60.0\\
Beta & 76.0 & 75.5 & 58.0\\\hline
\end{tabular}\end{table}
Table~\ref{tab:results} shows a higher conditional accuracy on Alpha and a lower
conditional accuracy on Beta. Different evaluation populations prevent interpreting
these differences as an overall accuracy improvement. No significance tests,
runtime measurements, or distribution-shift evaluations were performed.
\section{Limitations and Conclusion}
A single baseline and synthetic datasets do not establish state-of-the-art
performance. The classifier and data-generation details are
\AuthorPending{required for reproducibility}. The fixed threshold does not
guarantee lower risk, and deployment benefits remain untested.
\label{pw:main-end}
'''
abstract='''We examine a fixed confidence gate for selective classification using a synthetic
evaluation fixture. The gate abstains from predictions below a specified confidence
threshold. We report accuracy among accepted predictions alongside coverage on two
synthetic datasets. Conditional accuracy increases on one dataset and decreases on
the other; these observations do not establish an overall accuracy improvement.
A range bound for selective risk requires positive coverage and bounded loss.
Statistical uncertainty, runtime, and distribution shift remain unevaluated.'''
proof=r'''\appendix
\section{Proof of the Range Bound}\label{app:proof}
Since $0\leq a(X)\ell(h(X),Y)\leq a(X)$, taking expectations gives
$0\leq\mathbb{E}[a(X)\ell(h(X),Y)]\leq C$. Division by $C>0$ proves
Theorem~\ref{thm:bounded}. This argument applies to the specified distribution;
it does not establish a bound under distribution shift.
'''
p=init('E02-icml-writing','icml')
(p/'sections/body.tex').write_text(body)
main=(p/'main.tex').read_text()
start=main.index(r'\begin{abstract}')
end=main.index(r'\input{sections/body.tex}')
main=main[:start]+r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}'+'\n'+main[end:]
main=main.replace('AUTHOR INPUT PENDING: Manuscript Title','A Fixed Confidence Gate for Selective Classification')
main=main.replace(r'\icmltitlerunning{Manuscript Title}',r'\icmltitlerunning{Confidence Gate Evaluation}')
main=main[:main.index(r'\bibliographystyle')]+'\n'+r'\section*{Related Work}'+'\n'+r'\AuthorPending{Supply verified nearest references; no research citation has been invented.}'+'\n'+r'\clearpage'+'\n'+proof+'\n'+r'\end{document}'+'\n'
(p/'main.tex').write_text(main)
(p/'references.bib').write_text('% No research reference was supplied.\n')
(p/'delivery.md').write_text('合成评测短稿；已给方法、条件、范围界与完整证明。表格均来自合成输入。研究引用、具体模型与生成器细节待补；ICML 2026 投稿模板；不是可投稿论文。Impact Statement 尚待实际研究事实补齐。\n')
(p/'claim-evidence.md').write_text('主张：固定阈值选择性分类；证据：定义与表格；边界：固定模型、两个合成数据集。\n主张：风险范围界；证据：附录证明；条件：有界损失与正覆盖率；不是泛化或性能提升保证。\n术语：Gate、Always、conditional accuracy、coverage、selective risk；条件准确率不可换成总体准确率。\n')

# Existing author project for polishing: different entrypoint, reachable input, macros,
# a genuine author-guide reference, exact facts and redundant nonnumeric exposition.
p=init('E03-original','icml')
main=(p/'main.tex').read_text()
main=main[:main.index(r'\begin{abstract}')]+r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}'+'\n'+r'\input{sections/body.tex}'+'\n'+r'\bibliographystyle{icml2026}\bibliography{references}'+'\n'+r'\clearpage'+'\n'+proof+'\n'+r'\end{document}'+'\n'
main=main.replace(r'\newcommand{\AuthorPending}[1]{\textbf{AUTHOR INPUT PENDING:} #1}',r'\newcommand{\riskname}{selective risk}'+'\n'+r'\def\methodname{Gate}')
main=main.replace('AUTHOR INPUT PENDING: Manuscript Title','A Fixed Confidence Gate for Selective Classification').replace('AUTHOR INFORMATION PENDING','Anonymous submission')
main=main.replace(r'\icmltitlerunning{Manuscript Title}',r'\icmltitlerunning{Confidence Gate Evaluation}')
(p/'main.tex').unlink()
(p/'article.tex').write_text(main)
metadata=json.loads((p/'project.json').read_text());metadata['entrypoint']='article.tex';(p/'project.json').write_text(json.dumps(metadata,indent=2)+'\n')
original_body=body.replace(r'\AuthorPending{required for reproducibility}','missing and required for reproducibility')
original_body=original_body.replace(r'\label{pw:main-end}',r'''Formatting uses the official author guidelines~\cite{formatguide}.
Our revolutionary approach is guaranteed to work for any classification problem
and is significantly faster than all alternatives. This sentence is an unsupported
claim in the supplied draft, not an additional result.
In the context of the evaluation considered in this manuscript, the possibility of
rejecting predictions is an important aspect of the procedure that we consider.
We emphasize that reporting conditional accuracy is part of how the evaluation is
organized. The organization of the evaluation includes reporting coverage alongside
conditional accuracy, because the set of accepted predictions is relevant to how
the result should be read. In this manuscript we present a fixed gate, and this
gate is a procedure which is fixed rather than trained. This is the approach to the
presentation of the procedure that we use throughout the manuscript.
The purpose of the discussion is to explain the conditions under which readers
should interpret the experimental comparison. Readers should consider these
conditions when interpreting the comparison, and the comparison must be interpreted
with those conditions in mind. The range bound is a bound on the range, and its
purpose is to delimit that range, rather than to describe empirical improvements.
These points reiterate the stated definitions and observations without adding
another experimental result or an additional theoretical claim.
\label{pw:main-end}''')
(p/'sections/body.tex').write_text(original_body)
(p/'references.bib').write_text('''@misc{formatguide,
 title={ICML 2026 Author Instructions},
 year={2026},
 howpublished={\\url{https://icml.cc/Conferences/2026/AuthorInstructions}}
}
''')
(p/'author-assets.txt').write_text('Synthetic fixture author asset. Preserve during polishing and migration.\n')
revised=E/'outputs/E03-revised';shutil.copytree(p,revised)
revised_body=body.replace(r'\AuthorPending{required for reproducibility}','missing and required for reproducibility')
revised_body=revised_body.replace(r'\label{pw:main-end}',r'Formatting uses the official author guidelines~\cite{formatguide}.'+'\n'+r'\label{pw:main-end}')
(revised/'sections/body.tex').write_text(revised_body)
(revised/'delivery.md').write_text('保留 article.tex 与 sections/body.tex、多文件结构、作者宏和引用。删除重复解释及原稿不受证据支持的普遍性、速度措辞，未增添实验或理论结论。原工程保留。仅合成评测，不是可投稿研究论文。\n')

p=init('E04-tai-migrated','tai')
orig=E/'outputs/E03-revised'
shutil.copytree(orig/'sections',p/'sections',dirs_exist_ok=True)
shutil.copy2(orig/'references.bib',p/'references.bib')
shutil.copy2(orig/'author-assets.txt',p/'author-assets.txt')
impact='''Abstaining from uncertain predictions can change both the set of cases a system
handles and the information available to a decision maker. The synthetic fixture
here illustrates why conditional accuracy should be considered together with
coverage: a higher accuracy among accepted cases does not establish better
performance across all cases. This distinction may help researchers design
evaluations that expose the tradeoff between selective prediction and abstention.
The observations do not support deploying the gate in a particular application.
Only a fixed classifier, a single baseline, and synthetic data are considered;
distribution shift, statistical uncertainty, latency, and consequences for affected
people remain unevaluated. Application-specific evidence and an assessment of
rejected cases would be needed before making deployment decisions.'''
main=r'''\documentclass[journal]{IEEEtai}
\usepackage{graphicx,amsmath,amssymb,url}
\newtheorem{PWTheorem}{Theorem}
\newcommand{\riskname}{selective risk}
\def\methodname{Gate}
\title{A Fixed Confidence Gate for Selective Classification}
\author{Anonymous submission}
\markboth{Confidence Gate}{Anonymous submission}
\begin{document}\maketitle
\begin{abstract}
'''+abstract+'\n'+r'''\end{abstract}
\begin{IEEEImpStatement}
'''+impact+'\n'+r'''\end{IEEEImpStatement}
\begin{IEEEkeywords}Selective classification, abstention, evaluation, coverage\end{IEEEkeywords}
\input{sections/body.tex}
\bibliographystyle{IEEEtran}\bibliography{references}
'''+proof+'\n'+r'\end{document}'+'\n'
(p/'main.tex').write_text(main)
(p/'delivery.md').write_text('从 ICML 2026 的匿名多文件稿迁移到当前核验的 TAI Regular 双匿名工程。只替换目标格式的入口与官方类，保留原正文、公式、宏、引用、标签和作者资产。新增影响声明解释条件准确率与覆盖率对判读的意义，未把可能用途写成部署收益。原工程未修改；未通过任何外部投稿系统验证。\n')
(E/'execution-mode.json').write_text(json.dumps({'E01':'independent agent','E02':'root-assisted, explicitly authored output, not blind model benchmark','E03':'root-assisted, explicitly authored output','E04':'root-assisted, explicitly authored output','E05':'independent agent, actual executor differs from frozen initial plan; raw request only, rubric withheld','fixture_status':'synthetic, not research findings'},ensure_ascii=False,indent=2)+'\n')
