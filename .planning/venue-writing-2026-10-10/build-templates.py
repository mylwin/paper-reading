from pathlib import Path
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[2]/'.claude/skills/paper-writing-shared'
ASSETS=ROOT/'assets/templates'
configs={
 'aaai':('2027','https://aaai.org/authorkit27/','https://aaai.org/conference/aaai/aaai-27/submission-instructions/',7,7,True),
 'ijcai':('2026','https://www.ijcai.org/sites/default/files/Guidelines/FormattingGuidelines-IJCAI-ECAI-26.zip','https://www.ijcai.org/authors_kit',7,None,True),
 'icml':('2026','https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip','https://icml.cc/Conferences/2026/AuthorInstructions',8,9,True),
 'neurips':('2026','https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip','https://neurips.cc/Conferences/2026/MainTrackHandbook',9,10,True),
 'iclr':('2027','https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip','https://iclr.cc/Conferences/2027/AuthorGuidelines',9,10,True),
 'colt':('2026','https://learningtheory.org/colt2026/colt2026-style.zip','https://learningtheory.org/colt2026/cfp.html',12,None,True),
 'jmlr':('2026','https://github.com/JmlrOrg/jmlr-style-file','https://www.jmlr.org/format/format.html',None,None,False),
 'tmlr':('2026','https://github.com/JmlrOrg/tmlr-style-file','https://www.jmlr.org/tmlr/author-guide.html',None,None,True),
 'tpami':('2026','https://template-selector.ieee.org/','https://www.computer.org/digital-library/journals/tp/cfp-ieee-pattern-analysis-machine-intelligence',None,None,None),
 'tnnls':('2026','https://template-selector.ieee.org/','https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls',None,None,True),
 'tai':('2026','https://template-selector.ieee.org/','https://cis.ieee.org/publications/ieee-transactions-on-artificial-intelligence/information-for-authors-tai',None,None,True),
}

body=r'''\section{Introduction}
\label{sec:introduction}
\AuthorPending{Research problem, closest work, exact gap, contributions and evidence.}
This project is a formatting skeleton, not a research result. The citation
\cite{formatguide} points to the official author guidance solely to demonstrate bibliography formatting.

\section{Method and Analysis}
\label{sec:method}
\AuthorPending{Definitions, design decisions, assumptions, algorithm inputs and outputs.}
The following identity is a typesetting demonstration only:
\begin{equation}
\label{eq:identity}
\begin{aligned}
 (a+b)^2 &= a^2 + 2ab + b^2,\\
 \sum_{i=1}^{n}(x_i-\bar x)^2
 &= \sum_{i=1}^{n}x_i^2-n\bar x^2.
\end{aligned}
\end{equation}
\begin{PWTheorem}[Typesetting demonstration]
For a real number $x$, the square $x^2$ is nonnegative.
\end{PWTheorem}
\textit{Proof (format example).} The statement follows from the ordering of the real numbers.
The displayed theorem is not a contribution of the manuscript.

ALGORITHM

\section{Experiments or Further Analysis}
\label{sec:results}
\AuthorPending{Provide real evidence and conditions. Replace this section for theory-only papers.}
Equation~\ref{eq:identity}, Figure~\ref{fig:single}, Figure~\ref{fig:wide}, and
Table~\ref{tab:conditions} demonstrate cross-references, not measurements.
\begin{figure}[htbp]
\centering
\fbox{\parbox{0.86\linewidth}{\centering DRAFT PLACEHOLDER: insert a real figure with a source and meaningful caption.}}
\caption{Figure formatting example; no experimental result is represented.}
\label{fig:single}
\end{figure}
\begin{figure*}[t]
\centering
\fbox{\parbox{0.90\textwidth}{\centering DRAFT PLACEHOLDER: wide figure across the available page width.}}
\caption{Wide figure formatting example; replace with author-provided evidence.}
\label{fig:wide}
\end{figure*}
\begin{table}[htbp]
\centering
\caption{Reporting structure only; values have not been supplied.}
\label{tab:conditions}
\begin{tabular}{ll}
\hline
Item & Status\\
\hline
Data and task & Pending\\
Comparison conditions & Pending\\
Measurements and uncertainty & Pending\\
\hline
\end{tabular}
\end{table}

\section{Limitations and Conclusion}
\AuthorPending{State supported conclusions and claim-specific boundaries.}
\label{pw:main-end}
'''
abstract=r'''\begin{abstract}
AUTHOR INPUT PENDING: write a single-paragraph abstract from the actual research problem,
method or theoretical contribution, key evidence, and scope. This template contains
no research findings and cannot be submitted as a manuscript.
\end{abstract}
'''
common=r'''\usepackage{amsmath,amssymb}
\newtheorem{PWTheorem}{Theorem}
\newcommand{\AuthorPending}[1]{\textbf{AUTHOR INPUT PENDING:} #1}
'''
algorithm=r'''\begin{algorithm}[htbp]
\caption{Pseudocode formatting example only}
\label{alg:format}
\begin{algorithmic}[1]
\REQUIRE Author-supplied inputs
\ENSURE Author-supplied outputs
\STATE Replace this line with the actual method.
\end{algorithmic}
\end{algorithm}'''
colt_alg=r'''\begin{algorithm}[htbp]
\caption{Pseudocode formatting example only}
\label{alg:format}
\KwIn{Author-supplied inputs}
\KwOut{Author-supplied outputs}
Replace this line with the actual method\;
\end{algorithm}'''
ieee_alg=r'''\begin{figure}[htbp]
\begin{algorithmic}[1]
\REQUIRE Author-supplied inputs
\ENSURE Author-supplied outputs
\STATE Replace this line with the actual method.
\end{algorithmic}
\caption{IEEE pseudocode formatting example, in a figure rather than a custom algorithm float.}
\label{alg:format}
\end{figure}'''

def main_text(venue,year,mode):
 final=mode=='final';anon='Anonymous submission' if not final else 'AUTHOR INFORMATION PENDING'
 title='AUTHOR INPUT PENDING: Manuscript Title'
 head='% Derived skeleton; official class/style files are unmodified.\n'
 if venue=='aaai':
  head+=r'\documentclass[letterpaper]{article}'+'\n'+rf'\usepackage[{"" if final else "submission"}]{{aaai2027}}'+'\n'+r'''\usepackage[hyphens]{url}
\usepackage{graphicx,natbib,caption,algorithm,algorithmic}
\urlstyle{rm}\def\UrlFont{\rm}\frenchspacing
\pdfinfo{/TemplateVersion (2027.1)}
'''+common+rf'\title{{{title}}}\author{{{anon}}}\affiliations{{{anon}}}'+'\n'+r'\begin{document}\maketitle'+'\n'
  tail=r'\bibliography{references}'+'\n'+r'''% Complete each item in the official reproducibility checklist; do not invent answers.
\clearpage
\input{ReproducibilityChecklist.tex}
'''
 elif venue=='icml':
  head+=r'\documentclass{article}\usepackage{microtype,graphicx,booktabs,hyperref}'+'\n'+rf'\usepackage{ "[accepted]" if final else "" }{{icml2026}}'+'\n'+r'\usepackage{algorithm,algorithmic}'+ '\n'+common+r'\providecommand{\theHalgorithm}{\arabic{algorithm}}'+'\n'+rf'\icmltitlerunning{{Manuscript Title}}'+'\n'+r'\begin{document}\twocolumn['+'\n'+rf'\icmltitle{{{title}}}'+'\n'+r'\begin{icmlauthorlist}\icmlauthor{AUTHOR INFORMATION PENDING}{pending}\end{icmlauthorlist}'+'\n'+r'\icmlaffiliation{pending}{AUTHOR INFORMATION PENDING}\icmlcorrespondingauthor{AUTHOR INFORMATION PENDING}{author-input-pending}\icmlkeywords{Author input pending}\vskip 0.3in]'+'\n'+r'\printAffiliationsAndNotice{}'+'\n'
  tail=r'\section*{Impact Statement}\AuthorPending{Describe the actual impacts without making unsupported assurances.}'+'\n'+r'\bibliographystyle{icml2026}\bibliography{references}'+'\n'
 elif venue=='neurips':
  head+=r'\documentclass{article}'+'\n'+rf'\usepackage[main{",final" if final else ""}]{{neurips_2026}}'+'\n'+r'\usepackage[T1]{fontenc}\usepackage{graphicx,hyperref,xcolor,algorithm,algorithmic}'+'\n'+common+rf'\title{{{title}}}\author{{AUTHOR INFORMATION PENDING}}'+'\n'+r'\begin{document}\maketitle'+'\n'
  tail=r'\bibliographystyle{plainnat}\bibliography{references}'+'\n'
 elif venue=='iclr':
  head+=r'\documentclass{article}\usepackage{iclr2027_conference,times}'+'\n'+r'\usepackage{graphicx,hyperref,url,algorithm,algorithmic}'+'\n'+common+rf'\title{{{title}}}\author{{AUTHOR INFORMATION PENDING}}'+'\n'+(r'\iclrfinalcopy'+'\n' if final else '')+r'\begin{document}\maketitle'+'\n'
  tail=r'\subsection*{AI use statement}\AuthorPending{Disclose actual AI use according to the ICLR 2027 author policy, including task scope and author verification; do not invent use or non-use.}'+'\n'+r'\subsection*{Reproducibility statement}\AuthorPending{Link to actual experimental details and complete proofs; omit unsupported promises.}'+'\n'+r'\bibliographystyle{iclr2027_conference}\bibliography{references}'+'\n'
 elif venue=='ijcai':
  head+=r'\documentclass{article}\usepackage{ijcai26,times,graphicx,amsmath,amssymb,amsthm,booktabs,algorithm,algorithmic,url}'+'\n'+r'\pdfinfo{/TemplateVersion (IJCAI.2026.0)}'+'\n'+common+rf'\title{{{title}}}\author{{{anon}}}'+'\n'+r'\begin{document}\maketitle'+'\n'
  tail=r'\bibliographystyle{named}\bibliography{references}'+'\n'
 elif venue=='colt':
  head+=rf'\documentclass[{"final" if final else "anon"},12pt]{{colt2026}}'+'\n'+r'\usepackage{times}'+'\n'+common+rf'\title[Manuscript Title]{{{title}}}'+'\n'+r'\coltauthor{\Name{AUTHOR INFORMATION PENDING}\Email{TBD}\addr TBD}'+'\n'+r'\begin{document}\maketitle'+'\n'
  tail=r'\bibliography{references}'+'\n'
 elif venue=='jmlr':
  head+=r'\documentclass[twoside,11pt]{article}\usepackage{jmlr2e}'+'\n'+r'\usepackage{algorithm,algorithmic}'+'\n'+common+r'\jmlrheading{TBD}{TBD}{TBD}{TBD}{TBD}{TBD}{TBD}'+'\n'+r'\ShortHeadings{Manuscript Title}{Authors pending}\firstpageno{1}'+'\n'+r'\begin{document}'+ '\n'+rf'\title{{{title}}}'+ '\n'+r'\author{\name AUTHOR INFORMATION PENDING\email author-input-pending\\\addr AUTHOR INFORMATION PENDING}\editor{EDITOR INFORMATION PENDING}\maketitle'+'\n'
  tail=r'\acks{\AuthorPending{Provide actual funding and competing-interest disclosures.}}'+'\n'+r'\bibliography{references}'+'\n'
 elif venue=='tmlr':
  head+=r'\documentclass[10pt]{article}'+ '\n'+rf'\usepackage{ "[accepted]" if final else "" }{{tmlr}}'+'\n'+r'\usepackage{graphicx,hyperref,url,algorithm,algorithmic}'+'\n'+common+rf'\title{{{title}}}'+ '\n'+r'\author{\name AUTHOR INFORMATION PENDING\email author-input-pending\\\addr AUTHOR INFORMATION PENDING}'+'\n'+r'\def\month{MM}\def\year{YYYY}\def\openreview{AUTHOR INPUT PENDING: OpenReview link}'+'\n'+r'\begin{document}\maketitle'+'\n'
  tail=r'\bibliographystyle{tmlr}\bibliography{references}'+'\n'
 else:
  options='lettersize,journal' if venue=='tpami' else 'journal'
  cls='IEEEtai' if venue=='tai' else 'IEEEtran'
  head+=rf'\documentclass[{options}]{{{cls}}}'+'\n'+r'\usepackage{graphicx,algorithmic}'+'\n'+common+rf'\title{{{title}}}\author{{{anon if venue in ["tai","tnnls"] else "AUTHOR INFORMATION PENDING"}}}'+'\n'+r'\begin{document}\maketitle'+'\n'
  tail=r'\bibliographystyle{IEEEtran}\bibliography{references}'+'\n'
 afterabstract=''
 if venue in ['jmlr']:
  afterabstract=r'\begin{keywords}Author-supplied keywords\end{keywords}'+'\n'
 if venue in ['tpami','tnnls','tai']:
  afterabstract=r'\begin{IEEEkeywords}AUTHOR INPUT PENDING: keyword one, keyword two, keyword three, keyword four\end{IEEEkeywords}'+'\n'
 if venue=='tai':
  afterabstract=r'\begin{IEEEImpStatement}\AuthorPending{Provide a 100--150 word impact statement grounded in the actual research.}\end{IEEEImpStatement}'+'\n'+afterabstract
 appendix=''
 if venue not in ['aaai','ijcai']:
  appendix=r'''\appendix
\section{Supporting Details}
\label{app:supporting}
\AuthorPending{Complete derivations, additional settings, and evidence referenced in the main text.}
This appendix is a formatting placeholder. It refers to Section~\ref{sec:method} and
Equation~\ref{eq:identity}; no omitted research result has been assumed.
'''
 if venue in ['icml','iclr','neurips']: appendix=r'\clearpage'+'\n'+appendix
 if venue=='neurips': appendix+=r'\input{checklist.tex}'+'\n'
 return head+abstract+afterabstract+r'\input{sections/body.tex}'+'\n'+tail+appendix+r'\end{document}'+'\n'

registry={'schema_version':1,'verified_on':'2026-10-11','corpus_window':[2024,2025],'semantic_corpus_status':'see each venue profile and evidence manifest; never infer completion from PDF count','venues':{}}
for venue,(year,source,guide,limit,flimit,anonymous) in configs.items():
 base=ASSETS/venue/year;off=base/'official'
 files=[p for p in off.glob('*') if p.suffix in ['.sty','.cls','.bst'] or p.name in ['math_commands.tex','checklist.tex','ReproducibilityChecklist.tex','LICENSE','LICENSE.txt','README','IEEEtran-README']]
 available=bool(files)
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 modes={}
 for mode in ['submission','final']:
  dest=base/'project'/mode
  if available:
   (dest/'sections').mkdir(parents=True,exist_ok=True)
   for p in files:shutil.copy2(p,dest/p.name)
   b=body.replace('ALGORITHM',colt_alg if venue=='colt' else ieee_alg if venue in ['tpami','tnnls','tai'] else algorithm)
   if venue=='colt': b=b.replace(r'\section{Limitations and Conclusion}',r'\FloatBarrier'+'\n'+r'\section{Limitations and Conclusion}')
   if venue=='tmlr': b=b.replace(r'\begin{algorithmic}[1]',r'\let\AND\relax'+'\n'+r'\begin{algorithmic}[1]')
   (dest/'sections/body.tex').write_text(b)
   (dest/'main.tex').write_text(main_text(venue,year,mode))
   if venue in ['aaai','ijcai']:
    supplement=main_text(venue,year,mode).replace('Manuscript Title','Supplementary Material').replace('sections/body.tex','sections/supplement.tex').replace(r'\input{ReproducibilityChecklist.tex}','')
    (dest/'supplement.tex').write_text(supplement)
    (dest/'sections/supplement.tex').write_text(r'\section{Supporting Details}\label{app:supporting}'+'\n'+r'\AuthorPending{Insert complete proofs, details and evidence referenced in the main paper.}'+'\n'+r'The official guidance \cite{formatguide} is cited only to demonstrate bibliography formatting.'+'\n')
   (dest/'references.bib').write_text('@misc{formatguide,\n author = {{Official author guidelines}},\n title = {Author and formatting instructions},\n year = {'+year+'},\n url = {'+guide+'},\n note = {Template-format demonstration only; replace for the research manuscript}\n}\n')
  modes[mode]={'available':available,'directory':'project/'+mode,'anonymous':anonymous if mode=='submission' else False,'main_page_limit':limit if mode=='submission' else flimit,'page_limit_status':'verified' if (limit if mode=='submission' else flimit) else 'requires-target-stage-verification','counting':'Main text unless journal-specific rules explicitly include references','engine':'pdflatex','article_type':'original-research','full_manuscript_page_limit':15 if venue in ['tai','tnnls'] else None,'max_pdf_megabytes':(50 if mode=='submission' else 20) if venue=='icml' else None}
 spec={'venue':venue,'version':year,'verified_on':'2026-10-11','template_source':source,'author_guidelines':guide,'official_files':hashes,'modes':modes,'format_status':'assets-acquired-compilation-pending' if available else 'official-assets-pending','limitations':['Compilation is not submission compliance. Publication metadata and actual results must be provided.','Unknown page limits or anonymity policies are not assumed compliant.']}
 if venue=='aaai':spec['limitations'].append('Official style internal date is 2027/05/04 although downloaded on 2026-10-10; preserve as distributed and recheck target author kit before submission.')
 if venue in ['aaai','ijcai']:spec['limitations'].append('Supplementary appendix is scaffolded in supplement.tex and compiled separately; it is not attached to main.tex.')
 if venue=='tpami':spec['guidelines_status']='official-scope-verified; full-current-submission-guide-pending'
 if venue=='iclr':spec['guidelines_status']='official-2027-author-guidelines-and-AI-policy-verified';spec['rules']={'required_sections':['AI use statement'],'ai_policy':'https://iclr.cc/Conferences/2027/AIPolicyForAuthors'}
 if venue=='jmlr':spec['rules']={'abstract_max_words':200,'length_guidance':{'main_above_35':'may delay review, not a hard rejection limit','including_appendix_above_50':'justify in cover letter'}}
 if venue in ['tpami','tnnls','tai']:
  tid={'tpami':199,'tnnls':262,'tai':542}[venue]
  spec['template_source']=f'https://template-selector.ieee.org/api/ieee-template-selector/template/{tid}/download'
  spec['template_selector']='https://template-selector.ieee.org/'
  spec['snapshot_year']=year
  spec['dependency_sources']={'IEEEtran.bst':'https://tug.ctan.org/macros/latex/contrib/IEEEtran/bibtex/IEEEtran.bst'}
  if venue=='tpami':spec['dependency_sources']['IEEEtran.cls']='https://tug.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran.cls'
 if venue in ['jmlr','colt','neurips']:spec['dependency_sources']={'plainnat.bst':'TeX Live 2026 natbib distribution','abbrvnat.bst':'TeX Live 2026 natbib distribution'}
 if venue=='tnnls':spec['rules']={'keywords_min':4,'keywords_max':5,'regular_pages':{'normal':10,'maximum':15,'counting':'full manuscript excluding supplementary material'}}
 if venue=='tai':spec['rules']={'title_max_words':15,'running_head_max_characters':25,'abstract_max_words':250,'keywords_min':3,'keywords_max':6,'impact_statement_words':[100,150],'regular_pages':{'normal':10,'maximum':15,'counting':'full manuscript according to author guidelines; extra pages have charges'}}
 base.mkdir(parents=True,exist_ok=True)
 (base/'spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
 registry['venues'][venue]={'profile':'venues/'+venue+'.md','latest':year,'versions':{year:str((base/'spec.json').relative_to(ROOT))}}
(ROOT/'registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
print('Template registry and skeletons generated')
