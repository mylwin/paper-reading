from pathlib import Path
import re,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[2]
stem='MUON2_Boosting_MUON_via_Adaptive_Second_Moment_Preconditioning'
d=ROOT/'03-notes/2026-10'/stem
p=d/'精读.md';s=p.read_text()
s=s.replace('若以谱的均值、标准差之比定义变异系数，则本笔记进一步得到：','若以谱的标准差除以均值定义变异系数 $\\nu_z$，则本笔记进一步得到：')
s=s.replace('\\mathrm{CV}(z)','\\nu_z')
s=s.replace('小值区域有 $\\phi(x)\\approx ax$，','小值区域主要由线性项 $ax$ 支配，')
s=s.replace('一般有：\n\n$$\n\\mathcal{P}(H_t)\\neq\\mathcal{P}(M_t),\n$$\n\n其中 $\\mathcal{P}$ 表示精确极因子。','一般不能将 $\\mathcal{P}(H_t)$ 等同于 $\\mathcal{P}(M_t)$；其中 $\\mathcal{P}$ 表示精确极因子。')
s=s.replace('Muon也是','Muon 也是')
# Normalize Chinese spacing around inline math, including punctuation.
fixed=[]
for line in s.splitlines():
 if '$$' not in line:
  line=re.sub(r'([\u4e00-\u9fff，。；：、（）])(\$[^$\n]+\$)',r'\1 \2',line)
  line=re.sub(r'(\$[^$\n]+\$)([\u4e00-\u9fff，。；：、（）])',r'\1 \2',line)
 fixed.append(line)
s='\n'.join(fixed)+'\n'
p.write_text(s)
src=ROOT/'02-markdown/2026-10/images'/stem
mapping={1:'图1a',2:'图1b',3:'图1c',4:'图1d',5:'图2a',6:'图2b',7:'图2c',8:'图2d',9:'图3',10:'图4',11:'图5',12:'图6',13:'图7a',14:'图7b',15:'图8',16:'图9',17:'图10',18:'图11',19:'图12',20:'图13'}
source_lines=['# 图片来源与编号映射','', '本目录保留本地 MinerU 解析的全部20个图片资产；内容未改动，编号按解析出现顺序，与论文整图编号不同。','', '| 笔记图片 | 原论文图号 | SHA-256 |','|---|---|---|']
for i in range(1,21):
 dest=d/'images'/f'fig{i}.jpg';shutil.copy2(src/f'fig{i}.jpg',dest)
 digest=hashlib.sha256(dest.read_bytes()).hexdigest()
 assert digest==hashlib.sha256((src/f'fig{i}.jpg').read_bytes()).hexdigest()
 source_lines.append(f'| fig{i}.jpg | {mapping[i]} | {digest} |')
source_lines+=['','## 原始 CDN 地址','', '以下转录原解析目录的来源表，便于整个笔记目录独立保存时溯源。','']
raw=(src/'_sources.md').read_text();source_lines.append(raw[raw.index('| 本地文件'):])
(d/'images/_sources.md').write_text('\n'.join(source_lines))
# Check abstract against the supplied parsed version with three PDF-verified OCR repairs.
raw=(ROOT/'02-markdown/2026-10'/f'{stem}.md').read_text()
abstract=raw.split('## Abstract')[1].split('## 1 Introduction')[0].strip()
abstract=abstract.replace('MUON<sup>2</sup>','MUON²').replace('Newton– Schulz','Newton–Schulz').replace('Adamstyle','Adam-style').replace('Mixtureof-Experts','Mixture-of-Experts')
quoted=s.split('### 英文摘要（v2 PDF 原文，完整转录）')[1].split('> 转录')[0].strip()
assert abstract==quoted,'English abstract differs'
commands=sorted(set(re.findall(r'\\([A-Za-z]+)',s)))
allowed={'Sigma','nabla','mathcal','beta','epsilon','frac','sqrt','eta','Vert','top','phi','min','sum','nu','times'}
assert set(commands)<=allowed,commands
for bad in [r'\tag',r'\big',r'\quad',r'\qquad',r'\boxed',r'\underbrace',r'\overset',r'\underset',r'\|',r'\(',r'\)']:
 assert bad not in s,bad
assert s.count('$$')%2==0
math_count=0
for block,inline in re.findall(r'\$\$(.*?)\$\$|(?<!\$)\$([^\n$]+)\$(?!\$)',s,re.S):
 expr=block or inline;math_count+=1
 assert expr.count('{')==expr.count('}'),expr
 assert not inline or inline==inline.strip(),expr
for i,line in enumerate(s.splitlines(),1):
 if '$$' not in line:assert line.count('$')%2==0,(i,line)
 if line.startswith('|'):
  for expr in re.findall(r'\$([^$]+)\$',line):assert '|' not in expr
links=[]
for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',s):
 if '://' not in target:
  assert (p.parent/target).exists(),target;links.append(target)
images=re.findall(r'!\[[^\]]*\]\((images/[^)]+)\)',s)
assert len(images)==20 and len(set(images))==20
for placeholder in ['[SCORE]','[作者列表]','[论文英文摘要原文]','[流畅准确']:
 assert placeholder not in s
report={'date':'2026-10-19','source_version':'2604.09967v2','english_abstract_verified':True,
 'abstract_OCR_repairs':['Newton– Schulz → Newton–Schulz','Adamstyle → Adam-style','Mixtureof-Experts → Mixture-of-Experts'],
 'image_assets':20,'image_hashes_match_source':True,'all_note_local_links_resolve':True,
 'formula_count':math_count,'formula_commands':commands,'chinese_char_count':len(re.findall(r'[\u4e00-\u9fff]',s)),
 'LLM_training_reproduced':False}
Path(__file__).with_name('validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
