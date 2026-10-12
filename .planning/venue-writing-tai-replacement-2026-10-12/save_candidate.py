import json, pathlib, hashlib

ROOT=pathlib.Path('/Users/myl/Documents/paper')
OLD=ROOT/'07-research/2026-10-10-顶会顶刊写作风格调研_证据'
OUT=ROOT/'07-research/2026-10-12-顶会顶刊写作风格调研_证据/tai-replacement'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data):
 p=OUT/name
 if p.exists(): raise FileExistsError(p)
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
manifest=json.loads((OLD/'corpus/tai/manifest.json').read_text())
existing={p['doi'].lower() for p in manifest['papers']}
existing_first={p['authors'].split(',')[0].strip().casefold() for p in manifest['papers'] if p['authors']}
sources=['tai-openalex.json','tai-openalex-expanded.json','tai-openalex-page2.json']
frame={}
for name in sources:
 for p in json.loads((OLD/'sources/tai'/name).read_text())['results']:
  doi=(p.get('doi')or'').removeprefix('https://doi.org/').lower()
  if doi and doi not in existing and p.get('publication_year')==2024:
   frame[doi]={'doi':doi,'title':p['display_name'],'openalex_id':p['id'],'locator_year':2024,'locator_date':p['publication_date'],
     'authors':[a['author']['display_name'] for a in p['authorships']],
     'first_author_new': bool(p['authorships']) and p['authorships'][0]['author']['display_name'].casefold() not in existing_first,
     'locations':p['locations'], 'frame_source':str((OLD/'sources/tai'/name).relative_to(ROOT)),
     'scope':'Locator metadata only; not an independent PDF/version or article-type audit.'}
write('original-frame-unused-2024.json', {'date':'2026-10-12','ordering':'ascending normalized DOI; preserve original source snapshot row IDs',
  'source_checksums':{n:sha(OLD/'sources/tai'/n) for n in sources},'candidates':[frame[k] for k in sorted(frame)]})
cf=json.loads((OUT/'snn-slimming-crossref.json').read_text())['message']
authors=[a['given']+' '+a['family'] for a in cf['author']]
selected='10.1109/tai.2024.3352533'
selection={
 'date':'2026-10-12','status':'qualified_candidate_pending_parent_registration',
 'original_slot':{'id':'tai-2024-03','year':2024,'split':'derivation','original_doi':'10.1109/tai.2024.3372796','original_title':'Front Cover','reason':'nonresearch front cover; no author; original lock preserved'},
 'selection_policy':'Reuse original archived public-locator frame. Sort unused 2024 entries by normalized DOI. Exclude editorial/front matter/review by content-type screen; require a novel original-research paper, unique existing first author, publisher 2024 print identity, and independently confirmed accepted/published full-text version. Among otherwise eligible rows, prefer confirmable accepted/published versions, as in the original sampling method. Do not inspect holdout outcomes or select based on writing-style findings.',
 'screen_log':[
  {'doi':'10.1109/tai.2023.3342563','title':'Guest Editorial: Special Issue on Physics-Informed Machine Learning','decision':'exclude','reason':'guest editorial; title-level type exclusion, no style analysis'},
  {'doi':'10.1109/tai.2023.3345689','title':'Front Cover','decision':'exclude','reason':'front matter'},
  {'doi':'10.1109/tai.2023.3345691','title':'IEEE Transactions on Artificial Intelligence Publication Information','decision':'exclude','reason':'front matter'},
  {'doi':'10.1109/tai.2023.3345693','title':'Table of Contents','decision':'exclude','reason':'front matter'},
  {'doi':'10.1109/tai.2024.3350543','title':'Semisupervised Transfer Boosting (SS-TrBoosting)','decision':'defer_version_confirmation','reason':'Crossref confirms 2024 print vol.5(7),3431–3444 and original-research identity; arXiv v1 authors/title match and author journal reference links DOI, but actual PDF lacks acceptance/publication/venue/DOI header. Cannot independently label this deposited edition accepted/published merely from related DOI. Candidate retained, not falsely excluded as nonresearch.'},
  {'doi':'10.1109/tai.2024.3351798','title':'A Survey on Verification and Validation, Testing and Evaluations of Neurosymbolic Artificial Intelligence','decision':'exclude','reason':'survey, outside default original-research corpus'},
  {'doi':selected,'title':cf['title'][0],'decision':'propose','reason':'first independently confirmable accepted-original-research full text in this documented deterministic screen; novel neuron/channel slimming methods and empirical tests, new first author, 2024 print identity'}
 ],
 'candidate':{'title':cf['title'][0],'authors':authors,'doi':selected,'publisher_url':cf['resource']['primary']['URL'],'print_year':2024,'print_month':8,'volume':'5','issue':'8','pages':'3823–3831','early_online_date':'2024-01-11',
  'year_evidence':'Publisher-deposited Crossref published-print [[2024,8]], independent institutional 2024 publication identity. DOI year not used as year proof.',
  'article_type':'peer-reviewed original research; independent institution Article + actual Methods/Experiments content; not editorial/survey/front cover',
  'first_author_new': True,'already_in_existing_50':False,'topic':'spiking neural networks; biologically inspired structural sparsification; empirical computer-science methods',
  'topic_balance_note':'Preserves unique first-author diversity and adds a neuromorphic representation, rather than selecting exclusively optimizer papers. Existing TAI corpus includes a 2025 neuromorphic Sudoku paper and several pruning papers; this is additional coverage, not a claim of uniformly balanced topic frequency.',
  'version':'accepted_author_manuscript','version_evidence':'Institution Access to Document states accepted author manuscript; actual PDF cover states Peer reviewed version and accepted author manuscript of an article published in TAI, with matching DOI/title/all 8 authors. OpenAlex submittedVersion locator label is inaccurate for this actual deposit.',
  'source_url':'https://pure-oai.bham.ac.uk/ws/files/219032020/LiY24.pdf','pdf_file':'snn-slimming-accepted-author.pdf','pdf_pages':10,'paper_pages':9,'cover_pages':1,'bytes':(OUT/'snn-slimming-accepted-author.pdf').stat().st_size,'sha256':sha(OUT/'snn-slimming-accepted-author.pdf'),
  'source_page':'https://research.birmingham.ac.uk/en/publications/efficient-structure-slimming-for-spiking-neural-networks/',
  'final_publisher_pdf_byte_equivalence':'not checked; never claimed; current deposit has institutional cover dated 2026-10-12'},
 'original_files_modified':False,'holdout_slots_modified':False,'registered_in_corpus':False,'download_or_full_review_counts_modified':False,
 'primary_sources':[{'url':'https://api.crossref.org/works/10.1109%2FTAI.2024.3352533','saved':'snn-slimming-crossref.json','sha256':sha(OUT/'snn-slimming-crossref.json')},
 {'url':'https://research.birmingham.ac.uk/en/publications/efficient-structure-slimming-for-spiking-neural-networks/','saved':'snn-repository.html','sha256':sha(OUT/'snn-repository.html')},
 {'url':'https://arxiv.org/abs/2412.03212','saved':'primary-page-results.json','purpose':'deferred alternative version/date inventory'}],
 'limitations':['Parent must approve/register a traceable slot-repair amendment before counting this candidate among the 550. Original manifest must remain as lock history; no holdout is changed.','Publisher Xplore and DOI page inaccessible through web tool; publisher-deposited Crossref plus matching institution accepted deposit establish identity/version, not current publisher PDF equivalence.','Institution permits private/noncommercial research use; PDF retained locally as evidence, not published or redistributed.','PDF extraction emitted xref/path reconstruction warnings, exit 0; all ten pages extracted and three key pages visually rendered. Warnings and PDFs retained.']
}
write('replacement-candidate.json',selection)

dimensions={
 'reader_argument':{'pages':'paper 1–3 (PDF 2–4)','observation':'Bio-inspired motivation is explained to broad AI readers before specialist SNN training/pruning taxonomies; target is sparse, useful deep SNN structure, not an optimizer convergence theorem.'},
 'title_abstract':{'pages':'paper 1 (PDF 2)','observation':'Title names mechanism and model family. Abstract proceeds biological premise → dense-structure gap → neuron/channel methods with regrowth → datasets → retention/accuracy trade-off. Separate Impact Statement and Index Terms are present; their mandatory status must come from official rules, not sample frequency.'},
 'introduction':{'pages':'paper 1–2 (PDF 2–3)','observation':'Builds practical computational/embedded-device motivation and ends with three contribution bullets: sparse method, quantitative memory/energy analysis, experiments. Recurring biological premise is a candidate for compression, not a venue requirement.'},
 'related_work':{'pages':'paper 2–4 (PDF 3–5)','observation':'Early standalone related work divided into Model training, Pruning method, Motivation; ANN vs SNN and unstructured vs structured pruning distinctions justify design choices. No universal TAI section-order claim.'},
 'method_theory':{'pages':'paper 4–5 (PDF 5–6)','observation':'Main text gives IF charge/fire/reset equations, surrogate function, pruning masks, BN/tdBN, sparsity loss, and two algorithms. Neuron variant prunes/regrows during training; channel variant pretrains with sparsity penalty, prunes, then fine-tunes. These distinct stage maps must survive polishing. No independent convergence theorem is claimed.'},
 'evidence':{'pages':'paper 6–8 (PDF 7–9)','observation':'Results sections separate settings, pruning granularity, noise, parameter/operation analysis, comparison. Tables expose baseline, retained/pruned configuration, accuracy, fine-tuned accuracy and resources; comparison table explicitly lists architecture/training/time steps/connectivity, enabling fairness bounds. Noise is specified Gaussian test corruptions, not a general robustness theorem.'},
 'discussion_limits':{'pages':'paper 7–9 (PDF 8–10)','observation':'No dedicated limitations section. Trade-offs are discussed near result comparisons: sparsity may be less than competing models although time steps lower. Conclusion returns biological motivation and deployment potential. Existing tested bounds belong in conclusions; no silent new hardware-deployment claim.'},
 'paragraph_language':{'pages':'paper 1–9 (PDF 2–10)','observation':'Numbered technical subsections and bold run-in labels aid scanning; explanatory prose is repeated and occasionally grammatically imprecise. Publication is not evidence that repeated generic claims or unqualified significantly are suitable style norms.'},
 'main_appendix':{'pages':'all paper 1–9 (PDF 2–10)','observation':'All equations, algorithm specifications and empirical tables are in nine-page accepted manuscript; references finish paper 9. No appendix/supplement reference or code link appears in actual full text; institution lists one manuscript asset. No separately referenced prose appendix was identified. Institutional cover is source provenance, not journal-format text.'},
 'actionable':{'pages':'paper 1,4–8 (PDF 2,5–9)','observation':'For an empirical sparse-model paper, maintain Abstract/Impact/Index Terms as required by official configuration, use a stage-specific method map, expose resource-performance trade-offs and comparison differences, and distinguish estimated energy from measured deployment. Transfer only under the same article type, not all TAI topics.'}
}
counterexamples=[
 {'id':'TAI-SNN-C1','pages':'paper 1 vs 7–8; PDF 2,8–9','finding':'Abstract about 10% retained parameters with <1% loss is supported by CIFAR10 channel retention 11.13% and 0.60 percentage-point loss. Table V CIFAR100 at 13.06% retained loses 10.06 points. Do not generalize the headline to every dataset/variant or silently treat percentage-point differences as relative percent.','action':'Scope the summary to dataset, architecture, variant and fine-tuning stage.'},
 {'id':'TAI-SNN-C2','pages':'paper 7 Table IV and paragraph D; PDF 8','finding':'Energy numbers derive from synaptic operations and a cited TrueNorth 26 pJ assumption; no documented hardware energy-measurement procedure or actual chip deployment. Large parameter reduction coexists with synaptic-operation counts near 28K across pruned channel settings.','action':'Name proxy/assumption and separate parameter-memory reductions from energy/latency evidence; do not rewrite as measured deployment.'},
 {'id':'TAI-SNN-C3','pages':'paper 7 paragraph E vs Table V paper 8; PDF 8–9','finding':'Prose describes 4.82 times the original; Table V connectivity is 4.82%, and Table IV 0.71M/14.72M = 4.82%. This is a unit conflict in the source, not a linguistic norm.','action':'Flag author clarification; do not preserve misleading multiplier or fabricate replacement experimental values.'},
 {'id':'TAI-SNN-C4','pages':'paper 2 vs 4–5; PDF 3,5–6','finding':'Broad from-scratch wording and pretrained wording are reconciled only by separating neuron dynamic training from channel sparse-pretrain/prune/fine-tune stages.','action':'Use variant/stage-specific descriptions rather than one uniform chronological claim.'},
 {'id':'TAI-SNN-C5','pages':'paper 8 Table V; PDF 9','finding':'Competing entries differ in architectures, training methods, time steps and connectivity. Some competitors have higher absolute accuracy or greater sparsity.','action':'Do not write universally superior at matched conditions; state observed trade-off and actual comparison settings.'},
 {'id':'TAI-SNN-C6','pages':'paper 2,6–8; PDF 3,7–9','finding':'Introduction says significantly, while inspected experimental text/tables do not report repeated-seed count, variance or significance testing.','action':'Avoid adding statistical-significance claims solely because the accepted paper uses that adjective.'}
]
review={
 'candidate_id':'tai-2024-03-replacement-proposal','doi':selected,'title':cf['title'][0],'review_date':'2026-10-12','split':'derivation',
 'registered_in_corpus':False,'review_mode':'Continuous semantic writing-style reading of actual PDF cover and all nine paper pages; no independent theorem verification, code execution or result reproduction.',
 'full_text_style_review_complete':True,'external_appendix_review_complete':True,
 'coverage':[{'pdf_pages':'1','sections':'institution provenance/rights/version cover; excluded from paper-style observations'},
  {'pdf_pages':'2–4','paper_pages':'1–3','sections':'abstract/impact/terms; introduction; all related work beginning'},
  {'pdf_pages':'5–7','paper_pages':'4–6','sections':'remaining motivation; all methods/equations/algorithms; experimental setup/results'},
  {'pdf_pages':'8–10','paper_pages':'7–9','sections':'all remaining experiments/discussion; conclusion; references'}],
 'pdf_sha256':sha(OUT/'snn-slimming-accepted-author.pdf'),'pdf_file':'snn-slimming-accepted-author.pdf','text_file':'snn-slimming-accepted-author.txt',
 'version':'accepted_author_manuscript','version_audit':'replacement-candidate.json',
 'supplement_audit':{'references_in_full_text':'none found; all paper pages continuously read','institution_asset_inventory':'one accepted manuscript link; no separate prose supplement on page','publisher_inventory':'Xplore inaccessible; no assertion of exhaustive unlisted publisher assets','result':'no referenced relevant appendix remains unread for the actual accepted deposit'},
 'visual_check':{'pdf_pages':[2,8,9],'files':['renders/pdf-p2.png','renders/pdf-p8.png','renders/pdf-p9.png'],'result':'title/author/front-matter and Tables IV/V visually confirmed; extraction warnings retained in rendering logs'},
 'ten_dimensions':dimensions,'counterexamples':counterexamples,
 'derived_rules':[
  {'id':'TAI-SNN-R1','kind':'corpus_observation','rule':'Separate biological motivation, algorithm implementation and empirically observed resource/accuracy trade-offs.','support':'paper 1–8','counterexamples':['TAI-SNN-C1','TAI-SNN-C2'],'condition':'empirical sparse/neuromorphic model manuscript','confidence':'high for this sample; low as venue-wide tendency; no frequency estimate'},
  {'id':'TAI-SNN-R2','kind':'editorial_advice','rule':'Preserve method-variant stage maps and report fair comparison conditions; use precise retention/accuracy units.','support':'paper 4–8','counterexamples':['TAI-SNN-C3','TAI-SNN-C4','TAI-SNN-C5'],'condition':'compression or resource comparison','confidence':'high evidence-bound advice; not an official TAI rule'}
 ],
 'original_rewrite_examples':[{'facts':'CIFAR10/VGG16 channel pruning+fine-tuning, Table V; 11.13% retained,91.71 baseline,91.11 fine-tuned','english':'On CIFAR-10, the fine-tuned VGG16 channel-pruning model retained 11.13% of the original parameters, with accuracy decreasing from 91.71% to 91.11% (0.60 percentage points).','reason':'Original evidence-faithful example bounds headline by dataset, architecture, stage and correct units.'},
 {'facts':'Table IV energy estimation via cited 26pJ synaptic cost; no measured hardware test','english':'We estimated energy from synaptic-operation counts using the cited per-operation cost; these estimates do not establish measured savings on a deployed device.','reason':'Editorial original sentence distinguishes proxy evidence from a deployment claim.'}],
 'holdout_validation':None,'confidence_note':'One independently read accepted original research candidate; contributes no claim about 50-sample TAI frequency or validated holdout generalization.'
}
write('candidate-semantic-review.json',review)
files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
write('candidate-files-sha256.json',{'date':'2026-10-12','files':files})
print('saved qualification, selection frame and continuous full-paper review; corpus unchanged')
