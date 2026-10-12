import json,hashlib,shutil,subprocess,sys
from pathlib import Path
root=Path.cwd();base=root/'07-research/2026-10-10-顶会顶刊写作风格调研_证据';out=root/'07-research/2026-10-12-顶会顶刊写作风格调研_证据'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

ledger=[]
for venue,ids in [('aaai',['aaai-2025-12','aaai-2024-22']),('tai',['tai-2025-16','tai-2025-01'])]:
 d=base/'corpus'/venue;rows=[json.loads(l) for l in (d/'records.jsonl').read_text().splitlines()]
 for pid in ids:
  review=json.loads((out/venue/f'{pid}.json').read_text());row=next(r for r in rows if r['id']==pid)
  pdf=review.get('pdf',review.get('pdf_path'))
  assert sha(root/pdf)==review['pdf_sha256']
  review.update(parent_inspection='Root inspected coverage, full10dimensions, counterexamples, version/supplement evidence and SHA; relies on designated reviewer for continuous reading, no duplicate full read',publication_identity_and_acquired_version=review.get('version_audit',review.get('version')))
  dest=d/'semantic-reviews'/f'{pid}.json';save(dest,review)
  complete=bool(review['full_text_style_review_complete'] and review['external_appendix_review_complete'])
  row.update(pdf_path=pdf,pdf_sha256=review['pdf_sha256'],full_text_style_review_complete=True,external_appendix_review_complete=complete,semantic_review_status='continuous full PDF and relevant scientific appendix reading' if complete else 'continuous full PDF reading; referenced external scientific supplement pending',reviewed_date='2026-10-12',semantic_review_path=str(dest.relative_to(root)))
  if venue=='tai':
   row.update(acquired_version=review['version_audit']['verified_version'],version_verification=review['version_audit']['evidence'],eligibility='accepted_author_manuscript_content_confirmed; supplementary reading pending')
  ledger.append(dict(id=pid,venue=venue,year=row['year'],split=row['split'],main_complete=True,full_and_appendix_complete=complete,review_path=str(dest.relative_to(root)),pdf_path=pdf,pdf_sha256=review['pdf_sha256']))
 (d/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

for pid in ['neurips-2024-2aada42a','neurips-2025-aac4a9fe']:
 p=base/'corpus/neurips/semantic-reviews'/f'{pid}.json';r=json.loads(p.read_text())
 ledger.append(dict(id=pid,venue='neurips',year=r['year'],split=r['split'],main_complete=True,full_and_appendix_complete=True,review_path=str(p.relative_to(root)),pdf_path=r['pdf_path'],pdf_sha256=r['pdf_sha256']))

# Replace only the confirmed nonresearch induction slot; preserve exact old manifest/record history.
d=base/'corpus/tai';records_path=d/'records.jsonl';manifest_path=d/'manifest.json'
before_manifest=manifest_path.read_bytes();before_records=records_path.read_bytes()
backup_manifest=d/'manifest.locked-before-2026-10-12-repair.json';backup_records=d/'records.before-2026-10-12-slot-repair.jsonl'
assert not backup_manifest.exists() and not backup_records.exists(),'Refuse repeating slot repair'
backup_manifest.write_bytes(before_manifest);backup_records.write_bytes(before_records)
manifest=json.loads(before_manifest);rows=[json.loads(l) for l in before_records.decode().splitlines()];old=next(r for r in rows if r['id']=='tai-2024-03')
assert old['year']==2024 and old['split']=='derivation' and old['title']=='Front Cover'
proposal=json.loads((out/'tai-replacement/replacement-candidate.json').read_text());c=proposal['candidate']
assert c['print_year']==2024 and c['article_type'].startswith('peer-reviewed original research') and c['version']=='accepted_author_manuscript'
assert c['doi'].lower() not in {r.get('doi','').lower() for r in rows}
source=out/'tai-replacement'/c['pdf_file'];assert sha(source)==c['sha256']
sys.path.insert(0,str(root/'.claude/skills/paper-daily/scripts'));from paper_config import sanitize_paper_title
stem=sanitize_paper_title(c['title']);cache=root/'.claude/skills/paper-writing-shared/tmp/corpus/tai';pdf=cache/'pdf'/f'{stem}.pdf';md=cache/'markdown'/f'{stem}.md'
pdf.parent.mkdir(parents=True,exist_ok=True);md.parent.mkdir(parents=True,exist_ok=True)
assert not pdf.exists() and not md.exists(),'Refuse overwriting existing user paper assets'
shutil.copy2(source,pdf)
text=subprocess.run(['pdftotext','-layout',str(pdf),'-'],capture_output=True,text=True,check=True).stdout
pages=[s for s in text.split('\f') if s.strip()];assert len(pages)==10
md.write_text('# '+c['title']+'\n\nAccepted author manuscript with an institutional provenance cover. PDF text extraction may lose formula/table layout; no image extraction performed.\n\n'+'\n\n'.join(f'## PDF page {i+1}\n\n```text\n{s}\n```' for i,s in enumerate(pages)))
new=dict(id='tai-2024-03-r1',selection_slot='tai-2024-03',replaces='tai-2024-03',venue='tai',year=2024,split='derivation',selection_locked='2026-10-12',original_slot_locked='2026-10-10',title=c['title'],authors=', '.join(c['authors']),doi=c['doi'],publication_url=c['publisher_url'],pdf_url=c['source_url'],source_index='original archived OpenAlex locator frame; deterministic DOI-order eligibility repair',version='accepted_author_manuscript',version_metadata_source='actual institutional PDF cover and matching institution/crossref',publication_verified='publisher-deposited Crossref print2024 plus institution Article/accepted manuscript declaration',eligibility='accepted_original_research_version_content_confirmed',paper_type='empirical sparse neuromorphic model method',type_status='continuous semantic content classification',provisional_stratum='learning',pdf_path=str(pdf.relative_to(root)),pdf_sha256=sha(pdf),markdown_path=str(md.relative_to(root)),page_count=10,paper_pages=9,acquisition_status='downloaded accepted-author PDF and text parsed',download_ok=True,parse_status='full_pdf_text_extracted',acquired_version='accepted_author_manuscript',version_verification=c['version_evidence'],full_text_style_review_complete=True,external_appendix_review_complete=True,semantic_review_status='continuous all10PDFpages writing-style review; no referenced prose appendix',reviewed_date='2026-10-12',selection_amendment='selection-amendments.json')
review=json.loads((out/'tai-replacement/candidate-semantic-review.json').read_text());review.update(id=new['id'],venue='tai',year=2024,registered_in_corpus=True,registration_date='2026-10-12',pdf_path=new['pdf_path'],markdown_path=new['markdown_path'],markdown_sha256=sha(md),publication_identity_and_acquired_version=c['version_evidence'],original_candidate_review_path=str((out/'tai-replacement/candidate-semantic-review.json').relative_to(root)),parent_inspection='Root inspected all10dimensions, exact version/DOI/year evidence, scope/counterexamples and sourceSHA; trusts designated continuous reader, no duplicated full read',coverage_pdf_pages=list(range(1,11)))
review_path=d/'semantic-reviews'/f'{new["id"]}.json';save(review_path,review);new['semantic_review_path']=str(review_path.relative_to(root))
rows[rows.index(old)]=new
original_entry=next(r for r in manifest['papers'] if r['id']=='tai-2024-03');manifest['papers'][manifest['papers'].index(original_entry)]=new
amendment=dict(date='2026-10-12',operation='repair nonresearch induction slot; no holdout change',old_record=old,new_record=new,reason='Front Cover is not original research',selection_policy=proposal['selection_policy'],screen_log=proposal['screen_log'],source_proposal=str((out/'tai-replacement/replacement-candidate.json').relative_to(root)),original_manifest_backup=str(backup_manifest.relative_to(root)),original_manifest_sha256=hashlib.sha256(before_manifest).hexdigest(),original_records_backup=str(backup_records.relative_to(root)),original_records_sha256=hashlib.sha256(before_records).hexdigest(),holdout_preserved=True,original_active_positions=50,new_active_positions=50,qualification='An eligibility repair after original pilot lock. It does not claim all active50papers were selected before first pilot reading or that all remaining candidate qualifications are verified.')
save(d/'selection-amendments.json',[amendment]);save(out/'tai-slot-repair.json',amendment)
save(d/'excluded-records.json',[dict(old,excluded_from_active_corpus=True,exclusion_reason='nonresearch Front Cover',replacement_id=new['id'])])
manifest.update(original_lock_preserved_in=str(backup_manifest.relative_to(root)),original_lock_before_analysis=manifest.get('locked_before_analysis'),locked_before_analysis=False,active_selection_amended_at='2026-10-12',selection_amendments='selection-amendments.json',sampling_completion_note='50 active candidates with25/year20induction5holdout. One nonresearch Front Cover repaired by source-order eligibility audit; original lock and excluded record retained. Full review/qualification of all50 is still incomplete.')
save(manifest_path,manifest);records_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
ledger.append(dict(id=new['id'],venue='tai',year=2024,split='derivation',main_complete=True,full_and_appendix_complete=True,review_path=new['semantic_review_path'],pdf_path=new['pdf_path'],pdf_sha256=new['pdf_sha256'],new_pdf_acquisition=True,eligibility_repair=True))
save(out/'batch-ledger.json',dict(date='2026-10-12',papers=ledger,new_main_reviews=7,new_full_reviews=5,new_holdout_reviews=0,new_pdf_acquisitions=1,pdf_pages_continuously_read=117,additional_author_supplement_pages_read=7,institution_covers_in_pdf_page_count=3,qualification='Continuous scientific writing-style reading, not independent theorem/code correctness or reproduction. AAAI7page supplement read via full web PDF text without binary archive/visual glyph certification. TAI2025two cited supplements unreviewed and excluded from full count.'))

# Skill-only research assets are recorded in corpus metadata, never user indexes.
print(json.dumps(dict(new_main=7,new_full=5,slot_repair_id=new['id'],new_pdf=new['pdf_path']),ensure_ascii=False))
