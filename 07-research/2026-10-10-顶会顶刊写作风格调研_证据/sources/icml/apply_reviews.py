#!/usr/bin/env python3
import json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[4]
base=root/'07-research/2026-10-10-顶会顶刊写作风格调研_证据'
for venue in sys.argv[1:] or ['icml','neurips','iclr']:
 f=base/'corpus'/venue;m=json.loads((f/'manifest.json').read_text());reviews=json.loads((f/'reviews.json').read_text()) if (f/'reviews.json').exists() else {}
 for p in m['papers']:
  if p.get('parse_status')=='complete text extraction':p.pop('error',None)
  if p['id'] in reviews:
   p.update(reviews[p['id']]);p['semantic_review']=reviews[p['id']]['dimensions']
 (f/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
 (f/'records.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in m['papers']))
 done=[p for p in m['papers'] if p.get('semantic_review_status')=='complete_writing_focused_review']
 stats=dict(venue=venue,target_count=50,locked_count=len(m['papers']),downloaded_count=sum(p.get('download_status')=='available' for p in m['papers']),parsed_count=sum(p.get('parse_status')=='complete text extraction' for p in m['papers']),semantic_reviewed_count=len(done),holdout_reviewed_count=sum(p['split']=='holdout' for p in done),pending_semantic_count=50-len(done),status='complete' if len(done)==50 else 'incomplete',review_scope='Writing-focused semantic audit of main prose and appendix organization, purpose, claim statements and evidence links; not a proof correctness or replication audit.',holdout_validation_status='pending' if sum(p['split']=='holdout' for p in done)<10 else 'ready for rule validation')
 (f/'validation.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
 print(venue,stats)
