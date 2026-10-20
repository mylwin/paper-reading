import sys,json,urllib.request,urllib.parse,concurrent.futures
from pathlib import Path
from datetime import datetime,timedelta
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).parent
sys.path.insert(0,str(ROOT/'.claude/skills/paper-daily/scripts'))
import search_arxiv as search
coverage=[]
def bounded_openreview(keywords, start_date, end_date, max_results=200, **kwargs):
    def query(term):
        params = {'term': term, 'limit': 100, 'offset': 0,
                  'content': 'all', 'source': 'forum'}
        request = urllib.request.Request(
            search.OPENREVIEW_SEARCH_URL + '?' + urllib.parse.urlencode(params),
            headers={'User-Agent': search.OPENREVIEW_USER_AGENT})
        record = {'term': term, 'page_limit': 1, 'page_size': 100}
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                data = json.load(response)
            notes = data.get('notes') or []
            papers = []
            for note in notes:
                paper = search.parse_openreview_note(note)
                if not paper:
                    continue
                published = paper.get('published_date')
                if published is None or start_date.date() <= published.date() <= end_date.date():
                    papers.append(paper)
            record.update(retrieved=len(notes), in_window=len(papers),
                          truncated=len(notes) == 100, status='ok')
        except Exception as error:
            papers = []
            record.update(status='failed', detail=str(error))
        return papers, record

    papers, seen = [], set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for batch, record in executor.map(query, keywords):
            coverage.append(record)
            print('OpenReview coverage:', json.dumps(record), file=sys.stderr)
            for paper in batch:
                key = paper.get('id') or paper['title']
                if key not in seen:
                    seen.add(key)
                    papers.append(paper)
    return papers



c=search.load_research_config(None)
kw=list(dict.fromkeys(k for d in c['research_domains'].values() for k in d['keywords']))
papers=bounded_openreview(kw,datetime(2026,9,20),datetime(2026,10,20))
(RUN/'openreview-raw.json').write_text(json.dumps(papers,ensure_ascii=False,indent=2,default=str)+'\n')
(RUN/'openreview-coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
print('Unique in-window OpenReview:',len(papers))
