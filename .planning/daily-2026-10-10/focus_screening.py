"""Focus-screen the already retrieved pool; no new source requests."""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.claude/skills/paper-daily/scripts'))
import search_arxiv as search


def cached(source, **kwargs):
    papers = json.loads((RUN / (source + '-raw.json')).read_text(encoding='utf-8'))
    for paper in papers:
        if paper.get('published_date'):
            paper['published_date'] = datetime.fromisoformat(paper['published_date'])
    return papers


search.fetch_with_paper_config = cached
# The original live Semantic Scholar requests exhausted retries with HTTP 429.
search.search_semantic_scholar_hot_papers = lambda **kwargs: []
raise SystemExit(search.main())
