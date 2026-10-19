"""Remove only this run's excluded provisional record via the index writer."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.claude/skills/paper-daily/scripts'))
import sync_indexes as s
from paper_config import load_config, clean_stem

excluded = 'The_Ephemeral_Cost_of_Switching_Optimizers_Across_LLM_Training_Stages'
assert not (ROOT / '01-raw/2026-10' / (excluded + '.pdf')).exists()
original = s.read_archive_dates
def archive(workspace, config):
 return {key: value for key, value in original(workspace, config).items() if clean_stem(key) != clean_stem(excluded)}
s.read_archive_dates = archive
report = s.refresh(ROOT, load_config())
print('Final registry:', report['papers'], '; updated indexes:', len(report['changed']))
