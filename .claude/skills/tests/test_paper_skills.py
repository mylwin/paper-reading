#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper-* skills 的单元测试（纯本地 tmp 工作区，不联网）。

运行：
    cd .claude/skills && python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

SKILLS_ROOT = Path(__file__).resolve().parents[1]
DAILY_SCRIPTS = SKILLS_ROOT / 'paper-daily' / 'scripts'
WEEKLY_SCRIPTS = SKILLS_ROOT / 'paper-weekly' / 'scripts'
for candidate in (DAILY_SCRIPTS, WEEKLY_SCRIPTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import paper_config  # noqa: E402
import sync_indexes  # noqa: E402
import fetch_pdfs  # noqa: E402
import collect  # noqa: E402


CONFIG = {
    'papers_dir': '01-raw',
    'markdown_dir': '02-markdown',
    'notes_dir': '03-notes',
    'equation_dir': '04-equation_problem',
    'translation_dir': '06-translation',
    'reading_dir': '08-reading',
    'daily_dir': '08-daily',
    'research_dir': '07-research',
    'layout': {
        'month_dirs': ['01-raw', '02-markdown', '03-notes', '06-translation'],
        'month_pattern': 'YYYY-MM',
        'readme_name': 'README.md',
        'index_name': 'index.md',
    },
}


def touch(path: Path, text: str = 'x') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


class MonthHelperTests(unittest.TestCase):
    def test_month_of(self):
        self.assertEqual(paper_config.month_of('2026-09-17'), '2026-09')
        self.assertEqual(paper_config.month_of('2026/9/1'), '2026-09')
        self.assertEqual(paper_config.month_of('20260917'), '2026-09')
        self.assertEqual(paper_config.month_of('2026-09'), '2026-09')
        self.assertEqual(paper_config.month_of(''), '')
        self.assertEqual(paper_config.month_of('not-a-date'), '')

    def test_is_month_dir(self):
        self.assertTrue(paper_config.is_month_dir('2026-09'))
        self.assertTrue(paper_config.is_month_dir('2026-01'))
        self.assertFalse(paper_config.is_month_dir('2026-13'))
        self.assertFalse(paper_config.is_month_dir('2026-9'))
        self.assertFalse(paper_config.is_month_dir('README'))

    def test_iter_month_dirs_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('2026-09', '2026-07', 'README', '2026-08'):
                (root / name).mkdir()
            self.assertEqual([p.name for p in paper_config.iter_month_dirs(root)],
                             ['2026-07', '2026-08', '2026-09'])
            self.assertEqual([p.name for p in paper_config.iter_month_dirs(root, descending=True)],
                             ['2026-09', '2026-08', '2026-07'])

    def test_paper_path_and_dir_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root / '2026-09' / 'Paper_A.pdf')
            self.assertEqual(paper_config.paper_path(root, 'Paper_A', '.pdf'),
                             root / '2026-09' / 'Paper_A.pdf')
            # 迁移期的旧平铺结构仍可命中
            touch(root / 'Paper_B.pdf')
            self.assertEqual(paper_config.paper_path(root, 'Paper_B', '.pdf'), root / 'Paper_B.pdf')
            self.assertIsNone(paper_config.paper_path(root, 'Missing', '.pdf'))
            (root / '2026-09' / 'Paper_C').mkdir()
            self.assertEqual(paper_config.paper_dir(root, 'Paper_C'), root / '2026-09' / 'Paper_C')

    def test_rel_link_depth(self):
        origin = Path('08-daily/2026-09-17')
        target = Path('01-raw/2026-09/Paper_A.pdf')
        self.assertEqual(paper_config.rel_link(target, origin), '../../01-raw/2026-09/Paper_A.pdf')
        # 03-notes/YYYY-MM/<标题>/ 下的文件位于第三层
        self.assertEqual(paper_config.rel_link(target, Path('03-notes/2026-09/Paper_A')),
                         '../../../01-raw/2026-09/Paper_A.pdf')

    def test_scan_paper_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root / '2026-09' / 'Paper_A.pdf')
            touch(root / '2026-09' / 'README.md')
            touch(root / '2026-09' / 'index.md')
            touch(root / 'Legacy.pdf')
            stems = sorted((e['month'], e['stem']) for e in
                           paper_config.scan_paper_entries(root, 'file', '.pdf'))
            self.assertEqual(stems, [('', 'Legacy'), ('2026-09', 'Paper_A')])


class ArchiveDateTests(unittest.TestCase):
    def test_month_readme_is_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            touch(workspace / '01-raw' / '2026-09' / 'Paper_A.pdf')
            touch(workspace / '01-raw' / '2026-09' / 'README.md', """# 2026-09 原始论文

<!-- INDEX:BEGIN -->
| 入库日期 | 论文标题 | paper_stem | 来源 | PDF |
|---|---|---|---|---|
| 2026-09-05 | Paper A | Paper_A | arxiv | [PDF](Paper_A.pdf) |
<!-- INDEX:END -->
""")
            dates = paper_config.read_archive_dates(workspace, CONFIG)
            self.assertEqual(dates['Paper_A']['date'], '2026-09-05')
            self.assertEqual(dates['Paper_A']['month'], '2026-09')
            self.assertEqual(dates['Paper_A']['title'], 'Paper A')

    def test_legacy_bullet_fallback_handles_unicode_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            touch(workspace / '01-raw' / '2026-09' / 'Rethinking_3D_Convolution_in_lp-norm_Space.pdf')
            touch(workspace / '01-raw' / 'index.md',
                  '# 索引\n\n- Rethinking 3D Convolution in ℓ_p-Norm Space —— 2026-09-16\n')
            dates = paper_config.read_archive_dates(workspace, CONFIG)
            self.assertEqual(dates['Rethinking_3D_Convolution_in_lp-norm_Space']['date'], '2026-09-16')


class FetchPdfsTests(unittest.TestCase):
    def test_index_existing_pdfs_is_month_recursive_and_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            papers_dir = Path(tmp) / '01-raw'
            touch(papers_dir / '2026-09' / 'lp-norm_Paper.pdf')
            touch(papers_dir / 'Legacy_Paper.pdf')
            index = fetch_pdfs.index_existing_pdfs(papers_dir)
            self.assertIn(paper_config.clean_stem('lp-norm_Paper'), index)
            self.assertIn(paper_config.clean_stem('lp_norm_Paper'), index)
            self.assertIn(paper_config.clean_stem('Legacy_Paper'), index)


class WeeklyStemTests(unittest.TestCase):
    def test_paper_stem_skips_month_segment(self):
        root = Path('02-markdown')
        cases = {
            '02-markdown/2026-09/Paper_A.md': 'Paper_A',
            '02-markdown/Paper_B.md': 'Paper_B',
        }
        for raw, expected in cases.items():
            self.assertEqual(collect._paper_stem(Path(raw), root),
                             paper_config.clean_stem(expected), raw)

    def test_paper_stem_uses_folder_for_paper_dirs(self):
        for root_name in ('03-notes', '06-translation', '08-reading', '04-equation_problem'):
            rel = Path('%s/2026-07/Paper_C/精读.md' % root_name)
            self.assertEqual(collect._paper_stem(rel, Path(root_name)),
                             paper_config.clean_stem('Paper_C'), root_name)

    def test_stem_from_repo_path(self):
        self.assertEqual(collect.stem_from_repo_path('01-raw/2026-09/Paper_A.pdf'),
                         paper_config.clean_stem('Paper_A'))
        self.assertEqual(collect.stem_from_repo_path('03-notes/2026-09/Paper_A/精读.md'),
                         paper_config.clean_stem('Paper_A'))
        self.assertEqual(collect.stem_from_repo_path('03-notes/Paper_B/精读.md'),
                         paper_config.clean_stem('Paper_B'))
        self.assertEqual(collect.stem_from_repo_path('06-translation/2026-09/Paper_C/双栏对比.pdf'),
                         paper_config.clean_stem('Paper_C'))

    def test_month_from_repo_path(self):
        self.assertEqual(collect.month_from_repo_path('01-raw/2026-09/Paper_A.pdf'), '2026-09')
        self.assertEqual(collect.month_from_repo_path('01-raw/Paper_A.pdf'), '')


class SyncIndexesTests(unittest.TestCase):
    def _fixture(self, tmp):
        workspace = Path(tmp)
        touch(workspace / '01-raw' / '2026-09' / 'README.md', """# 2026-09 原始论文

<!-- INDEX:BEGIN -->
| 入库日期 | 论文标题 | paper_stem | 来源 | PDF |
|---|---|---|---|---|
| 2026-09-05 | Paper One | Paper_One | arxiv | [PDF](Paper_One.pdf) |
| 2026-09-06 | Paper Two | Paper_Two | -- | [PDF](Paper_Two.pdf) |
| 2026-09-07 | Paper Three | Paper_Three | -- | [PDF](Paper_Three.pdf) |
<!-- INDEX:END -->
""")
        touch(workspace / '01-raw' / '2026-09' / 'Paper_One.pdf')
        touch(workspace / '01-raw' / '2026-09' / 'Paper_Two.pdf')
        touch(workspace / '01-raw' / '2026-09' / 'Paper_Three.pdf')
        touch(workspace / '02-markdown' / '2026-09' / 'Paper_One.md', '# Paper One\n')
        touch(workspace / '03-notes' / '2026-09' / 'Paper_One' / '精读.md', '# 精读\n')
        # 人工填写过的失败原因必须被保留
        touch(workspace / '02-markdown' / 'index.md', """# 论文解析结果索引

<!-- INDEX:BEGIN -->
| 月份 | 论文标题 | paper_stem | 解析状态 | 解析文件 | 入库日期 | 失败原因 |
|---|---|---|---|---|---|---|
| 2026-09 | Paper Three | Paper_Three | 解析失败 | -- | 2026-09-07 | PDF 为扫描版，需 OCR |
<!-- INDEX:END -->
""")
        # 人工填写的精读中状态与优先级也必须被保留
        touch(workspace / '03-notes' / 'index.md', """# 精读笔记索引

<!-- INDEX:BEGIN -->
| 月份 | 论文标题 | paper_stem | 精读状态 | 精读开始 | 精读完成 | 优先级 | 精读文件 |
|---|---|---|---|---|---|---|---|
| 2026-09 | Paper Two | Paper_Two | 精读中 | 2026-09-08 | -- | 高 | -- |
<!-- INDEX:END -->
""")
        return workspace

    def test_refresh_generates_indexes_and_month_readmes(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self._fixture(tmp)
            report = sync_indexes.refresh(workspace, CONFIG)
            self.assertGreater(len(report['changed']), 0)
            for rel in ('01-raw/index.md', '02-markdown/index.md',
                        '03-notes/index.md', '06-translation/index.md'):
                self.assertTrue((workspace / rel).is_file(), rel)
            for rel in ('02-markdown/2026-09/README.md', '03-notes/2026-09/README.md',
                        '06-translation/2026-09/README.md'):
                self.assertTrue((workspace / rel).is_file(), rel)

            registry = {item['paper_stem']: item for item in sync_indexes.build_registry(workspace, CONFIG)}
            self.assertEqual(registry['Paper_One']['parse_status'], '已解析')
            self.assertEqual(registry['Paper_One']['reading_status'], '已精读')
            self.assertEqual(registry['Paper_Two']['parse_status'], '未解析')
            self.assertEqual(registry['Paper_Two']['reading_status'], '精读中')
            self.assertEqual(registry['Paper_Two']['priority'], '高')
            # 脚本推不出的日期保留上一版人工填写值
            self.assertEqual(registry['Paper_Two']['reading_start'], '2026-09-08')
            self.assertEqual(registry['Paper_Three']['parse_status'], '解析失败')
            self.assertEqual(registry['Paper_Three']['parse_error'], 'PDF 为扫描版，需 OCR')
            self.assertTrue(all(i['translation_status'] == '未翻译' for i in registry.values()))

            # 未解析/未精读/未翻译清单必须出现在索引里
            raw_index = (workspace / '01-raw/index.md').read_text(encoding='utf-8')
            self.assertIn('**未解析论文**', raw_index)
            self.assertIn('**未精读论文**', raw_index)
            self.assertIn('**未翻译论文**', raw_index)
            # 索引内的 PDF 链接必须相对索引文件本身
            self.assertIn('(2026-09/Paper_One.pdf)', raw_index)

    def test_refresh_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self._fixture(tmp)
            sync_indexes.refresh(workspace, CONFIG)
            again = sync_indexes.refresh(workspace, CONFIG)
            self.assertEqual(again['changed'], [])

    def test_refresh_records_new_paper_month_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self._fixture(tmp)
            sync_indexes.refresh(workspace, CONFIG)
            touch(workspace / '01-raw' / '2026-10' / 'Paper_Four.pdf')
            sync_indexes.refresh(workspace, CONFIG, overrides={
                'Paper_Four': {'date': '2026-10-02', 'month': '2026-10', 'title': 'Paper Four'},
            })
            registry = {item['paper_stem']: item for item in sync_indexes.build_registry(workspace, CONFIG)}
            self.assertEqual(registry['Paper_Four']['archive_month'], '2026-10')
            self.assertEqual(registry['Paper_Four']['parse_status'], '未解析')
            readme = (workspace / '01-raw/2026-10/README.md').read_text(encoding='utf-8')
            self.assertIn('2026-10-02', readme)
            self.assertIn('Paper_Four.pdf', readme)

    def test_replace_marked_block_keeps_surrounding_text(self):
        text = 'before\n<!-- INDEX:BEGIN -->\nold\n<!-- INDEX:END -->\nafter\n'
        updated, ok = sync_indexes.replace_marked_block(text, 'new')
        self.assertTrue(ok)
        self.assertEqual(updated, 'before\n<!-- INDEX:BEGIN -->\nnew\n<!-- INDEX:END -->\nafter\n')


class CheckLinksTests(unittest.TestCase):
    def test_dangling_link_and_missing_readme_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            touch(workspace / '01-raw' / '2026-09' / 'README.md', 'ok\n')
            touch(workspace / '01-raw' / '2026-09' / 'Paper_A.pdf')
            touch(workspace / '03-notes' / '2026-09' / 'Paper_A' / '精读.md',
                  '[PDF](../../../01-raw/2026-09/Paper_A.pdf)\n[坏链](../../../01-raw/2026-09/Missing.pdf)\n')
            result = sync_indexes.check_links(workspace, CONFIG)
            self.assertEqual(result['dangling_count'], 1)
            self.assertEqual(result['dangling'][0]['target'], '../../../01-raw/2026-09/Missing.pdf')
            self.assertIn('03-notes/2026-09', result['missing_readme'])
            self.assertGreater(result['problems'], 0)

    def test_clean_workspace_has_no_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            for root, month in (('01-raw', '2026-09'), ('02-markdown', '2026-09'),
                                ('03-notes', '2026-09'), ('06-translation', '2026-09')):
                touch(workspace / root / month / 'README.md', 'ok\n')
                touch(workspace / root / 'README.md', 'ok\n')
            touch(workspace / '01-raw' / '2026-09' / 'README.md', """# 2026-09

<!-- INDEX:BEGIN -->
| 入库日期 | 论文标题 | paper_stem | 来源 | PDF |
|---|---|---|---|---|
| 2026-09-05 | Paper One | Paper_One | -- | [PDF](Paper_One.pdf) |
<!-- INDEX:END -->
""")
            touch(workspace / '01-raw' / '2026-09' / 'Paper_One.pdf')
            touch(workspace / '02-markdown' / '2026-09' / 'Paper_One.md',
                  '[PDF](../../01-raw/2026-09/Paper_One.pdf)\n')
            for root in ('04-equation_problem', '05-books', '07-research', '08-daily', '08-reading'):
                touch(workspace / root / 'README.md', 'ok\n')
            result = sync_indexes.check_links(workspace, CONFIG)
            self.assertEqual(result['dangling'], [])
            self.assertEqual(result['missing_readme'], [])
            self.assertEqual(result['consistency_errors'], [])
            self.assertEqual(result['problems'], 0)


class ConsistencyTests(unittest.TestCase):
    def test_month_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            touch(workspace / '01-raw' / '2026-09' / 'README.md', """# 2026-09

<!-- INDEX:BEGIN -->
| 入库日期 | 论文标题 | paper_stem | 来源 | PDF |
|---|---|---|---|---|
| 2026-09-05 | Paper One | Paper_One | -- | [PDF](Paper_One.pdf) |
<!-- INDEX:END -->
""")
            touch(workspace / '01-raw' / '2026-09' / 'Paper_One.pdf')
            # 解析结果被错误地放进了 2026-07
            touch(workspace / '02-markdown' / '2026-07' / 'Paper_One.md', '# x\n')
            result = sync_indexes.check_links(workspace, CONFIG)
            joined = ' '.join(result['consistency_errors'])
            self.assertIn('月份错位', joined)
            self.assertIn('Paper_One', joined)


if __name__ == '__main__':
    unittest.main()
