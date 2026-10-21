from pathlib import Path
import hashlib
import json
import subprocess
from pypdf import PdfReader, PdfWriter

root = Path('/Users/myl/Documents/paper')
stem = 'ICLR-2025-stochastic-polyak-step-sizes-and-momentum-convergence-guarantees-and-practical-performance-Paper-Conference'
source = root / '01-raw/2026-10' / (stem + '.pdf')
dest = root / '03-notes/2026-10' / stem / 'images'
dest.mkdir(parents=True, exist_ok=True)
# Rectangles are PDF points measured from the top-left of the original page.
figures = [
    ('fig1', 4, (285, 531, 504, 630), '原文图 1：直接加入动量与修正步长的比较'),
    ('fig2', 9, (265, 388, 503, 497), '原文图 2：确定性最小二乘与步长轨迹'),
    ('fig3', 10, (107, 365, 290, 453), '原文图 3：vowel 训练损失与训练准确率'),
    ('fig4', 10, (293, 365, 474, 453), '原文图 4：letter 的衰减步长比较'),
    ('fig5', 10, (107, 590, 290, 692), '原文图 5：ResNet34 / CIFAR10'),
    ('fig6', 10, (293, 590, 474, 692), '原文图 6：ResNet34 / CIFAR100'),
    ('eq26', 30, (103, 612, 508, 732), '附录 D.2 式 (26) 及其上下文'),
]
records = []
for name, number, rect, caption in figures:
    page = PdfReader(str(source)).pages[number - 1]
    height = float(page.mediabox.height)
    x0, y0, x1, y1 = rect
    page.cropbox.lower_left = (x0, height-y1)
    page.cropbox.upper_right = (x1, height-y0)
    writer = PdfWriter()
    writer.add_page(page)
    crop = Path('/tmp') / ('sps-crop-' + name + '.pdf')
    with crop.open('wb') as handle:
        writer.write(handle)
    subprocess.run(['pdftoppm', '-cropbox', '-r', '300', '-png', '-singlefile', str(crop), str(dest/name)], check=True)
    records.append({'file': name+'.png', 'page': number, 'rectangle_top_left_points': rect, 'caption': caption})
manifest = {'source': str(source.relative_to(root)), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'dpi':300, 'figures':records}
(root/'.planning/sps-momentum-reading-2026-10-21/figure-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
lines = ['# 原图来源', '', '全部图片裁自用户提供的 ICLR 2025 会议版 PDF；保留曲线、坐标和图例，未重绘或估读数值。', '', '源文件 SHA-256：`'+manifest['sha256']+'`。栅格化分辨率：300 dpi。', '', '| 文件 | PDF 页码 | 内容 |', '|---|---|---|']
lines += ['| '+r['file']+' | '+str(r['page'])+' | '+r['caption']+' |' for r in records]
(dest/'_sources.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'images': len(records), 'directory':str(dest)}, ensure_ascii=False))
