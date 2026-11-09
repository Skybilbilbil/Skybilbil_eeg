
# -*- coding: utf-8 -*-
"""把一个 docx 的排版参数抓成可读快照：页面、样式、正文段落分布、表格、图片。
用途：用户手改过格式后，据此把规则复刻进 render_paper_draft.py。"""
import sys
from collections import Counter
from docx import Document
from docx.oxml.ns import qn

def cm(v):
    return round(v / 360000, 2) if v is not None else None

def ea_font(style):
    rpr = style.element.find(qn('w:rPr'))
    if rpr is None:
        return None
    rf = rpr.find(qn('w:rFonts'))
    return rf.get(qn('w:eastAsia')) if rf is not None else None

def line_rule(pf):
    ls = pf.line_spacing
    if ls is None:
        return None
    return ('exact %s pt' % ls.pt) if hasattr(ls, 'pt') else ('multiple %.2f' % ls)

path = sys.argv[1]
doc = Document(path)
s = doc.sections[0]
print('== page ==')
print('  size cm      ', cm(s.page_width), 'x', cm(s.page_height))
print('  margins cm   ', 'top', cm(s.top_margin), 'bottom', cm(s.bottom_margin),
      'left', cm(s.left_margin), 'right', cm(s.right_margin))

print('== styles ==')
for nm in ['Normal', 'Title', 'Heading 1', 'Heading 2', 'Heading 3', 'Heading 4']:
    try:
        st = doc.styles[nm]
    except KeyError:
        continue
    f, pf = st.font, st.paragraph_format
    print('  %-10s font=%s ea=%s size=%s bold=%s align=%s line=%s before=%s after=%s first=%s'
          % (nm, f.name, ea_font(st), f.size.pt if f.size else None, f.bold,
             pf.alignment, line_rule(pf),
             pf.space_before.pt if pf.space_before else None,
             pf.space_after.pt if pf.space_after else None,
             cm(pf.first_line_indent)))

print('== body paragraphs (by style, size/line/first-indent) ==')
cnt = Counter()
for p in doc.paragraphs:
    if not p.text.strip():
        continue
    pf = p.paragraph_format
    sz = None
    for r in p.runs:
        if r.font.size:
            sz = r.font.size.pt
            break
    cnt[(p.style.name, sz, line_rule(pf), cm(pf.first_line_indent), str(pf.alignment))] += 1
for k, v in cnt.most_common(12):
    print('  %-14s size=%-6s line=%-16s first=%-6s align=%-8s  x%d' % (k[0], k[1], k[2], k[3], k[4], v))
print('  (total non-empty paragraphs: %d)' % sum(cnt.values()))

print('== tables ==')
for i, t in enumerate(doc.tables[:3], 1):
    print('  table%d style=%s rows=%d cols=%d' % (i, t.style.name if t.style else None, len(t.rows), len(t.columns)))

print('== images ==')
for i, sh in enumerate(list(doc.inline_shapes)[:5], 1):
    print('  img%d width=%.2f cm height=%.2f cm' % (i, sh.width / 360000, sh.height / 360000))
