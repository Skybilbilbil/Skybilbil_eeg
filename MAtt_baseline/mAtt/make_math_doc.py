#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把《黎曼几何数学基础：从线性代数到流形》md 转成 docx（含带页码的目录）。

两遍渲染：
  1) 第一遍生成临时 docx（目录页码为空）；
  2) 用 LibreOffice 转 PDF，逐页检索章标题，得到真实页码；
  3) 第二遍把页码写入目录（PAGEREF 域，可再按 F9 刷新）。
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import make_geometry_doc as G

HERE = Path(__file__).resolve().parent
SOFFICE = '/home/lenovo/.local/bin/soffice'


def dollar_to_fence(text):
    lines = text.split('\n')
    out = []
    i, n = 0, len(lines)
    while i < n:
        raw = lines[i]
        s = raw.strip()
        if s.startswith('$$') and s.endswith('$$') and len(s) > 4:
            out.append(G.FENCE)
            out.append(s[2:-2].strip())
            out.append(G.FENCE)
            i += 1
            continue
        if s.startswith('$$'):
            out.append(G.FENCE)
            head = s[2:].strip()
            if head:
                out.append(head)
            i += 1
            while i < n and '$$' not in lines[i]:
                out.append(lines[i].strip())
                i += 1
            if i < n:
                tail = lines[i].strip().replace('$$', '').strip()
                if tail:
                    out.append(tail)
                i += 1
            out.append(G.FENCE)
            continue
        out.append(raw)
        i += 1
    return '\n'.join(out)


def _norm(s):
    return re.sub(r'\s+', '', s)


def docx_to_pdf(docx_path, outdir):
    cmd = [SOFFICE, '--headless',
           '-env:UserInstallation=file://' + str(Path(outdir) / 'lo_profile'),
           '--convert-to', 'pdf', '--outdir', str(outdir), str(docx_path)]
    subprocess.run(cmd, check=True, timeout=600,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdfs = list(Path(outdir).glob('*.pdf'))
    if not pdfs:
        raise RuntimeError('PDF 未生成')
    return pdfs[0]


def locate_pages(pdf_path, entries, skip_pages=2):
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    texts = [_norm(p.extract_text() or '') for p in reader.pages]
    pages = {}
    for lvl, text, anchor in entries:
        key = _norm(text)
        if not key:
            continue
        for idx, pt in enumerate(texts):
            if idx < skip_pages:
                continue
            if key in pt:
                pages[anchor] = idx + 1
                break
        else:
            head = key[:10]
            if head:
                for idx, pt in enumerate(texts):
                    if idx < skip_pages:
                        continue
                    if head in pt:
                        pages[anchor] = idx + 1
                        break
    return pages


def main():
    src = HERE / '黎曼几何数学基础_从线性代数到流形.md'
    dst = HERE / '黎曼几何数学基础_从线性代数到流形.docx'

    G.SRC = src
    G.SUBTITLE = '从线性代数到流形 · 数学与算法基础'
    G.PREPROCESS = dollar_to_fence
    G.HEADER_TEXT = '黎曼几何数学基础 · 从线性代数到流形'

    entries = G._collect_toc(src.read_text(encoding='utf-8').split('\n'))
    print('目录条目:', len(entries))

    tmpdir = Path(tempfile.mkdtemp(prefix='mathdoc_'))
    try:
        tmp_docx = tmpdir / (dst.stem + '.docx')
        G.DST = tmp_docx
        G.PAGES = {}
        G.main()
        print('第一遍完成，转 PDF ...')
        pdf = docx_to_pdf(tmp_docx, tmpdir)
        pages = locate_pages(pdf, entries)
        hit = sum(1 for _, _, a in entries if a in pages)
        print('定位到页码:', hit, '/', len(entries))
        for lvl, text, anchor in entries[:4]:
            print('   %-36s p.%s' % (text[:34], pages.get(anchor)))

        G.DST = dst
        G.PAGES = pages
        G.main()
        print('第二遍完成（含页码）')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
