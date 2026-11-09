#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把《黎曼几何与微分几何_算法数学原理讲义.md》转成排版精修过的 .docx。

与旧版的差别（用户要求"Word 里公式要好看"）：
  * 公式改成 **Word 原生公式 (OMML)**：真上下标、真分式、Cambria Math，
    而不是等宽字体里的 Unicode 文本；
  * 裸 \`\`\` 围栏块 = 数学公式（本讲义一律如此），带语言标记的 = 代码；
  * 正文里的反引号片段自动区分「数学符号」与「代码标识符」；
  * 标题层级 / 表格（表头深色+隔行浅底+跨页重复表头）/ 代码块（灰底+左侧竖条）/
    引用块（左竖线）/ 图注 / 页眉页脚页码 / 目录域 全部重做。

不依赖 pandoc；只用 python-docx + lxml。
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import (WD_ALIGN_PARAGRAPH, WD_BREAK,
                            WD_LINE_SPACING, WD_TAB_ALIGNMENT,
                            WD_TAB_LEADER)
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
SRC = HERE / '黎曼几何数学基础_从线性代数到流形.md'
DST = HERE / '黎曼几何数学基础_从线性代数到流形.docx'
SUBTITLE = 'mAtt / SPD 流形注意力的数学地基'
PREPROCESS = None
PAGES = None
HEADER_TEXT = '黎曼几何与微分几何 · mAtt 算法数学原理讲义'   # {anchor: 页码}，由调用方在两遍渲染时注入   # 可选：读入 markdown 后、解析前调用的文本预处理函数

BT = chr(96)
FENCE = BT * 3

# ---------------- 配色 ----------------
C_H1 = RGBColor(0x1F, 0x38, 0x64)
C_H2 = RGBColor(0x1F, 0x4E, 0x79)
C_H3 = RGBColor(0x2E, 0x5C, 0x8A)
C_BODY = RGBColor(0x21, 0x21, 0x21)
C_CODE = RGBColor(0x8E, 0x24, 0x00)
C_MUTED = RGBColor(0x60, 0x6A, 0x72)
FILL_HDR = '1F4E79'
FILL_ZEBRA = 'EEF3F8'
FILL_CODE = 'F6F7F9'
BORDER_CODE = 'B0BEC5'


M_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

SUP_MAP = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
           '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁺': '+', '⁻': '−',
           'ⁿ': 'n', 'ᵀ': 'T', 'ᵏ': 'k', 'ᵈ': 'd', 'ᵖ': 'p'}
SUB_MAP = {'₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5',
           '₆': '6', '₇': '7', '₈': '8', '₉': '9', 'ᵢ': 'i', 'ⱼ': 'j',
           'ₖ': 'k', 'ₙ': 'n'}

FUNCS = {'log', 'exp', 'det', 'tr', 'diag', 'argmin', 'argmax', 'max',
         'min', 'sym', 'dim', 'span', 'expm', 'logm', 'sign', 'qf',
         'cos', 'sin', 'rank', 'det', 'sup', 'inf'}

# 这些字符在公式里应保持正体（不斜体）
UPRIGHT = set('Σ∏∈‖⟨⟩→⇒≥≤≠≈±×·−√ℝΛ∂∇⋯…∞□△℃°←⇐↦↔⊤⊥')

# 中文字符与中文标点
CJK = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]')

COMBINING = re.compile(r'[\u0300-\u036f]')


def esc(t):
    return (t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def _run(text, nor=False):
    """一个 m:r。nor=True 表示正体普通文本（n 内的说明文字/函数名）。"""
    if not text:
        return ''
    if nor:
        return ('<m:r><m:rPr><m:nor/></m:rPr>'
                '<w:rPr><w:rFonts w:ascii="Cambria" w:hAnsi="Cambria" '
                'w:eastAsia="宋体"/></w:rPr>'
                '<m:t xml:space="preserve">' + esc(text) + '</m:t></m:r>')
    return ('<m:r><m:rPr><m:sty m:val="p"/></m:rPr>'
            '<m:t xml:space="preserve">' + esc(text) + '</m:t></m:r>')


def _var(text):
    return '<m:r><m:t xml:space="preserve">' + esc(text) + '</m:t></m:r>'


# ---------- 词法：把一行切成立即量（atom）与上下标标记 ----------

def tokenize(s):
    """返回 token 列表，元素为 ('atom', omml) / ('sup', omml) / ('sub', omml)。"""
    toks = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        # 显式 ^{...} / _{...}
        if c in '^_' and i + 1 < n:
            kind = 'sup' if c == '^' else 'sub'
            if s[i + 1] == '{':
                j = s.find('}', i + 2)
                if j > 0:
                    inner = s[i + 2:j]
                    toks.append((kind, build_seq(inner)))
                    i = j + 1
                    continue
            elif s[i + 1] in '([':
                oc = s[i + 1]
                cc = ')' if oc == '(' else ']'
                depth, j = 0, i + 1
                while j < n:
                    if s[j] == oc:
                        depth += 1
                    elif s[j] == cc:
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if j < n:
                    toks.append((kind, build_seq(s[i + 2:j])))
                    i = j + 1
                    continue
            else:
                m2 = re.match(r'[A-Za-z0-9]+', s[i + 1:])
                if m2:
                    nxt = m2.group(0)
                else:
                    nxt = s[i + 1]
                toks.append((kind, build_seq(nxt)))
                i += 1 + len(nxt)
                continue
        # Unicode 上标 / 下标字符
        if c in SUP_MAP:
            run = ''
            while i < n and s[i] in SUP_MAP:
                run += SUP_MAP[s[i]]
                i += 1
            toks.append(('sup', build_seq(run)))
            continue
        if c in SUB_MAP:
            run = ''
            while i < n and s[i] in SUB_MAP:
                run += SUB_MAP[s[i]]
                i += 1
            toks.append(('sub', build_seq(run)))
            continue
        # 中文片段
        if CJK.match(c):
            j = i
            while j < n and CJK.match(s[j]):
                j += 1
            toks.append(('atom', _run(s[i:j], nor=True)))
            i = j
            continue
        # 数字（含 1e-6、1e-10 这类）
        m = re.match(r'\d+(?:\.\d+)?(?:e[+-]\d+)?', s[i:])
        if m:
            toks.append(('atom', _run(m.group(0))))
            i += len(m.group(0))
            continue
        # 正体符号
        if c in UPRIGHT:
            toks.append(('atom', _run(c)))
            i += 1
            continue
        # 括号组：整体递归（方括号不分组：公式里 [0,∞) 这类半开区间会被后文 ] 误配）
        if c in '({':
            close = {'(': ')', '{': '}'}[c]
            depth, j = 0, i
            while j < n:
                if s[j] in '({':
                    depth += 1
                elif s[j] in ')}':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j >= n:
                toks.append(('atom', _run(c)))
                i += 1
                continue
            inner = s[i + 1:j]
            toks.append(('atom', _run(c) + build_seq(inner) + _run(close)))
            i = j + 1
            continue
        # 连字符统一成数学减号
        if c == '-':
            toks.append(('atom', _run('−')))
            i += 1
            continue
        # 多字母词（函数名或缩写）
        m = re.match(r'[A-Za-z]{2,}', s[i:])
        if m:
            w = m.group(0)
            if w in FUNCS:
                toks.append(('atom', _run(w, nor=True)))
            else:
                toks.append(('atom', _run(w, nor=True)))
            i += len(w)
            continue
        # 单字母变量
        if c.isalpha():
            ch = c
            if COMBINING.match(s[i + 1:i + 2] or ''):
                ch = s[i:i + 2]
            toks.append(('atom', _var(ch)))
            i += len(ch)
            continue
        # 其余（= + , . : ; 等）
        toks.append(('atom', _run(c)))
        i += 1
    return toks


NORM = _run('‖')


def _final(entry):
    if entry['sup'] is not None:
        return ('<m:sSup><m:e>' + entry['omml'] + '</m:e><m:sup>' +
                entry['sup'] + '</m:sup></m:sSup>')
    if entry['sub'] is not None:
        return ('<m:sSub><m:e>' + entry['omml'] + '</m:e><m:sub>' +
                entry['sub'] + '</m:sub></m:sSub>')
    return entry['omml']


def _merge_norm(out):
    # 把 ‖ ... ‖ 整段并成一个原子，好让后面的 _F 挂在整个范数上
    if not out or out[-1]['omml'] != NORM:
        return False
    for k in range(len(out) - 2, -1, -1):
        if out[k]['omml'] == NORM:
            inner = ''.join(_final(o) for o in out[k:])
            del out[k:]
            out.append({'omml': inner, 'sup': None, 'sub': None})
            return True
    return False

def build_seq(s):
    """把一段文本转成 OMML 序列；处理上下标挂到前一个原子上。"""
    toks = tokenize(s)
    out = []
    for kind, payload in toks:
        if kind == 'atom':
            out.append({'omml': payload, 'sup': None, 'sub': None})
            continue
        if not out:
            out.append({'omml': payload, 'sup': None, 'sub': None})
            continue
        if kind in ('sup', 'sub'):
            _merge_norm(out)
        tgt = out[-1]
        if kind == 'sup':
            if tgt['sub'] is not None:      # 先下后上 -> sSubSup
                out[-1] = {'omml': (
                    '<m:sSubSup><m:e>' + tgt['omml'] + '</m:e>'
                    '<m:sub>' + tgt['sub'] + '</m:sub>'
                    '<m:sup>' + payload + '</m:sup></m:sSubSup>'), 'sup': None, 'sub': None}
            elif tgt['sup'] is not None:    # 连写两个 ^ -> 合并
                out[-1] = {'omml': (
                    '<m:sSup><m:e>' + tgt['omml'] + '</m:e>'
                    '<m:sup>' + payload + '</m:sup></m:sSup>'), 'sup': None, 'sub': None}
            else:
                out[-1] = dict(tgt, sup=payload)
        else:
            if tgt['sup'] is not None:      # 先上后下 -> sSubSup
                out[-1] = {'omml': (
                    '<m:sSubSup><m:e>' + tgt['omml'] + '</m:e>'
                    '<m:sub>' + payload + '</m:sub>'
                    '<m:sup>' + tgt['sup'] + '</m:sup></m:sSubSup>'), 'sup': None, 'sub': None}
            else:
                out[-1] = dict(tgt, sub=payload)
    # 收尾：把只挂了 sub 或 sup 的原子变成真正的 sSub / sSup
    parts = []
    for o in out:
        if o['sup'] is not None:
            parts.append('<m:sSup><m:e>' + o['omml'] + '</m:e>'
                         '<m:sup>' + o['sup'] + '</m:sup></m:sSup>')
        elif o['sub'] is not None:
            parts.append('<m:sSub><m:e>' + o['omml'] + '</m:e>'
                         '<m:sub>' + o['sub'] + '</m:sub></m:sSub>')
        else:
            parts.append(o['omml'])
    return ''.join(parts)


def line_to_omml(text):
    body = build_seq(text.strip())
    return ('<m:oMath xmlns:m="' + M_NS + '" xmlns:w="' + W_NS + '">' +
            body + '</m:oMath>')


def has_cjk(s):
    return bool(CJK.search(s))

# =====================================================================
# 第三部分：排版辅助
# =====================================================================

def set_run_font(run, ascii_font='Cambria', east='宋体', size=None,
                 bold=None, color=None, italic=None):
    run.font.name = ascii_font
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:ascii'), ascii_font)
    rFonts.set(qn('w:hAnsi'), ascii_font)
    rFonts.set(qn('w:eastAsia'), east)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def shade_paragraph(par, fill, left_border=None):
    pPr = par._p.get_or_add_pPr()
    if left_border:
        pBdr = OxmlElement('w:pBdr')
        left = OxmlElement('w:left')
        left.set(qn('w:val'), 'single')
        left.set(qn('w:sz'), '18')
        left.set(qn('w:space'), '6')
        left.set(qn('w:color'), left_border)
        pBdr.append(left)
        pPr.append(pBdr)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill)
    pPr.append(shd)


def bottom_rule(par, color='D0D7DE', size='8'):
    pPr = par._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bot = OxmlElement('w:bottom')
    bot.set(qn('w:val'), 'single')
    bot.set(qn('w:sz'), size)
    bot.set(qn('w:space'), '4')
    bot.set(qn('w:color'), color)
    pBdr.append(bot)
    pPr.append(pBdr)


def shade_cell(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)


def set_table_width_pct(table, pct):
    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn('w:tblW')):
        tblPr.remove(old)
    w = OxmlElement('w:tblW')
    w.set(qn('w:type'), 'pct')
    w.set(qn('w:w'), str(pct))
    tblPr.append(w)


def cell_margins(table, top=40, bottom=40, left=80, right=80):
    tblPr = table._tbl.tblPr
    mar = OxmlElement('w:tblCellMar')
    for tag, val in (('top', top), ('left', left), ('bottom', bottom), ('right', right)):
        el = OxmlElement('w:' + tag)
        el.set(qn('w:w'), str(val))
        el.set(qn('w:type'), 'dxa')
        mar.append(el)
    tblPr.append(mar)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement('w:tblHeader')
    el.set(qn('w:val'), 'true')
    trPr.append(el)


def add_field(par, instr, placeholder=''):
    run = par.add_run()
    fld = OxmlElement('w:fldChar')
    fld.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText')
    it.set(qn('xml:space'), 'preserve')
    it.text = instr
    sep = OxmlElement('w:fldChar')
    sep.set(qn('w:fldCharType'), 'separate')
    t = OxmlElement('w:t')
    t.text = placeholder
    end = OxmlElement('w:fldChar')
    end.set(qn('w:fldCharType'), 'end')
    for el in (fld, it, sep, t, end):
        run._r.append(el)
    return run


def setup_document(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)
    sec.header_distance = Cm(1.3)
    sec.footer_distance = Cm(1.3)

    st = doc.styles['Normal']
    st.font.name = 'Cambria'
    st.font.size = Pt(10.5)
    st.font.color.rgb = C_BODY
    rPr = st.element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:ascii'), 'Cambria')
    rFonts.set(qn('w:hAnsi'), 'Cambria')
    rFonts.set(qn('w:eastAsia'), '宋体')
    pf = st.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.45
    pf.space_after = Pt(5)
    pf.space_before = Pt(0)

    for name, size, color, before, after in (
            ('Heading 1', 16, C_H1, 20, 10),
            ('Heading 2', 13, C_H2, 14, 7),
            ('Heading 3', 11.5, C_H3, 11, 5)):
        s = doc.styles[name]
        s.font.name = 'Cambria'
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = color
        rPr = s.element.get_or_add_rPr()
        rFonts = rPr.find(qn('w:rFonts'))
        if rFonts is None:
            rFonts = OxmlElement('w:rFonts')
            rPr.insert(0, rFonts)
        rFonts.set(qn('w:ascii'), 'Cambria')
        rFonts.set(qn('w:hAnsi'), 'Cambria')
        rFonts.set(qn('w:eastAsia'), '微软雅黑')
        s.paragraph_format.space_before = Pt(before)
        s.paragraph_format.space_after = Pt(after)
        s.paragraph_format.keep_with_next = True
        s.paragraph_format.line_spacing = 1.2

    # 页眉
    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = hp.add_run(HEADER_TEXT)
    set_run_font(r, 'Cambria', '微软雅黑', 8.5, color=C_MUTED)
    bottom_rule(hp, 'D9E1E8', '4')

    # 页脚：页码
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = fp.add_run('— ')
    set_run_font(r1, 'Cambria', '宋体', 9, color=C_MUTED)
    fr = add_field(fp, 'PAGE', '1')
    set_run_font(fr, 'Cambria', '宋体', 9, color=C_MUTED)
    r2 = fp.add_run(' —')
    set_run_font(r2, 'Cambria', '宋体', 9, color=C_MUTED)


def add_page_break(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_break(WD_BREAK.PAGE)


def _clean_title(t):
    """把标题里的 markdown 标记清掉，供目录显示。"""
    t = t.replace('**', '').replace(chr(96), '')
    def _f(m):
        inner = m.group(1)
        inner = re.sub(r'\\[A-Za-z]+', '', inner)
        return inner.replace('{', '').replace('}', '')
    t = re.sub(r'\$([^$]+)\$', _f, t)
    return t.strip()


def _collect_toc(lines):
    """第一遍扫描：收集 # 与 ## 标题（跳过围栏块），返回 [(level, text, anchor)]。"""
    out = []
    in_code = False
    first_h1 = True
    for raw in lines:
        s = raw.strip()
        if s.startswith(FENCE):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = re.match(r'^(#{1,2})\s+(.*)$', s)
        if not m:
            continue
        lvl = len(m.group(1))
        if lvl != 1:
            continue          # 目录只列到章一级
        if first_h1:
            first_h1 = False
            continue
        out.append((lvl, _clean_title(m.group(2)), 'bm%d' % (len(out) + 1)))
    return out


def _add_internal_link(paragraph, text, anchor, level):
    hl = OxmlElement('w:hyperlink')
    hl.set(qn('w:anchor'), anchor)
    hl.set(qn('w:history'), '1')
    r = paragraph.add_run(text)
    set_run_font(r, 'Cambria', '宋体', 10.5, bold=False, color=C_BODY)
    paragraph._p.remove(r._r)
    hl.append(r._r)
    paragraph._p.append(hl)


def _page_field(paragraph, anchor, fallback):
    """插入 PAGEREF 域；域结果预填 fallback，按 F9 可刷新为精确页码。"""
    def _fld(kind):
        r = paragraph.add_run()
        e = OxmlElement('w:fldChar')
        e.set(qn('w:fldCharType'), kind)
        r._r.append(e)
        return r

    _fld('begin')
    r = paragraph.add_run()
    it = OxmlElement('w:instrText')
    it.set(qn('xml:space'), 'preserve')
    it.text = ' REF %s \\p \\h ' % anchor
    r._r.append(it)
    _fld('separate')
    rn = paragraph.add_run(str(fallback) if fallback else '')
    set_run_font(rn, 'Cambria', '宋体', 10, color=C_BODY)
    _fld('end')


def add_bookmark(paragraph, name, bid):
    start = OxmlElement('w:bookmarkStart')
    start.set(qn('w:id'), str(bid))
    start.set(qn('w:name'), name)
    end = OxmlElement('w:bookmarkEnd')
    end.set(qn('w:id'), str(bid))
    p_el = paragraph._p
    # bookmarkStart 必须位于 w:pPr 之后，否则 Word 修复文档时书签错位
    ppr = p_el.find(qn('w:pPr'))
    p_el.insert(1 if ppr is not None else 0, start)
    p_el.append(end)


def add_toc_page(doc, entries, pages=None):
    lbl = doc.add_paragraph()
    lbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lbl.paragraph_format.space_before = Pt(4)
    lbl.paragraph_format.space_after = Pt(12)
    r = lbl.add_run('目  录')
    set_run_font(r, 'Cambria', '微软雅黑', 15, bold=True, color=C_H1)
    pages = pages or {}
    last = lbl
    for lvl, text, anchor in entries:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Cm(0.0 if lvl == 1 else 0.9)
        pf.space_before = Pt(7 if lvl == 1 else 0)
        pf.space_after = Pt(1)
        pf.line_spacing = 1.3
        pf.tab_stops.add_tab_stop(Cm(15.0), WD_TAB_ALIGNMENT.RIGHT,
                                  WD_TAB_LEADER.DOTS)
        _add_internal_link(p, text, anchor, lvl)
        rt = p.add_run('\t')
        set_run_font(rt, 'Cambria', '宋体', 10, color=C_BODY)
        _page_field(p, anchor, pages.get(anchor))
        last = p
    return last


MATH_HINT = set('ᵀ‖Σλ∈≥≤→⇒·≠≈√Λ∂∇−×')


def is_math_span(t):
    if any(ch in MATH_HINT for ch in t):
        return True
    if re.match(r'^[A-Za-z\u03b1-\u03c9]{1,3}(_\{?[A-Za-z0-9,<>]+\}?)+$', t):
        return True
    if re.match(r'^[A-Za-z]{2,5}\([^()]*\)$', t):
        return True
    return False


def add_omml(par, text):
    xml = None
    try:
        from latex2omml import latex_to_omml
        xml = latex_to_omml(text)
    except Exception:
        xml = None
    if xml is None:
        try:
            xml = line_to_omml(text)
        except Exception:
            xml = None
    if xml is not None:
        try:
            par._p.append(parse_xml(xml))
            return
        except Exception:
            pass
    r = par.add_run(text)
    set_run_font(r, 'Cambria', '宋体', 10.5)


SPAN = re.compile(BT + r'([^' + BT + r']+)' + BT + r'|\$([^$\n]+?)\$|\*\*([^*]+)\*\*')


def add_inlines(par, text, size=10.5, base_color=C_BODY, bold_all=False):
    text = text.replace('\u200b', '')
    pos = 0
    for m in SPAN.finditer(text):
        if m.start() > pos:
            r = par.add_run(text[pos:m.start()])
            set_run_font(r, 'Cambria', '宋体', size, bold=bold_all or None,
                         color=base_color)
        if m.group(1) is not None:
            token = m.group(1)
            if is_math_span(token):
                add_omml(par, token)
            else:
                r = par.add_run(token)
                set_run_font(r, 'Consolas', 'Consolas', size - 1, color=C_CODE)
        elif m.group(2) is not None:
            add_omml(par, m.group(2))
        else:
            inner = m.group(3)
            if BT in inner or chr(36) in inner:
                add_inlines(par, inner, size=size, base_color=base_color,
                            bold_all=True)
            else:
                r = par.add_run(inner)
                set_run_font(r, 'Cambria', '宋体', size, bold=True,
                             color=base_color)
        pos = m.end()
    if pos < len(text):
        r = par.add_run(text[pos:])
        set_run_font(r, 'Cambria', '宋体', size, bold=bold_all or None, color=base_color)


# =====================================================================
# 第四部分：Markdown 解析与落盘
# =====================================================================

def add_math_block(doc, lines):
    """一个裸围栏块 = 一组显示公式。单行居中；多行左对齐保留对齐空档。"""
    body = [l for l in lines if l.strip()]
    if not body:
        return
    multi = len(body) > 1
    for line in body:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(6 if not multi else 3)
        pf.space_after = Pt(6 if not multi else 3)
        pf.line_spacing = 1.3
        if multi or has_cjk(line):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.left_indent = Cm(0.9)
        else:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_omml(p, line.strip())
    tail = doc.add_paragraph()
    tail.paragraph_format.space_before = Pt(0)
    tail.paragraph_format.space_after = Pt(0)
    tail.paragraph_format.line_spacing = Pt(4)


def add_code_block(doc, lines, lang=''):
    if not lines:
        return
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(0.55)
    pf.right_indent = Cm(0.2)
    pf.space_before = Pt(5)
    pf.space_after = Pt(7)
    pf.line_spacing = 1.15
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    shade_paragraph(p, FILL_CODE, BORDER_CODE)
    for i, line in enumerate(lines):
        if i:
            br = p.add_run()
            br.add_break()
        r = p.add_run(line if line else ' ')
        set_run_font(r, 'Consolas', 'Consolas', 9, color=RGBColor(0x2B, 0x2B, 0x2B))
        for t in r._element.findall(qn('w:t')):
            t.set(qn('xml:space'), 'preserve')


def add_table(doc, header, rows):
    ncol = len(header)
    table = doc.add_table(rows=1 + len(rows), cols=ncol)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    cell_margins(table)
    set_table_width_pct(table, 5000)
    small = 8.2 if ncol >= 5 else 9.0

    for j, h in enumerate(header):
        cell = table.rows[0].cells[j]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.05
        add_inlines(p, h, size=small, base_color=RGBColor(0xFF, 0xFF, 0xFF),
                    bold_all=True)
        shade_cell(cell, FILL_HDR)
    repeat_header(table.rows[0])

    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = table.rows[i + 1].cells[j]
            cell.text = ''
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.12
            val = row[j] if j < len(row) else ''
            add_inlines(p, val, size=small)
            if i % 2 == 1:
                shade_cell(cell, FILL_ZEBRA)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(0)
    sp.paragraph_format.line_spacing = Pt(6)


def add_image(doc, rel, caption=None):
    path = (HERE / rel).resolve()
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(15.8))
    if caption:
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after = Pt(10)
        cp.paragraph_format.left_indent = Cm(0.8)
        cp.paragraph_format.right_indent = Cm(0.8)
        cp.paragraph_format.line_spacing = 1.2
        add_inlines(cp, caption, size=9, base_color=C_MUTED)
        for r in cp.runs:
            r.italic = True


def add_quote(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(0.7)
    pf.right_indent = Cm(0.6)
    pf.space_before = Pt(4)
    pf.space_after = Pt(6)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '18')
    left.set(qn('w:space'), '8')
    left.set(qn('w:color'), '8FA8BF')
    pBdr.append(left)
    pPr.append(pBdr)
    add_inlines(p, text, size=10.5, base_color=RGBColor(0x37, 0x47, 0x4F))
    for r in p.runs:
        r.italic = True



def set_compat_mode(doc, val=15):
    st = doc.settings.element
    for el in st.findall(qn('w:compat')):
        for cs in el.findall(qn('w:compatSetting')):
            if cs.get(qn('w:name')) == 'compatibilityMode':
                cs.set(qn('w:val'), str(val))
                return
    compat = OxmlElement('w:compat')
    cs = OxmlElement('w:compatSetting')
    cs.set(qn('w:name'), 'compatibilityMode')
    cs.set(qn('w:uri'), 'http://schemas.microsoft.com/office/word')
    cs.set(qn('w:val'), str(val))
    compat.append(cs)
    st.append(compat)


def enable_update_fields(doc):
    # 让 Word 打开时自动刷新目录 / 页码域，避免目录显示占位文字
    st = doc.settings.element
    el = OxmlElement('w:updateFields')
    el.set(qn('w:val'), 'false')
    st.append(el)


def _split_table_row(line):
    """按未转义的竖线切分表格行；LaTeX 的 \\| 不算分隔符。"""
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|') and not line.endswith(chr(92) + '|'):
        line = line[:-1]
    return [c.strip() for c in re.split(r'(?<!' + re.escape(chr(92)) + r')\|', line)]


def main():
    text = SRC.read_text(encoding='utf-8')
    if PREPROCESS is not None:
        text = PREPROCESS(text)
    lines = text.split(chr(10))
    doc = Document()
    setup_document(doc)
    set_compat_mode(doc)
    enable_update_fields(doc)

    toc_entries = _collect_toc(lines)
    pages = PAGES or {}
    toc_idx = [0]
    bm_seq = [1]

    i, n = 0, len(lines)
    first_h1 = True
    in_code = False
    code_lang = ''
    code_buf = []

    while i < n:
        raw = lines[i]
        s = raw.strip()

        # ---- 围栏块 ----
        if s.startswith(FENCE):
            if in_code:
                if code_lang == '':
                    add_math_block(doc, code_buf)
                else:
                    add_code_block(doc, code_buf, code_lang)
                code_buf = []
                in_code = False
                code_lang = ''
            else:
                in_code = True
                code_lang = s[3:].strip().lower()
            i += 1
            continue
        if in_code:
            code_buf.append(raw.rstrip())
            i += 1
            continue

        if not s:
            i += 1
            continue

        # ---- 图片 ----
        m = re.match(r'!\[[^\]]*\]\(([^)]+)\)', s)
        if m:
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            cap = None
            if i < n and lines[i].strip().startswith('*'):
                cap = lines[i].strip().strip('*').strip()
                i += 1
            add_image(doc, m.group(1), cap)
            continue

        # ---- 标题 ----
        m = re.match(r'^(#{1,4})\s+(.*)$', s)
        if m:
            level = min(len(m.group(1)), 3)
            text = m.group(2).strip()
            if first_h1 and level == 1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(4)
                r = p.add_run(text)
                set_run_font(r, 'Cambria', '微软雅黑', 21, bold=True, color=C_H1)
                first_h1 = False
                sub = doc.add_paragraph()
                sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
                sub.paragraph_format.space_after = Pt(14)
                r = sub.add_run(SUBTITLE)
                set_run_font(r, 'Cambria', '微软雅黑', 11, color=C_MUTED)
                bottom_rule(sub, 'C9D6E2', '6')
                sub.add_run().add_break(WD_BREAK.PAGE)
                toc_p = add_toc_page(doc, toc_entries, pages)
                toc_p.add_run().add_break(WD_BREAK.PAGE)
            else:
                hp = doc.add_heading('', level=level)
                add_inlines(hp, text, size={1: 16, 2: 13, 3: 11.5}[level],
                            base_color={1: C_H1, 2: C_H2, 3: C_H3}[level],
                            bold_all=True)
                if level == 1 and toc_idx[0] < len(toc_entries):
                    anchor = toc_entries[toc_idx[0]][2]
                    toc_idx[0] += 1
                    add_bookmark(hp, anchor, bm_seq[0])
                    bm_seq[0] += 1
                if level == 1 and i > 0:
                    bottom_rule(hp, 'D6E0EA', '6')
            i += 1
            continue

        # ---- 表格 ----
        if s.startswith('|') and i + 1 < n and re.match(r'^\|[\s:\-|]+\|$', lines[i + 1].strip()):
            header = _split_table_row(s)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith('|'):
                rows.append(_split_table_row(lines[i]))
                i += 1
            add_table(doc, header, rows)
            continue

        # ---- 引用 ----
        if s.startswith('>'):
            buf = [re.sub(r'^>\s?', '', s)]
            i += 1
            while i < n and lines[i].strip().startswith('>'):
                buf.append(re.sub(r'^>\s?', '', lines[i].strip()))
                i += 1
            add_quote(doc, ' '.join(buf))
            continue

        # ---- 分隔线 ----
        if re.match(r'^-{3,}$', s):
            i += 1
            continue

        # ---- 列表 ----
        m = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.*)$', raw)
        if m:
            indent = len(m.group(1))
            style = 'List Number' if m.group(2)[0].isdigit() else 'List Bullet'
            p = doc.add_paragraph(style=style)
            pf = p.paragraph_format
            pf.left_indent = Cm(0.75 + 0.55 * (indent // 2))
            pf.space_after = Pt(3)
            pf.line_spacing = 1.4
            add_inlines(p, m.group(3))
            i += 1
            continue

        # ---- 普通段落（连续行合并） ----
        buf = [s]
        i += 1
        while i < n:
            t = lines[i].strip()
            if not t:
                break
            if re.match(r'^(#{1,4}\s|\||>|!\[|-{3,}$)', t) or t.startswith(FENCE):
                break
            if re.match(r'^([-*+]|\d+\.)\s+', t):
                break
            buf.append(t)
            i += 1
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Cm(0.0)
        add_inlines(p, ' '.join(buf))

    doc.save(DST)
    print('saved', DST.name, '| paragraphs:', len(doc.paragraphs),
          '| tables:', len(doc.tables))


if __name__ == '__main__':
    raise SystemExit('本文件是排版引擎模块，请运行 make_math_doc.py 生成讲义 docx。')
