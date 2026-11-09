# -*- coding: utf-8 -*-
"""把论文初稿 md 渲染成 Word：A4、封面、页码、三线表、图题/表题、参考文献悬挂缩进。
复用 make_individual_diff_report.py 的字体/三线表/列宽规划组件，不重复实现。
产物：results/论文初稿/运动想象脑电的个体差异研究与分类算法设计.docx（+ 一份到 论文/）。
"""
import importlib.util, re, shutil, sys, time, zipfile
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn

HERE = Path(__file__).resolve().parent          # 论文/（仓库根）
BASE = HERE.parent / 'MAtt_baseline'            # MAtt_baseline/
OUT = BASE / 'results' / '论文初稿'
MD = HERE / '论文初稿_个体差异研究与分类算法设计.md'
DOCX = OUT / '运动想象脑电的个体差异研究与分类算法设计.docx'
REL = HERE / '运动想象脑电的个体差异研究与分类算法设计.docx'
FIGDIRS = [BASE / 'results' / '汇报材料' / 'figs', BASE / 'results' / 'individual_diff' / 'figs']


def load_renderer():
    spec = importlib.util.spec_from_file_location('mkreport', HERE.parent / 'MAtt_baseline' / 'analysis' / '差异报告' / 'make_individual_diff_report.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = load_renderer()
sys.path.insert(0, str(BASE / 'mAtt'))          # latex2omml 与模型本体同目录
from latex2omml import latex_to_omml             # noqa: E402
IMG = re.compile(r'^!\[图\s*(\d+)\s*·\s*(.+?)\]\((.+?)\)\s*$')
TABCAP = re.compile(r'^\*\*表\s*(\d+[a-z]?|[AB]\d+)\s*·\s*(.+?)\*\*$')
REFLINE = re.compile(r'^\[(\d+)\]\s')
BT3 = chr(96) * 3
INLINE = re.compile(r'\$([^$\n]+?)\$')
DISPLAY = re.compile(r'^\$\$(.+?)\$\$\s*(?:\((\d+[a-z]?)\))?\s*$')


def add_omml(par, latex):
    """LaTeX 子集 → Word 原生公式（OMML）。转换失败则原样落文本，不静默丢公式。"""
    try:
        par._p.append(parse_xml(latex_to_omml(latex)))
        return True
    except Exception:
        r = par.add_run(latex)
        R._fonts(r, R.SONG, R.LATIN)
        return False


def rich_runs(p, text, size=None):
    """含 $...$ 的正文段落：公式走 OMML，其余照旧交给 _runs（粗体/斜体/零宽空格不变）。"""
    pos = 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            R._runs(p, text[pos:m.start()], size=size)
        add_omml(p, m.group(1))
        pos = m.end()
    if pos < len(text):
        R._runs(p, text[pos:], size=size)


def math_line(doc, latex, num=None):
    """独立公式：居中制表位放公式、右制表位放编号，与中文期刊排法一致。"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.first_line_indent = Pt(0)
    pf.line_spacing = 1.5
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    pf.tab_stops.add_tab_stop(Cm(7.75), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Cm(15.5), WD_TAB_ALIGNMENT.RIGHT)
    p.add_run('\t')
    add_omml(p, latex)
    if num:
        r = p.add_run('\t(%s)' % num)
        R._fonts(r, R.SONG, R.LATIN)
        r.font.size = Pt(R.SZ_BODY)
    return p


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


def toc_domain(doc):
    """插入 Word TOC 域；打开文档时按 settings 里的 updateFields 自动刷新。"""
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    r = add_field(p, 'TOC \\o "1-3" \\h \\z \\u', '（打开后按 Ctrl+A 再 F9 更新目录）')
    R._fonts(r, R.SONG, R.LATIN)
    r.font.size = Pt(R.SZ_NOTE)


def enable_update_fields(doc):
    el = OxmlElement('w:updateFields')
    el.set(qn('w:val'), 'true')
    doc.settings.element.append(el)


def heading(doc, text, lvl, align, size, before, after, page_break=False):
    hp = doc.add_heading('', level=lvl)
    r = hp.add_run(R._symbols(text))
    r.bold = True
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor(0, 0, 0)
    R._fonts(r, R.HEI, R.LATIN)
    pf = hp.paragraph_format
    pf.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.keep_with_next = True
    pf.first_line_indent = Pt(0)
    pf.line_spacing = 1.5
    if page_break:
        pf.page_break_before = True
    return hp


def fig_caption(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(2)
    pf.space_after = Pt(10)
    r = p.add_run(R._symbols(text))
    r.font.size = Pt(R.SZ_NOTE)
    R._fonts(r, R.SONG, R.LATIN)


def add_table(doc, hdr, body):
    s = doc.sections[-1]
    avail = s.page_width.cm - s.left_margin.cm - s.right_margin.cm
    rows_txt = [hdr] + body
    size = R.SZ_TBL
    widths = R.plan_widths(rows_txt, avail)
    if not R.fits(rows_txt, widths, size):
        size = R.SZ_WIDE
        widths = R.plan_widths(rows_txt, avail, size)
    t = R.mk_table(doc, len(hdr))
    for j, h in enumerate(hdr):
        R.set_cell(t.rows[0].cells[j], h, True, size, widths[j])
    for row in body:
        cs = t.add_row().cells
        for j in range(len(hdr)):
            R.set_cell(cs[j], (row[j] if j < len(row) else '').replace('**', ''),
                       size=size, col_cm=widths[j])
    R.apply_widths(t, widths)
    R.three_line(t)
    for k, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
        if k == 0:
            trPr.append(OxmlElement('w:tblHeader'))
    gap = doc.add_paragraph()
    gap.paragraph_format.space_before = Pt(0)
    gap.paragraph_format.space_after = Pt(0)
    gap.paragraph_format.line_spacing = 1.0


def code_block(doc, rows):
    for k, row in enumerate(rows):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Cm(1.1)
        pf.first_line_indent = Pt(0)
        pf.line_spacing = 1.0
        pf.space_before = Pt(4 if k == 0 else 0)
        pf.space_after = Pt(4 if k == len(rows) - 1 else 0)
        r = p.add_run(row)
        r.font.size = Pt(9.5)
        R._fonts(r, R.SONG, R.MONO)


def add_md_paper(doc, md):
    lines = md.split('\n')
    i = 0
    after_table = False
    title_done = False
    subtitle_done = False
    preamble_done = False
    while i < len(lines):
        raw = lines[i]
        ln = raw.rstrip()
        s = ln.strip()

        if s.startswith('<!--'):          # md 注释不进 Word
            i += 1
            continue

        if ln.startswith(BT3):
            after_table = False
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith(BT3):
                block.append(lines[i].rstrip())
                i += 1
            i += 1
            code_block(doc, block)
            continue

        md_disp = DISPLAY.match(s)
        if md_disp:
            after_table = False
            math_line(doc, md_disp.group(1).strip(), md_disp.group(2))
            i += 1
            continue

        m = IMG.match(s)
        if m:
            after_table = False
            num, cap, path = m.groups()
            f = (MD.parent / path).resolve()
            if not f.exists():
                f = next((d / Path(path).name for d in FIGDIRS if (d / Path(path).name).exists()), None)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf = p.paragraph_format
            pf.space_before = Pt(8)
            pf.space_after = Pt(0)
            pf.keep_with_next = True
            if f:
                p.add_run().add_picture(str(f), width=Cm(14.6))
            else:
                p.add_run('[缺图 %s]' % path).font.size = Pt(10)
            fig_caption(doc, '图 %s · %s' % (num, cap))
            i += 1
            continue

        if s.startswith('|') and i + 1 < len(lines) and R.TAB.match(lines[i + 1].strip()):
            after_table = False
            hdr = [c.strip() for c in s.strip('|').split('|')]
            i += 2
            body = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                body.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            add_table(doc, hdr, body)
            after_table = True
            continue

        mt = TABCAP.match(s)
        if mt:
            R.caption(doc, s, bind=True)
            after_table = False
            i += 1
            continue

        if ln.startswith('# '):
            title_done = True
            i += 1
            continue
        if ln.startswith('## '):
            if title_done and not subtitle_done:
                subtitle_done = True
                if '摘要' not in s:
                    i += 1
                    continue
            txt = s.lstrip('#').strip()
            is_back = txt.startswith('参考文献') or txt.startswith('附录')
            if txt.startswith('摘要'):
                preamble_done = True
                heading(doc, txt, 2, WD_ALIGN_PARAGRAPH.CENTER, 14, 20, 8)
            elif txt == '目录':
                preamble_done = True
                heading(doc, txt, 2, WD_ALIGN_PARAGRAPH.CENTER, 14, 20, 10, page_break=True)
                toc_domain(doc)
            elif txt.startswith('图表清单') or txt.startswith('符号与缩略语'):
                preamble_done = True
                heading(doc, txt, 2, WD_ALIGN_PARAGRAPH.CENTER, 14, 0, 10, page_break=True)
            elif is_back:
                heading(doc, txt, 2, WD_ALIGN_PARAGRAPH.LEFT, 15, 14, 8, page_break=True)
            else:
                heading(doc, txt, 2, WD_ALIGN_PARAGRAPH.LEFT, 15, 14, 8)
            after_table = False
            i += 1
            continue
        if ln.startswith('### '):
            heading(doc, s.lstrip('#').strip(), 3, WD_ALIGN_PARAGRAPH.LEFT, 12, 10, 4)
            after_table = False
            i += 1
            continue

        if s.startswith('>'):
            if not preamble_done:
                i += 1
                continue
            p = doc.add_paragraph()
            pf = p.paragraph_format
            text = s.lstrip('> ').strip()
            if after_table:
                if not text.startswith('注：'):
                    text = '注：' + text
                pf.space_before = Pt(2)
                pf.space_after = Pt(8)
                pf.first_line_indent = Pt(0)
                rich_runs(p, text, size=R.SZ_TBL)
            else:
                pf.left_indent = Cm(0.6)
                pf.first_line_indent = Pt(0)
                pf.space_before = Pt(3)
                pf.space_after = Pt(3)
                rich_runs(p, text, size=R.SZ_NOTE)
            i += 1
            continue

        if ln.startswith('    ') and ln.strip():
            after_table = False
            block = []
            while i < len(lines) and lines[i].startswith('    ') and lines[i].strip():
                block.append(lines[i][4:].rstrip())
                i += 1
            code_block(doc, block)
            continue

        if REFLINE.match(s):
            after_table = False
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent = Cm(0.75)
            pf.first_line_indent = Cm(-0.75)
            pf.space_before = Pt(0)
            pf.space_after = Pt(2)
            pf.line_spacing = 1.25
            rich_runs(p, s, size=R.SZ_NOTE)
            i += 1
            continue

        if s.startswith('- '):
            after_table = False
            p = doc.add_paragraph(style='List Bullet')
            pf = p.paragraph_format
            pf.left_indent = Cm(0.75)
            pf.first_line_indent = Cm(-0.35)
            pf.line_spacing = 1.5
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            rich_runs(p, s[2:])
            i += 1
            continue

        if s and s not in ('---', '***'):
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.line_spacing = 1.5
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            R._indent_chars(p, 200)
            rich_runs(p, s)
            after_table = False
        i += 1


def cover(doc):
    for _ in range(5):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run('运动想象脑电的个体差异研究与分类算法设计')
    r.bold = True
    r.font.size = Pt(18)
    R._fonts(r, R.HEI, R.LATIN)
    for _ in range(6):
        doc.add_paragraph()
    for line, size in (('论文初稿', 14),
                       ('数据：BCI Competition IV-2a（9 人，4 类运动想象）', 11),
                       ('2026 年 11 月', 11)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(line)
        r.font.size = Pt(size)
        R._fonts(r, R.SONG, R.LATIN)
    doc.add_page_break()


def audit(path):
    z = zipfile.ZipFile(path)
    bad = z.testzip()
    doc = Document(path)
    n_tab, n_img = len(doc.tables), len(doc.inline_shapes)
    n_par = len(doc.paragraphs)
    missing = [p.text for p in doc.paragraphs if '缺图' in p.text]
    started = False
    body = []
    for p in doc.paragraphs:
        if p.style.name == 'Heading 2' and p.text.strip() == '摘要':
            started = True
            continue
        if started and p.style.name == 'Normal' and len(p.text) > 40:
            body.append(p)
    p0 = body[0] if body else None
    ind = None
    if p0 is not None:
        ppr = p0._p.pPr
        if ppr is not None and ppr.find(qn('w:ind')) is not None:
            ind = ppr.find(qn('w:ind')).get(qn('w:firstLineChars'))
    xml = z.read('word/document.xml').decode('utf-8')
    n_math = xml.count('<m:oMath>') + xml.count('<m:oMathPara>')
    n_toc = xml.count('TOC \\o')
    return dict(zip_bad=bad, paragraphs=n_par, tables=n_tab, images=n_img,
                equations=n_math, toc_fields=n_toc,
                missing_img=missing, body_indent_chars=ind,
                normal_size=doc.styles['Normal'].font.size.pt if doc.styles['Normal'].font.size else None,
                normal_spacing=doc.styles['Normal'].paragraph_format.line_spacing)


def strip_zwsp(doc):
    """清掉渲染库为英文断行插入的零宽空格：只改 w:t 文本节点，不碰公式与图片。"""
    n = 0
    for t in doc.element.body.iter(qn('w:t')):
        if t.text and '\u200b' in t.text:
            n += t.text.count('\u200b')
            t.text = t.text.replace('\u200b', '')
    return n


def backup_existing(path):
    """覆盖前把现有成稿另存一份：Word 里手改的格式只存在于 docx，重渲染会全部冲掉。"""
    if not path.exists():
        return None
    d = HERE / '_docx_backup'
    d.mkdir(exist_ok=True)
    dst = d / (path.stem + '_' + time.strftime('%Y%m%d-%H%M%S') + '.docx')
    shutil.copy2(str(path), str(dst))
    for f in sorted(d.glob('*.docx'))[:-20]:
        f.unlink()
    return dst


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bk = backup_existing(DOCX)
    md = MD.read_text(encoding='utf-8')
    doc = R.new_doc()          # 不生成页眉（用户 2026-11-03 指定删除）
    cp = doc.core_properties
    cp.title = '运动想象脑电的个体差异研究与分类算法设计'
    cp.subject = 'EEG 运动想象解码；个体差异；个体化解码；论文初稿'
    cp.author = '项目组'
    cover(doc)
    add_md_paper(doc, md)
    n_zw = strip_zwsp(doc)
    enable_update_fields(doc)
    doc.save(str(DOCX))
    doc.save(str(REL))
    a = audit(DOCX)
    print('backup_of_previous:', bk)
    print('saved:', DOCX)
    print('saved:', REL)
    print('bytes:', DOCX.stat().st_size)
    print('zwsp_removed           %d' % n_zw)
    for k, v in a.items():
        print('%-22s %s' % (k, v))


if __name__ == '__main__':
    main()
