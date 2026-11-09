# -*- coding: utf-8 -*-
"""把差异文档渲染成 Word：个体化差异研究报告.md(含 8 图 + 附录 B 表) 与 个体差异关键数据与结论.md。
纯 CPU, 秒级。"""
import json, re, unicodedata
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE = Path(__file__).resolve().parents[2]   # = MAtt_baseline/（本脚本在 analysis/差异报告/ 下）
OUT = BASE / 'results' / 'individual_diff'
FIGS = OUT / 'figs'
MD = Path(__file__).resolve().parent / '个体化差异研究报告.md'   # 版本化文字与产出它的脚本同目录(仓库既有规则)
DOCX = OUT / '个体化差异研究报告.docx'
KEY_MD = Path(__file__).resolve().parent / '个体差异关键数据与结论.md'
KEY_DOCX = OUT / '个体差异关键数据与结论.docx'

for _f in ('/mnt/c/Windows/Fonts/simhei.ttf', r'C:\Windows\Fonts\simhei.ttf'):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 140
C = {'b': '#2e6fb7', 'o': '#e07b39', 'r': '#c0392b', 'g': '#4a934a', 'p': '#7d5ba6', 'k': '#333333'}


def _sub(ax, subj, vals, title, ylab, color, fmt='%.2f'):
    ax.bar(subj, vals, color=color)
    ax.set_title(title, fontsize=11); ax.set_ylabel(ylab, fontsize=9)
    ax.tick_params(labelsize=8); ax.axhline(0, color='k', lw=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt % v, ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)
    ax.grid(alpha=0.3, axis='y')


def figs(rows, rc, erd):
    from scipy.stats import spearmanr
    FIGS.mkdir(parents=True, exist_ok=True)
    subj = [r['subject'] for r in rows]
    g = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
    made = []

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    lab = ['非EA 静息', '非EA 任务', 'EA后 静息', 'EA后 任务']
    v = [rows[0]['subj_id_nonEA_rest'], rows[0]['subj_id_nonEA_task'],
         rows[0]['subj_id_EA_rest'], rows[0]['subj_id_EA_task']]
    ax.bar(lab, v, color=[C['g'], C['g'], C['r'], C['r']])
    ax.axhline(1 / 9, ls='--', color='k', lw=1)
    ax.text(2.4, 0.13, '随机 11.1%', fontsize=8)
    for i, x in enumerate(v):
        ax.text(i, x, '%.1f%%' % (x * 100), ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel('9 类被试识别准确率')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(FIGS / 'fig1_identity.png'); plt.close(fig)
    made.append('fig1_identity.png')

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    f = np.array(rc['freq']); cmap = plt.get_cmap('tab10')
    for i, y in enumerate(rc['psd_db']):
        ax.plot(f, y, lw=1.1, color=cmap(i % 10), label=rows[i]['subject'])
        iaf = rows[i]['iaf_hz']
        ax.plot([iaf], [np.interp(iaf, f, y)], 'v', color=cmap(i % 10), ms=6)
    ax.set_xlim(2, 32); ax.set_xlabel('频率 (Hz)'); ax.set_ylabel('功率谱密度 (dB)')
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / 'fig2_psd.png'); plt.close(fig)
    made.append('fig2_psd.png')

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    _sub(ax[0], subj, g('mu_focus_rel'), '全脑 α 峰空间集中度(峰/均值)', 'x', C['b'])
    _sub(ax[1], subj, g('c3c4_asym_db'), 'C4/C3 静息 α 不对称', 'dB', C['r'])
    _sub(ax[2], subj, g('rest_eff_rank'), '静息协方差有效空间秩', '维', C['g'])
    fig.tight_layout(); fig.savefig(FIGS / 'fig3_spatial.png'); plt.close(fig)
    made.append('fig3_spatial.png')

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    t = np.array(rc['t_s'])
    for i, y in enumerate(rc['mu_logpow']):
        ax[0].plot(t, np.mean(y, axis=0), lw=1.2, color=cmap(i % 10), label=rows[i]['subject'])
    ax[0].set_xlim(0.35, 3.65)          # 去掉滤波/平滑的边界伪影
    ax[0].axvline(1.0, ls='--', color='k', lw=1)
    ax[0].text(1.03, ax[0].get_ylim()[1], ' cue', fontsize=8, va='top')
    ax[0].set_xlabel('epoch 内时间 (s)'); ax[0].set_ylabel('mu 频带 log 功率(全脑平均)')
    ax[0].set_title('全脑 α 功率时间过程(9 人，4 类平均)', fontsize=10)
    ax[0].legend(fontsize=7, ncol=3); ax[0].grid(alpha=0.3)
    _sub(ax[1], subj, g('erd_peak_lat_s'), '全脑 α 谷底时刻(0.25s 栅格)', 's', C['p'])
    fig.tight_layout(); fig.savefig(FIGS / 'fig4_erd.png'); plt.close(fig)
    made.append('fig4_erd.png')

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for k, (xx, yy, ttl) in enumerate([
            ('alpha_pow_db', 'tangent_lda_within', 'D1 静息 α 绝对功率 × 切空间被试内 LDA'),
            ('iaf_hz', 'csp_within', 'D2 IAF × 被试内 CSP+LDA'),
            ('csp_within', 'deep70_acc', 'D3 被试内 CSP+LDA × 主协议深度模型逐被试精度')]):
        xs, ys = g(xx), g(yy)
        rho, p = spearmanr(xs, ys)
        ax[k].scatter(xs, ys, s=70, color=C['b'], zorder=3)
        for i, s in enumerate(subj):
            ax[k].annotate(s, (xs[i], ys[i]), fontsize=8, xytext=(4, 3),
                           textcoords='offset points')
        z = np.polyfit(xs, ys, 1); xr = np.linspace(xs.min(), xs.max(), 10)
        ax[k].plot(xr, np.polyval(z, xr), '--', color=C['r'], lw=1)
        ax[k].set_xlabel(xx); ax[k].set_ylabel(yy)
        ax[k].set_title('%s\nSpearman ρ=%.2f, 未校正 p=%.3f' % (ttl, rho, p), fontsize=10)
        ax[k].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / 'fig6_scatter.png'); plt.close(fig)
    made.append('fig6_scatter.png')

    phys = ['iaf_hz', 'alpha_pow_db', 'alpha_snr_db', 'mu_focus_rel', 'c3c4_asym_db',
            'rest_eff_rank', 'erd_mu_db', 'erd_beta_db', 'erd_peak_lat_s', 'erd_contra_db']
    perf = ['csp_within', 'csp_cross_sess', 'tangent_lda_within', 'receivability', 'session_gap', 'deep70_acc']
    R = np.zeros((len(phys), len(perf))); P = np.zeros_like(R)
    for i, a in enumerate(phys):
        for j, b in enumerate(perf):
            R[i, j], P[i, j] = spearmanr(g(a), g(b))
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(R, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(perf))); ax.set_xticklabels(perf, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(phys))); ax.set_yticklabels(phys, fontsize=8)
    for i in range(len(phys)):
        for j in range(len(perf)):
            ax.text(j, i, '%.2f' % R[i, j], ha='center', va='center', fontsize=7,
                    color='k' if abs(R[i, j]) < 0.6 else 'w',
                    fontweight='normal')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(FIGS / 'fig7_corrheat.png'); plt.close(fig)
    made.append('fig7_corrheat.png')

    x = np.arange(9); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].bar(x - w / 2, g('csp_within'), w, label='被试内(混合两 session)', color=C['b'])
    ax[0].bar(x + w / 2, g('csp_cross_sess'), w, label='跨 session', color=C['o'])
    ax[0].set_xticks(x); ax[0].set_xticklabels(subj); ax[0].legend(fontsize=8)
    ax[0].set_ylabel('CSP+LDA 准确率'); ax[0].grid(alpha=0.3, axis='y')
    ax[0].set_title('被试内(混合两 session) vs 跨 session', fontsize=10)
    _sub(ax[1], subj, g('session_gap'),
         'session_gap 列(9/9 为正，均值 +6.83pp)\n其中 +2.04pp 为训练量混杂，见 §3.7', '', C['r'], '%.3f')
    fig.tight_layout(); fig.savefig(FIGS / 'fig8_session.png'); plt.close(fig)
    made.append('fig8_session.png')

    tc, mech, P = erd['timecourse'], erd['mechanisms'], erd['per_subject']
    t = np.array(tc['t_event_s'])
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for key, lab, col in (('contra', '对侧 (左cue→C4 / 右cue→C3)', C['b']),
                          ('ipsi', '同侧', C['o'])):
        y = np.array(tc[key + '_db']); ci = np.array(tc[key + '_ci95'])
        ax[0].plot(t, y, '-o', ms=3.5, lw=1.4, color=col, label=lab)
        ax[0].fill_between(t, ci[:, 0], ci[:, 1], color=col, alpha=0.2)
    ax[0].axhline(0, color='k', lw=0.7); ax[0].axvline(0, ls=':', color='k', lw=0.9)
    ax[0].text(0.03, ax[0].get_ylim()[1], ' cue', fontsize=8, va='top')
    ax[0].set_xlabel('cue 后时间 (s)'); ax[0].set_ylabel('μ ERD (dB，负=抑制)')
    ax[0].set_title('对侧/同侧单电极 ERD 时间过程\n(9 人各自算 ERD 后平均，带=跨被试 95%CI)',
                    fontsize=10)
    ax[0].legend(fontsize=8, loc='lower left'); ax[0].grid(alpha=0.3)

    cas = [('全脑\n四类', mech['M3_brain_all_classes']), ('全脑\n仅手类', mech['M3_brain_hand_only']),
           ('中央ROI\n手类', mech['M2_central_hand']), ('对侧ROI', mech['M1_contra_roi']),
           ('同侧ROI', mech['M1_ipsi_roi']), ('中线\nfeet', mech['feet_midline']),
           ('中线\ntongue', mech['tongue_midline'])]
    xs = np.arange(len(cas))
    ax[1].bar(xs, [v for _, v in cas], color=[C['k'], C['b'], C['b'], C['b'], C['o'], C['g'], C['g']])
    ax[1].axhline(0, color='k', lw=0.7)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([k for k, _ in cas], fontsize=7.5)
    for i, (_, v) in enumerate(cas):
        ax[1].text(i, v, '%.2f' % v, ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)
    ax[1].set_ylabel('μ ERD (dB)'); ax[1].grid(alpha=0.3, axis='y')
    ax[1].set_title('口径级联：全脑四类 → 手类 → 中央 → 对侧\n(逐步去除类别/空间/偏侧三重稀释)', fontsize=10)

    bh = np.array([P[s]['brain_hand'] for s in subj]); d70 = g('deep70_acc')
    rho, pv = spearmanr(bh, d70)
    loo = [spearmanr(np.delete(bh, i), np.delete(d70, i))[0] for i in range(len(bh))]
    ax[2].scatter(bh, d70, s=70, color=C['b'], zorder=3)
    for i, s in enumerate(subj):
        ax[2].annotate(s, (bh[i], d70[i]), fontsize=8, xytext=(4, 3), textcoords='offset points')
    z = np.polyfit(bh, d70, 1); xr = np.linspace(bh.min(), bh.max(), 10)
    ax[2].plot(xr, np.polyval(z, xr), '--', color=C['r'], lw=1)
    ax[2].set_xlabel('brain_hand μ ERD (dB)'); ax[2].set_ylabel('deep70_acc')
    ax[2].set_title('手类 μ ERD 深度 × 主协议逐被试精度\nSpearman ρ=%.2f, 未校正 p=%.4f' % (rho, pv),
                    fontsize=10)
    ax[2].text(0.98, 0.03, '留一法 ρ∈[%.2f, %.2f]' % (min(loo), max(loo)),
               transform=ax[2].transAxes, ha='right', fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / 'fig5_erd_decomp.png'); plt.close(fig)
    made.append('fig5_erd_decomp.png')
    return made


IMG = re.compile(r'^!\[图(\d+)\]\(([^)]+)\)$')
CAP = re.compile(u'^\\*\\*表 (\\d+) \\u00b7 .+\\*\\*$')       # 表题(表格上方)
FIGCAP = re.compile(u'^\\*\\*图 (\\d+) \\u00b7 .+\\*\\*$')    # 图题(图片下方)
TAB = re.compile(u'^[|][\\s:|-]+[|]$')                        # markdown 表格的分隔行
BREAK = re.compile(u'^(\\d|\\u53c2|\\u9644)')                 # 需另起一页的"章"标题(摘要不另起)

# ================= 论文排版 =================
# 字号取中文字号系列：正文小四 12pt；题注/表注/引注五号 10.5pt；表格小五 9pt；超宽材料表 8pt
SONG, HEI, LATIN, MONO = '宋体', '黑体', 'Times New Roman', 'Consolas'
SZ_BODY, SZ_NOTE, SZ_TBL, SZ_CODE, SZ_WIDE = 12, 10.5, 9, 10, 8
# 本文档的层级映射：# = 论文题目(小二 18 居中)，## = 一级标题(三号 16)，
# ### = 二级(小三 15)，#### = 三级(四号 14) —— 全部黑体。三级若与正文同号就会糊成一片。
HEADS = {1: (HEI, 18, WD_ALIGN_PARAGRAPH.CENTER),
         2: (HEI, 16, WD_ALIGN_PARAGRAPH.LEFT),
         3: (HEI, 15, WD_ALIGN_PARAGRAPH.LEFT),
         4: (HEI, 14, WD_ALIGN_PARAGRAPH.LEFT)}
# 页边距：规范是装订侧留宽、天地的 2.5cm；A4 正文宽度随之为 15.5cm
MARGIN = dict(top=Cm(2.5), bottom=Cm(2.5), left=Cm(3.0), right=Cm(2.5))
FIG_W = Cm(15.0)
# 文档里的标记符号不是论文用件，只在 Word 渲染时换成中文措辞（md 源文件不动）
SYMBOLS = ((u'⚠', u'注：'), (u'⇒', u'故'))
# OOXML 规定 tblPr/tcPr/pPr 的子元素有固定次序，直接 append 会让 Word 报"内容有问题"
AFTER_TBL_BORDERS = ('w:shd', 'w:tblLayout', 'w:tblCellMar', 'w:tblLook',
                     'w:tblCaption', 'w:tblDescription')
AFTER_TC_BORDERS = ('w:shd', 'w:noWrap', 'w:tcMar', 'w:textDirection',
                   'w:tcFitText', 'w:vAlign', 'w:hideMark')
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'
ZWSP = chr(0x200B)             # 零宽空格：给 Word 一个断行点，不产生可见字符
LABEL_CUT = 26                 # 段首结构标签的「标签名+冒号」长度上限(字)，超过就不当标签
PAD_CM = 0.38                  # Word 单元格左右内边距合计
MIN_COL_CM = 0.9               # 任何列不窄于此：S01 这类三字符短码实测 0.874cm，再窄就被 Word 拆行
BODY_CM = 15.5                 # 纵向正文行宽(21 − 3.0 − 2.5)，正文段断行点的判定基准


def _symbols(text):
    for a, b in SYMBOLS:
        text = text.replace(a, b)
    return text


def _before(parent, el, after):
    """把 el 插到 parent 里第一个"必须排在它后面"的子元素之前。"""
    for tag in after:
        nxt = parent.find(qn(tag))
        if nxt is not None:
            nxt.addprevious(el)
            return
    parent.append(el)


def _fonts(run, east=SONG, latin=LATIN):
    """python-docx 的 font.name 只写 w:ascii/w:hAnsi，中文必须另补 w:eastAsia。"""
    run.font.name = latin
    run._element.rPr.rFonts.set(qn('w:eastAsia'), east)


def _style_fonts(st, east=SONG, latin=LATIN):
    st.font.name = latin
    rf = st.element.rPr.rFonts
    rf.set(qn('w:eastAsia'), east)
    for a in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme'):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]      # 主题字体优先级高于显式 w:ascii，不摘掉标题就仍是 Calibri Light


def _indent_chars(p, chars):
    """中文的"首行缩进 N 字"必须写 w:firstLineChars(随字号缩放)；只给磅值，字号一改就错位。"""
    p.paragraph_format.first_line_indent = Pt(SZ_BODY * chars / 100.0)
    p._p.pPr.get_or_add_ind().set(qn('w:firstLineChars'), str(chars))


def _border(side, sz):
    e = OxmlElement('w:%s' % side)
    e.set(qn('w:val'), 'single' if sz else 'none')
    e.set(qn('w:sz'), str(sz))
    e.set(qn('w:space'), '0')
    e.set(qn('w:color'), '000000')
    return e


def three_line(t, outer=12, inner=6):
    """学术三线表：1.5 磅顶线/底线 + 0.75 磅栏目线，无竖线、无内部横线（sz 单位是 1/8 磅）。"""
    tblPr = t._tbl.tblPr
    old = tblPr.find(qn('w:tblBorders'))
    if old is not None:
        tblPr.remove(old)
    el = OxmlElement('w:tblBorders')
    for side, sz in (('top', outer), ('bottom', outer), ('left', 0),
                     ('right', 0), ('insideH', 0), ('insideV', 0)):
        el.append(_border(side, sz))
    _before(tblPr, el, AFTER_TBL_BORDERS)
    for c in t.rows[0].cells:                      # 表头行下沿的栏目线
        tcPr = c._tc.get_or_add_tcPr()
        b = OxmlElement('w:tcBorders')
        b.append(_border('bottom', inner))
        _before(tcPr, b, AFTER_TC_BORDERS)


def mk_table(doc, ncols):
    t = doc.add_table(rows=1, cols=ncols)
    t.style = 'Normal Table'                        # 不用 Table Grid：全框线不是论文格式
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    return t


def _plain(text):
    """一段 md 最终落到 Word 的纯文本(去掉 ** 与 ` 标记、⚠/⇒ 换成中文)，用于量宽与断点判定。"""
    return (_only_label(_symbols(text).replace('<br>', '\n'))
            .replace('**', '').replace('`', ''))


def _zwsp_commas(text):
    """逗号后的断点是无条件插入的(见 _zwsp)，量宽时必须先算进去 —— 否则
    [−3.10,−0.36] 会被当成一个 13 字的不可断片段，列宽被高估、别处被压窄。"""
    return re.sub(u'(?<=,)', ZWSP, text)


def _zwsp(text, size, col_cm):
    """给 Word 断行点(零宽空格 U+200B)，只给"放不进这一列"的片段插：
      · 逗号后一律可断 —— 置信区间 [−3.10,−0.36] 这类长数字串靠它折行；
      · _ / - 只在片段本身宽过可用列宽时才可断 —— 放得下的标识符(alpha_pow_db)与日期
        (2026-10-04)保持整段，插了 Word 会在列宽不够时把它们从中间拆成两行。"""
    text = _zwsp_commas(text)
    lim = col_cm - PAD_CM
    return re.sub(r'\S+', lambda m: m.group(0) if _cm(m.group(0), size) <= lim
                  else re.sub(u'(?<=[_/-])', ZWSP, m.group(0)), text)


def _only_label(text):
    """正文只保留"结构标签"的加粗：段首粗体块且去标记后前 LABEL_CUT 字内出现「：」
    (之前无句号)时，只把冒号(含)以前加粗；参考文献键 [R1] 同样算标签。
    其余位置的 ** 一律去掉 —— 句中强调(数字、结论、对比词)不是版式信息，全文三分之一字加粗只是噪音。"""
    plain = text.replace('**', '')
    if not text.startswith('**'):
        return plain
    m = re.match(u'^(\\[[A-Za-z]\\d+\\])', plain)
    if not m:
        m = re.match(u'^[^。；\\n]{0,%d}：' % (LABEL_CUT - 1), plain)
    if not m or plain[:m.end()].count('`') % 2:
        return plain
    return '**' + plain[:m.end()] + '**' + plain[m.end():]


def _runs(p, text, size=None, east=SONG, col_cm=BODY_CM):
    """先按 `**` 配对切粗体，再在段内切 `行内代码`，最后按单星号切斜体(参考文献题名)。
    不用一条正则同时匹配：`**…`erd_*`…**` 这种嵌套会让粗体闭合失败、把 ** 原样落进 Word。
    另外做四件纯渲染层的替换(只改 Word，md 源文件不动)：
      ⚠→注：、⇒→故；  <br>→真换行；  放不进 col_cm 的片段在 / _ - 后与逗号后插零宽空格；
      句中 ** 强调降级为普通文字，只有段首结构标签保留加粗。"""
    sz = size or SZ_BODY
    text = _only_label(_symbols(text).replace('<br>', '\n'))
    for bi, chunk in enumerate(text.split('**')):
        if not chunk:
            continue
        bold = bi % 2 == 1
        for code_chunk in re.split(r'(`[^`]+`)', chunk):
            if not code_chunk:
                continue
            if len(code_chunk) > 2 and code_chunk[0] == code_chunk[-1] == '`':
                _add_run(p, _zwsp(code_chunk[1:-1], sz, col_cm), bold, east, MONO, size)
            elif '*' in code_chunk and code_chunk.count('*') % 2 == 0:
                for ii, seg in enumerate(code_chunk.split('*')):
                    if seg:
                        _add_run(p, _zwsp(seg, sz, col_cm), bold, east, LATIN, size,
                                 italic=ii % 2 == 1)
            else:
                _add_run(p, _zwsp(code_chunk, sz, col_cm), bold, east, LATIN, size)


def _add_run(p, text, bold, east, latin, size, italic=False):
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    _fonts(r, east, latin)
    if size:
        r.font.size = Pt(size)
    return r


def set_cell(cell, text, bold=False, size=SZ_TBL, col_cm=BODY_CM):
    # 新建表格的单元格本就只有一个空段落，不再用 cell.text='' 重置(那会留下一个空 run)
    p = cell.paragraphs[0]
    pf = p.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(1)
    pf.space_after = Pt(1)
    # 数字/短标签列居中，长文字列左对齐
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(text) <= 16 else WD_ALIGN_PARAGRAPH.LEFT
    _runs(p, text, size=size, col_cm=col_cm)
    if bold:
        for r in p.runs:
            r.bold = True


def caption(doc, text, bind=True):
    """题注(表题/图题)：整行加粗居中、五号。bind=True 时 keep_with_next —— 表题必须与它下面的
    表格同页；图题在图下方，不需要再往后绑。
    不走 _runs 的行内标记解析：题注里可能带 `erd_*` 这类含星号的内容，会让粗体闭合失败；
    但反引号不能原样落进 Word，剥掉。"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(6)
    pf.space_after = Pt(2)
    pf.keep_with_next = bind
    r = p.add_run(_symbols(text.strip().strip('*').replace('`', '').strip()))
    r.bold = True
    r.font.size = Pt(SZ_NOTE)
    _fonts(r)


def check_md(md):
    """渲染前的文档自检：
    ① 粗体标记必须成对（落单会让 Word 里出现野生的 **）；
    ② 表题编号必须恰为 1..N 且 N = markdown 表格数；
    ③ 图题编号必须恰为 1..M 且 M = 图片数，且每张图下方紧跟其图题（论文要求图题在下）。"""
    odd = [n for n, ln in enumerate(md.split('\n'), 1) if ln.count('**') % 2]
    assert not odd, ('粗体 ** 不配对，行号: %s' % odd)
    lines = md.split('\n')
    nums = [int(CAP.match(ln).group(1)) for ln in lines if CAP.match(ln)]
    ntab = sum(1 for k, ln in enumerate(lines) if ln.startswith('|')
               and (k == 0 or not lines[k - 1].startswith('|'))
               and k + 1 < len(lines) and TAB.match(lines[k + 1].strip()))
    assert nums == list(range(1, ntab + 1)), ('表题编号与表格对不上', nums, ntab)
    figs = [int(IMG.match(ln).group(1)) for ln in lines if IMG.match(ln)]
    fnums = [int(FIGCAP.match(ln).group(1)) for ln in lines if FIGCAP.match(ln)]
    assert figs == list(range(1, len(figs) + 1)), ('图号未按出现顺序', figs)
    assert fnums == figs, ('图题与图不是一一对应且同序', figs, fnums)
    for k, ln in enumerate(lines):
        if IMG.match(ln):
            nxt = [j for j in range(k + 1, min(k + 4, len(lines))) if FIGCAP.match(lines[j].strip())]
            assert nxt, ('图下方缺图题', ln)
    return ntab, len(figs)


def _atomic(txt):
    """按 Word 的断行规则切出"不可断片段"：CJK 逐字可断、空白(含 <br> 换行)与零宽空格可断。"""
    out, cur = [], ''
    for ch in txt:
        if unicodedata.east_asian_width(ch) in 'WF' or ch in ' \u200b\n\t':
            if cur:
                out.append(cur)
                cur = ''
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out or ['']


def _cm(t, pt):
    em = pt * 0.03527          # 1pt = 0.03527cm；CJK 全角按 1em，西文按 0.5em 估
    return sum(em * (1.0 if unicodedata.east_asian_width(c) in 'WF' else 0.5) for c in t)


def plan_widths(rows_txt, avail_cm, size=SZ_TBL):
    """按内容定列宽，不用等宽分列 —— 等宽会让 alpha_pow_db 断成两行、短标签列大片留白。
    每列宽度被夹在两条几何边界之间：
      下限 floor = 该列最长"不可断片段"宽度 + 内边距(低于它标识符就会被拆成两行)；
      上限 ceil  = 该列最长单元格的整段宽度 + 内边距(超过它只是多留白)。
    行宽余量按各列"还想再要多少"(ceil − floor) 成比例分配 —— 于是「候选成因 | 与本数据的关系」
    这类两列长文本表按内容分到不同宽度，而不是各占一半。
    必须在 set_cell 之前调用：断点插在哪取决于列宽，先有宽度才谈得上"放不放得下"。"""
    n = max(len(r) for r in rows_txt)
    tok, nat = [0.0] * n, [0.0] * n
    for row in rows_txt:
        for j, raw in enumerate(row):
            txt = _zwsp_commas(_plain(raw))
            if not txt.strip():
                continue
            for seg in _atomic(txt):
                tok[j] = max(tok[j], _cm(seg, size))
            nat[j] = max(nat[j], _cm(txt, size))
    floor = [max(tok[j] + PAD_CM, MIN_COL_CM) for j in range(n)]
    ceil = [max(floor[j], nat[j] + PAD_CM) for j in range(n)]
    if sum(floor) >= avail_cm:                 # 预算不够：所有列都到不了 floor
        if n * min(floor) >= avail_cm:         # 连最短列都保不住：只能等比压，溢出由 fit_check 报
            return [w * avail_cm / sum(floor) for w in floor]
        # 把"高过统一水平线 L"的列压到 L，短列(被试号 S01、数字 10.701)保住自己的 floor。
        # 直接按 floor 等比缩放会把这类必须成行的短片段一起压断 —— 长文本列反正要折行，让它们让步。
        lo, hi = min(floor), max(floor)
        for _ in range(50):
            mid = (lo + hi) / 2
            if sum(min(w, mid) for w in floor) < avail_cm:
                lo = mid
            else:
                hi = mid
        return [min(w, lo) for w in floor]
    want = floor[:]
    for _ in range(4):                         # 注水：按剩余需求比例分，到 ceil 的列退出
        free = [j for j in range(n) if want[j] < ceil[j] - 1e-6]
        room = avail_cm - sum(want)
        if not free or room <= 1e-6:
            break
        d = sum(ceil[j] - want[j] for j in free)
        for j in free:
            want[j] += min(room * (ceil[j] - want[j]) / d, ceil[j] - want[j])
    if avail_cm - sum(want) > 1e-6:            # 各列都到 ceil 了还有富余：等比摊平，别让表比正文窄
        k = avail_cm / sum(want)
        want = [w * k for w in want]
    return want


def apply_widths(t, widths):
    t.autofit = False                          # 固定列宽，不让 Word 按内容乱分
    for j, col in enumerate(t.columns):
        w = Cm(widths[j])
        col.width = w
        for cell in col.cells:
            cell.width = w


def fits(rows_txt, widths, size):
    """按给定列宽渲染后，是否每个"自然断点之间"的片段都放得进它所在的列。
    断点规则用 _zwsp 本体(与 set_cell 完全同一套)：放不下的长片段会在 _ / - 后插零宽空格，
    这里必须按插完的结果算，否则会把"会在下划线处折行的标识符"误判成放不下。
    放不下 ⇒ Word 会把数字/标识符从中间拆成两行(如 −1.726 拆成 −1.72 / 6)，必须避免。"""
    for row in rows_txt:
        for j, raw in enumerate(row):
            if j >= len(widths):
                continue
            lim = widths[j] - PAD_CM
            for seg in _atomic(_zwsp(_plain(raw), size, widths[j])):
                if _cm(seg, size) > lim + 1e-6:
                    return False
    return True


def fit_check(doc, tag):
    """每张表的最小占宽 vs 所在页的行宽。超了就报出来 —— 版能不能放下是几何问题，
    不该靠"我觉得横向页够了"这种没有依据的断言。"""
    avail = [(s.page_width - s.left_margin - s.right_margin) / 360000. for s in doc.sections]
    limits, si = [], 0                       # 段落里的 sectPr 结束上一节：横向节从它之后开始
    for ch in doc.element.body.iterchildren():
        if ch.tag == qn('w:tbl'):
            limits.append(avail[min(si, len(avail) - 1)])
        elif ch.tag == qn('w:p'):
            pPr = ch.find(qn('w:pPr'))
            if pPr is not None and pPr.find(qn('w:sectPr')) is not None:
                si += 1
    bad = []
    for i, t in enumerate(doc.tables):
        limit = limits[i] if i < len(limits) else avail[-1]
        tot = 0.0
        for col in t.columns:
            need = 0.0
            for cell in col.cells:
                sz = max([r.font.size.pt for r in cell.paragraphs[0].runs if r.font.size] or [SZ_BODY])
                for seg in _atomic(cell.text):
                    need = max(need, _cm(seg, sz))
            tot += need
        if tot > limit:
            bad.append(u'表%d %.1fcm>%.1fcm(%d列)' % (i + 1, tot, limit, len(t.columns)))
    print('[fit] %s: %s' % (tag, u'; '.join(bad) or u'全部表格可放入行宽'))
    return bad


def add_md(doc, md, break_chapters=True):
    """break_chapters=False 用于短文档：章前分页会让"每节只有几行"的文档出现大量半空页。
    另外把整行的斜体注释 *（…）* 改写成引注块 —— 原逻辑按 *（ 开头把整行丢掉，注释因此没进 Word。"""
    md = re.sub(u'(?m)^\\*（(.+)）\\*$', lambda m: u'> （' + m.group(1) + u'）', md)
    lines = md.split('\n'); i = 0; after_table = False
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith('```'):          # 围栏代码块(ASCII 分解图)按等宽逐行落,保持对齐
            after_table = False
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                p = doc.add_paragraph()
                pf = p.paragraph_format
                pf.space_before = pf.space_after = Pt(0)
                pf.line_spacing = 1.0
                pf.left_indent = Cm(1.0)
                p.add_run(lines[i].rstrip()).font.name = MONO
                i += 1
            i += 1; doc.add_paragraph(); continue
        m = IMG.match(ln.strip())
        if m:
            # 图片按文件名在 FIGS 里找：md 里写的是相对自身目录的真实路径
            # (`../../results/individual_diff/figs/figN_*.png`)，Markdown 查看器能显示，
            # 生成的 docx 仍从 run 目录取图。图题紧随其后(论文规范：图题在图下方)。
            after_table = False
            f = FIGS / Path(m.group(2)).name
            pic = doc.add_paragraph()
            pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pic.paragraph_format.keep_with_next = True      # 图与图题同页
            if f.exists():
                pic.add_run().add_picture(str(f), width=FIG_W)
            else:
                pic.add_run(u'[缺图 %s]' % m.group(2))
            i += 1; continue
        if ln.startswith('|') and i + 1 < len(lines) and TAB.match(lines[i + 1].strip()):
            hdr = [c.strip() for c in ln.strip('|').split('|')]
            i += 2; body = []
            while i < len(lines) and lines[i].startswith('|'):
                body.append([c.strip() for c in lines[i].strip('|').split('|')]); i += 1
            s = doc.sections[-1]
            avail = s.page_width.cm - s.left_margin.cm - s.right_margin.cm
            rows_txt = [hdr] + body
            size, widths = SZ_TBL, plan_widths(rows_txt, avail)
            if not fits(rows_txt, widths, size):    # 9pt 下数字/标识符会被拆行：降一档到 8pt 重排
                size, widths = SZ_WIDE, plan_widths(rows_txt, avail, SZ_WIDE)
            t = mk_table(doc, len(hdr))
            for j, h in enumerate(hdr):
                set_cell(t.rows[0].cells[j], h, True, size, widths[j])
            for row in body:
                cs = t.add_row().cells
                for j in range(len(hdr)):
                    # 表内只有表头加粗：表体的 ** 一律剥掉。段首"标签加粗"的规则是给正文用的，
                    # 落到单元格里会把「成因」列那种句首短语加粗(表 18 的 4 个格子)，成为视觉噪音。
                    set_cell(cs[j], (row[j] if j < len(row) else '').replace('**', ''),
                             size=size, col_cm=widths[j])
            apply_widths(t, widths)
            three_line(t)
            doc.add_paragraph()
            after_table = True
            continue
        if ln.startswith('#'):
            lvl = min(len(ln) - len(ln.lstrip('#')), 4)
            txt = ln.lstrip('#').strip()
            h = doc.add_heading(_symbols(txt), level=lvl)
            # 一级标题(本文档的"章"= ##)从新页起排；摘要紧跟题目，不分页
            if break_chapters and lvl == 2 and BREAK.match(txt) and not txt.startswith(u'摘要'):
                h.paragraph_format.page_break_before = True
            after_table = False
            i += 1; continue
        if ln.startswith('>'):
            p = doc.add_paragraph()
            pf = p.paragraph_format
            text = ln.lstrip('> ')
            if after_table:                       # 紧跟表格的说明段 = 表注，置于表下、顶格
                if not text.startswith(u'注：'):
                    text = u'注：' + text
                pf.space_before = Pt(2)
                pf.space_after = Pt(6)
            else:                                 # 引注块：中文不用斜体(宋体斜体是伪斜)
                pf.left_indent = Cm(0.75)
                pf.space_before = pf.space_after = Pt(3)
            _runs(p, text, size=SZ_NOTE)
            i += 1; continue
        if ln.startswith('- '):
            after_table = False
            if ln.startswith('- **['):    # 参考文献条目：悬挂缩进，不打项目符号
                p = doc.add_paragraph()
                pf = p.paragraph_format
                pf.left_indent = Cm(1.0)
                pf.first_line_indent = Cm(-1.0)
                pf.space_after = Pt(3)
                _runs(p, ln[2:])
            else:
                _runs(doc.add_paragraph(style='List Bullet'), ln[2:])
            i += 1; continue
        if CAP.match(ln) or FIGCAP.match(ln):
            caption(doc, ln, bind=bool(CAP.match(ln)))
            after_table = False
            i += 1; continue
        if ln.strip() and not ln.strip().startswith('*（') and ln.strip() not in ('---', '***'):
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.first_line_indent = Pt(SZ_BODY * 2)
            _indent_chars(p, 200)                # 中文规范是"缩进 2 字符"，随字号缩放
            after_table = False
            _runs(p, ln)
        i += 1


def _page_field(p):
    """页码用 Word 的 PAGE 域，而不是写死数字。"""
    for kind, txt in (('begin', None), (None, 'PAGE'), ('end', None)):
        r = p.add_run()
        r.font.size = Pt(SZ_TBL)
        _fonts(r)
        if kind:
            e = OxmlElement('w:fldChar')
            e.set(qn('w:fldCharType'), kind)
        else:
            e = OxmlElement('w:instrText')
            e.set(XML_SPACE, 'preserve')
            e.text = txt
        r._element.append(e)


def new_doc(header=''):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)          # A4
    sec.top_margin, sec.bottom_margin = MARGIN['top'], MARGIN['bottom']
    sec.left_margin, sec.right_margin = MARGIN['left'], MARGIN['right']   # 装订侧留宽
    st = doc.styles['Normal']
    _style_fonts(st)
    st.font.size = Pt(SZ_BODY)
    st.font.color.rgb = RGBColor(0, 0, 0)
    pf = st.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for lvl, (east, size, align) in HEADS.items():
        h = doc.styles['Heading %d' % lvl]
        _style_fonts(h, east)
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.italic = False
        h.font.color.rgb = RGBColor(0, 0, 0)
        hp = h.paragraph_format
        hp.alignment = align
        hp.line_spacing = 1.5
        hp.first_line_indent = Pt(0)
        hp.space_before = Pt(14 if lvl == 1 else 10)
        hp.space_after = Pt(8 if lvl == 1 else 6)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _page_field(fp)
    if header:
        hp2 = sec.header.paragraphs[0]
        hp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = hp2.add_run(header)
        r.font.size = Pt(SZ_TBL)
        _fonts(r)
    return doc


def landscape(doc):
    """附录 B 那张 29 列材料表在纵向 A4 上必然压不成形，按论文惯例单独排横向页。"""
    s = doc.add_section(WD_SECTION.NEW_PAGE)
    s.orientation = WD_ORIENT.LANDSCAPE
    s.page_width, s.page_height = Cm(29.7), Cm(21.0)
    s.left_margin = s.right_margin = Cm(1.5)
    s.top_margin = s.bottom_margin = Cm(1.8)
    return s


def material_tables(doc, rows, first_num):
    """附录 B 的 29 列拆成两张横向表：29 列均分只有 0.92cm/列(算术上塞得进，但每列一行只放得下
    一个词片段，那不叫排得下)。按测量通路分组 = 生理侧 18 列(1.48cm) / 判别与身份侧 12 列(2.22cm)。
    列集合用断言兜住：拆分只许重排，不许丢列。"""
    cols = list(rows[0].keys())
    phys = [c for c in cols if c == 'subject' or c in (
        'iaf_hz', 'alpha_pow_db', 'alpha_snr_db', 'mu_focus_ch', 'mu_focus_rel',
        'c3c4_asym_db', 'rest_eff_rank') or c.startswith('erd_')]
    disc = [c for c in cols if c not in phys]
    # 不变量：两张表合起来必须恰好覆盖 29 列，不丢也不重(subject 作为行标签在两表各出现一次是允许的)
    assert set(phys) | set(disc) == set(cols), set(cols) - set(phys) - set(disc)
    assert not set(phys) & set(disc) and 'subject' in phys, (phys, disc)
    sec = doc.sections[-1]
    avail = (sec.page_width - sec.left_margin - sec.right_margin) / 360000.
    for n, (heads, title) in enumerate((
            (phys, u'生理侧读数（非 EA 通路）：静息 α 六项 + 十项已作废 erd_* 列，后者留档勿引用'),
            (disc, u'判别、身份与 session 读数（EA 通路）：四个判别基线、主协议逐被试精度、'
                   u'session 落差与身份可分性'))):
        cols_n = heads[:] if 'subject' in heads else ['subject'] + heads
        caption(doc, u'表 %d · 逐被试材料表 (%s) %s，共 %d 列 × %d 人' % (
            first_num + n, chr(ord('a') + n), title, len(cols_n), len(rows)))
        rows_txt = [cols_n] + [[str(r.get(h, '')) for h in cols_n] for r in rows]
        widths = plan_widths(rows_txt, avail, SZ_WIDE)
        t = mk_table(doc, len(cols_n))
        for j, h in enumerate(cols_n):
            set_cell(t.rows[0].cells[j], h, True, SZ_WIDE, widths[j])
        for row_txt in rows_txt[1:]:
            cs = t.add_row().cells
            for j, txt in enumerate(row_txt):
                set_cell(cs[j], txt, size=SZ_WIDE, col_cm=widths[j])
        apply_widths(t, widths)
        three_line(t)
        doc.add_paragraph()
    return first_num + 2


def main():
    rows = json.load(open(OUT / 'subject_table.json', encoding='utf-8'))
    rc = json.load(open(OUT / 'raw_curves.json', encoding='utf-8'))
    erd = json.load(open(BASE / 'results' / 'erd_decomposition' / 'erd_decomposition.json',
                         encoding='utf-8'))
    made = figs(rows, rc, erd)
    doc = new_doc(u'个体化差异研究报告')
    rmd = MD.read_text(encoding='utf-8')
    ntab, nfig = check_md(rmd)
    add_md(doc, rmd)
    landscape(doc)
    material_tables(doc, rows, ntab + 1)
    fit_check(doc, u'报告')
    doc.save(str(DOCX))
    print('[OK]', DOCX)
    print('[OK] %d figs, %d+%d captions, 附录材料表=表 %d/%d' % (len(made), ntab, nfig, ntab + 1, ntab + 2))

    kd = new_doc(u'个体差异关键数据与结论')
    kmd = KEY_MD.read_text(encoding='utf-8')
    ktab, knfig = check_md(kmd)
    add_md(kd, kmd, break_chapters=False)   # 短文档：7 节各几行，章前分页会造出半空页
    fit_check(kd, u'关键数据')
    print('[OK] key doc: %d captions, %d figures' % (ktab, knfig))
    kd.save(str(KEY_DOCX))
    print('[OK]', KEY_DOCX)


if __name__ == '__main__':
    main()