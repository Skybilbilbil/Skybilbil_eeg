# -*- coding: utf-8 -*-
"""把 个体化差异研究报告.md 渲染成 Word(含图)。纯 CPU, 秒级。"""
import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results' / 'individual_diff'
FIGS = OUT / 'figs'
MD = BASE / '个体化差异研究报告.md'
DOCX = OUT / '个体化差异研究报告.docx'

for _f in ('/mnt/c/Windows/Fonts/simhei.ttf',):
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


def figs(rows, rc):
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
    ax.set_title('图1 · 个体差异有多大：单次试次协方差即可辨认被试', fontsize=11)
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
    ax.set_title('图2 · 九名被试的静息功率谱(非 EA)；三角=各自 IAF，跨 8.07–11.37 Hz', fontsize=11)
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / 'fig2_psd.png'); plt.close(fig)
    made.append('fig2_psd.png')

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    _sub(ax[0], subj, g('mu_focus_rel'), '全脑 α 峰空间集中度(峰/均值)', 'x', C['b'])
    _sub(ax[1], subj, g('c3c4_asym_db'), 'C4/C3 静息 α 不对称', 'dB', C['r'])
    _sub(ax[2], subj, g('rest_eff_rank'), '静息协方差有效空间秩', '维', C['g'])
    fig.suptitle('图3 · 空间形态的个体差异(非 EA)', fontsize=12)
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
    fig.suptitle('图4 · 任务反应的时间差异(非 EA)', fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / 'fig4_erd.png'); plt.close(fig)
    made.append('fig4_erd.png')

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (xx, yy, ttl) in enumerate([
            ('alpha_pow_db', 'tangent_lda_within', 'D1 静息 α 绝对功率 × 切空间被试内 LDA'),
            ('iaf_hz', 'csp_within', 'D2 IAF × 被试内 CSP+LDA')]):
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
    fig.suptitle('图5 · 差异与判别性能的关系(探索性，n=9)', fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / 'fig5_scatter.png'); plt.close(fig)
    made.append('fig5_scatter.png')

    phys = ['iaf_hz', 'alpha_pow_db', 'alpha_snr_db', 'mu_focus_rel', 'c3c4_asym_db',
            'rest_eff_rank', 'erd_mu_db', 'erd_beta_db', 'erd_peak_lat_s', 'erd_contra_db']
    perf = ['csp_within', 'csp_cross_sess', 'tangent_lda_within', 'receivability', 'session_gap']
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
    ax.set_title('图6 · 生理 × 判别 Spearman 相关(n=9)\n未做多重比较校正；仅作探索性描述', fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(FIGS / 'fig6_corrheat.png'); plt.close(fig)
    made.append('fig6_corrheat.png')

    x = np.arange(9); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].bar(x - w / 2, g('csp_within'), w, label='被试内(同 session)', color=C['b'])
    ax[0].bar(x + w / 2, g('csp_cross_sess'), w, label='跨 session', color=C['o'])
    ax[0].set_xticks(x); ax[0].set_xticklabels(subj); ax[0].legend(fontsize=8)
    ax[0].set_ylabel('CSP+LDA 准确率'); ax[0].grid(alpha=0.3, axis='y')
    ax[0].set_title('同 session vs 跨 session', fontsize=10)
    _sub(ax[1], subj, g('session_gap'), 'session 落差(9/9 为正，均值 +4.7pp)', '', C['r'], '%.3f')
    fig.suptitle('图7 · 跨 session 稳定性(另一独立差异来源)', fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / 'fig7_session.png'); plt.close(fig)
    made.append('fig7_session.png')
    return made


SPLIT = re.compile('([*][*][^*]+[*][*]|[`][^`]+[`])')
IMG = re.compile(r'^!\[([^]]*)\]\(([^)]+)\)$')


def _runs(p, text):
    for seg in SPLIT.split(text):
        if not seg:
            continue
        if seg.startswith('**') and seg.endswith('**'):
            p.add_run(seg[2:-2]).bold = True
        elif seg.startswith('`') and seg.endswith('`'):
            p.add_run(seg[1:-1]).font.name = 'Consolas'
        else:
            p.add_run(seg)


def set_cell(cell, text, bold=False):
    cell.text = ''
    _runs(cell.paragraphs[0], text)
    if bold:
        for r in cell.paragraphs[0].runs:
            r.bold = True


def add_md(doc, md, Inches):
    lines = md.split('\n'); i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        m = IMG.match(ln.strip())
        if m:
            p = FIGS / m.group(2)
            if p.exists():
                doc.add_picture(str(p), width=Inches(6.3))
            else:
                doc.add_paragraph('[缺图 %s]' % m.group(2))
            i += 1; continue
        if ln.startswith('|') and i + 1 < len(lines) and re.match(r'^[|][\s:|-]+[|]$', lines[i + 1].strip()):
            hdr = [c.strip() for c in ln.strip('|').split('|')]
            i += 2; body = []
            while i < len(lines) and lines[i].startswith('|'):
                body.append([c.strip() for c in lines[i].strip('|').split('|')]); i += 1
            t = doc.add_table(rows=1, cols=len(hdr)); t.style = 'Table Grid'
            for j, h in enumerate(hdr):
                set_cell(t.rows[0].cells[j], h, True)
            for row in body:
                cs = t.add_row().cells
                for j in range(len(hdr)):
                    set_cell(cs[j], row[j] if j < len(row) else '')
            doc.add_paragraph(); continue
        if ln.startswith('#'):
            lvl = len(ln) - len(ln.lstrip('#'))
            doc.add_heading(ln.lstrip('#').strip(), level=min(lvl, 4)); i += 1; continue
        if ln.startswith('>'):
            doc.add_paragraph().add_run(ln.lstrip('> ')).italic = True
            i += 1; continue
        if ln.startswith('- '):
            _runs(doc.add_paragraph(style='List Bullet'), ln[2:]); i += 1; continue
        if ln.strip() and not ln.strip().startswith('*（'):
            _runs(doc.add_paragraph(), ln)
        i += 1


def main():
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.oxml.ns import qn
    rows = json.load(open(OUT / 'subject_table.json', encoding='utf-8'))
    rc = json.load(open(OUT / 'raw_curves.json', encoding='utf-8'))
    made = figs(rows, rc)
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.8)
    st = doc.styles['Normal']
    st.font.name = 'Microsoft YaHei'; st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    add_md(doc, MD.read_text(encoding='utf-8'), Inches)
    doc.add_heading('附录 B · 逐被试材料表', level=1)
    cols = list(rows[0].keys())
    t = doc.add_table(rows=1, cols=len(cols)); t.style = 'Table Grid'
    for j, c in enumerate(cols):
        set_cell(t.rows[0].cells[j], c, True)
    for r in rows:
        cs = t.add_row().cells
        for j, c in enumerate(cols):
            set_cell(cs[j], str(r.get(c, '')))
    doc.save(str(DOCX))
    print('[OK]', DOCX)
    print('[OK] %d figs' % len(made))


if __name__ == '__main__':
    main()