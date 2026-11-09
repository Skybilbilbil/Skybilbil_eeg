# -*- coding: utf-8 -*-
"""把 `项目成果汇报材料.md` 渲染成 Word（复用 make_individual_diff_report.py 的渲染器）。"""
import importlib.util, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # = analysis/差异报告/
BASE = HERE.parents[1]                          # = MAtt_baseline/
OUT = BASE / 'results' / '汇报材料'              # 大体积产物（图 + docx；results/ 被 gitignore）
FIG = OUT / 'figs'
MD = HERE / '项目成果汇报材料.md'
DOCX = OUT / '项目成果汇报材料.docx'
# 渲染器本体就在同目录（与另四份差异材料同级）
GEN = HERE / 'make_individual_diff_report.py'

spec = importlib.util.spec_from_file_location('gen', GEN)
gen = importlib.util.module_from_spec(spec)
sys.modules['gen'] = gen
spec.loader.exec_module(gen)

OUT.mkdir(parents=True, exist_ok=True)
gen.FIGS = FIG                      # 渲染器按文件名在 FIGS 里取图（md 里写的是相对路径）

rmd = MD.read_text(encoding='utf-8')
ntab, nfig = gen.check_md(rmd)      # 渲染前自检：表题/图题编号、粗体配对
print('自检通过：表 %d 张，图 %d 张' % (ntab, nfig))

doc = gen.new_doc(u'项目成果汇报材料')
gen.add_md(doc, rmd)
gen.fit_check(doc, u'汇报材料')     # 排版自检（表格是否超宽等）
doc.save(str(DOCX))
# 同步一份到 analysis/差异报告/：与人读的差异材料同级，便于就地打开（*.docx 不进版本库）
import shutil
REL = HERE / '项目成果汇报材料.docx'
shutil.copyfile(DOCX, REL)
print('[OK]', DOCX, DOCX.stat().st_size, 'bytes')
print('[OK]', REL)