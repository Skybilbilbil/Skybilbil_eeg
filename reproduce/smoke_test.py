# -*- coding: utf-8 -*-
"""冒烟测试：数据加载与窗口数计算。临时。"""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / 'MAtt_baseline'))
sys.path.insert(0, str(BASE.parent / 'preprocess'))

import exp_compare as E
from mAtt.mAtt import E2R

sd_o = E.load_official_data(add_band=True, add_ea=True)
print('official_band_ea T =', sd_o[0][0].shape, 'M =', E2R.n_windows(562, 250, 125))

sd_self = E.load_self_data()
print('self T =', sd_self[0][0].shape, 'M =', E2R.n_windows(1125, 250, 125))
img = E.cut_imagonly(sd_self, 0.5, 3.5)
print('self imagonly T =', img[0][0].shape, 'M =', E2R.n_windows(img[0][0].shape[-1], 250, 125))

# 实验3 窗口数
T_self = sd_self[0][0].shape[-1]
for nwin, win_sec in [(8, 1.0), (5, 2.5), (3, 3.5)]:
    M = E2R.n_windows(T_self, int(round(win_sec * 250)), int(round(0.5 * 250)))
    print(f'experiment3 win={win_sec}s → M={M}')