# -*- coding: utf-8 -*-
"""生成「8-30Hz + 无 ICA + 有 EA」数据 —— 只省 ICA 的干净对照。

对照(主协议 九人混合 70/15/15):
  现行 8-30 + ICA + EA = 58.82 (3 seed)
  本份 8-30 + 无ICA + EA = ?      <- 若 ≈58.82 则 ICA 确定可省
管线:原始 .mat -> 8-30Hz -> (不做 ICA) -> epoch [-1.0,3.5]s -> 逐被试 EA -> pkl
"""
import pickle
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/lenovo/EEG_Analysis/Riemannian_Trajectories')
sys.path.insert(0, str(BASE / 'preprocess'))

import mne                                              # noqa: E402
mne.set_log_level('ERROR')
from preprocessing import select_channels_physionetmi, epochs as mk_epochs  # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy        # noqa: E402
from moabb.datasets import BNCI2014_001                 # noqa: E402

OUT = BASE / 'saved_data' / 'preprocessed_data_bci2a_noica_ea.pkl'
NAMES = ['left', 'right', 'feet', 'hands']


def prep(raw):
    r = raw.copy()
    picks = mne.pick_types(r.info, eeg=True, exclude=[])
    r.filter(l_freq=8.0, h_freq=30.0, picks=picks, verbose=False)
    r.drop_channels([ch for ch in r.ch_names if 'EOG' in ch.upper()])
    r = select_channels_physionetmi(r)
    dsc = np.array([('hands' if str(x).strip().lower() == 'tongue'
                     else str(x).strip().lower()) for x in r.annotations.description], dtype=object)
    r.annotations.description = dsc
    return mk_epochs(r, tmin=-1.0, tmax=3.5, event_config=None)


def main():
    ds = BNCI2014_001()
    d = ds.get_data(subjects=list(range(1, 10)))
    out = {('aligned_%s_epochs_list' % n): [] for n in NAMES}
    meta = []
    for subj in range(1, 10):
        per = {n: [] for n in NAMES}
        for sess in d[subj].values():
            for r in sess.values():
                four = prep(r)
                for n, ep in zip(NAMES, four):
                    per[n].append(np.asarray(ep.get_data(), dtype=np.float32))
        arrs = {n: np.concatenate(per[n], 0) for n in NAMES}
        allx = np.concatenate([arrs[n] for n in NAMES], 0)
        P = compute_ea_matrix_from_numpy(allx.astype(np.float64))
        for n in NAMES:
            out['aligned_%s_epochs_list' % n].append(
                apply_ea_to_numpy(arrs[n].astype(np.float64), P).astype(np.float32))
        meta.append({'ea_matrix': P.tolist()})
        print('  S%02d ok: %s' % (subj, out['aligned_left_epochs_list'][-1].shape), flush=True)
    out['subject_metadata'] = meta
    with open(OUT, 'wb') as f:
        pickle.dump(out, f)
    print('[OK] %s (%.1f MB)' % (OUT, OUT.stat().st_size / 1e6))


if __name__ == '__main__':
    main()
