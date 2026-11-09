# -*- coding: utf-8 -*-
import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from utils.functions import load_matt_model

WIN, STEP = 375, 188
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)
RUNS = {'seed42': ('verify_samepool_70_15_15', 'verify70'),
        'seed2024': ('verify70_seed2024', 'v70s2024'),
        'seed7': ('verify70_seed7', 'v70s7')}
SUBJECTS = ['S02', 'S04', 'S05', 'S06']
SHOTS_PER_CLASS = [2, 5, 10, 20]
REPEATS = 20


def extract(net, X):
    net.eval()
    feats, logits = [], []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            xb = torch.Tensor(X[start:start + 256]).unsqueeze(1).to(DEVICE)
            feats.append(net.get_features(xb).cpu().numpy())
            logits.append(net(xb).cpu().numpy())
    return np.concatenate(feats), np.concatenate(logits)


def fit_probe(X_cal, y_cal, X_te, alpha=10.0):
    scaler = StandardScaler().fit(X_cal)
    clf = RidgeClassifier(alpha=alpha, class_weight='balanced')
    clf.fit(scaler.transform(X_cal), y_cal)
    return clf.predict(scaler.transform(X_te))


def fit_lda(X_cal, y_cal, X_te):
    scaler = StandardScaler().fit(X_cal)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(scaler.transform(X_cal), y_cal)
    return clf.predict(scaler.transform(X_te))


def fit_mlp(X_cal, y_cal, X_te):
    scaler = StandardScaler().fit(X_cal)
    clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=1000,
                        early_stopping=True, n_iter_no_change=20, random_state=0)
    clf.fit(scaler.transform(X_cal), y_cal)
    return clf.predict(scaler.transform(X_te))


def draw_balanced(y, shots, rng):
    idx = []
    for c in range(4):
        pool = np.where(y == c)[0]
        if len(pool) < shots:
            return None
        idx.extend(rng.choice(pool, size=shots, replace=False).tolist())
    return np.asarray(idx)


def main():
    subject_data, _ = load_preprocessed_dataset(
        str(BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_tr, y_tr, s_tr = ret[0], ret[1], ret[6]
    X_va, y_va, s_va = ret[2], ret[3], ret[7]
    X_te, y_te, s_te = ret[4], ret[5], ret[8]
    ref = json.loads((BASE / 'results' / 'individual_diff' /
                      'subject_table.json').read_text(encoding='utf-8'))
    ref = {r['subject']: r for r in ref}
    out = {'subjects': SUBJECTS, 'shots_per_class': SHOTS_PER_CLASS,
           'repeats': REPEATS, 'runs': {}}
    for seed, (dirname, tag) in RUNS.items():
        ck = BASE / 'results' / dirname / ('%s_cross_mix_best_acc.pt' % tag)
        net = load_matt_model(str(ck), WIN, STEP, ARGS).to(DEVICE)
        out['runs'][seed] = {}
        for sub_name in SUBJECTS:
            s = int(sub_name[1:]) - 1
            mtr, mte, mva = s_tr == s, s_te == s, s_va == s
            Xc, yc = X_tr[mtr], y_tr[mtr]
            Xn = np.concatenate([X_tr[mtr], X_va[mva]], axis=0)
            yn = np.concatenate([y_tr[mtr], y_va[mva]], axis=0)
            Xt, yt = X_te[mte], y_te[mte]
            F_cal, _ = extract(net, Xc)
            F_non, _ = extract(net, Xn)
            F_te, log_te = extract(net, Xt)
            acc_shared = float((log_te.argmax(1) == yt).mean())
            head = {}
            base_seed = 1000 + 100 * list(RUNS).index(seed) + s
            rng_master = np.random.RandomState(base_seed)
            for shots in SHOTS_PER_CLASS:
                vals = []
                for _ in range(REPEATS):
                    rng = np.random.RandomState(rng_master.randint(0, 2 ** 31 - 1))
                    ci = draw_balanced(yc, shots, rng)
                    pred = fit_probe(F_cal[ci], yc[ci], F_te)
                    vals.append(float((pred == yt).mean()))
                head[str(shots)] = {'mean': float(np.mean(vals)),
                                    'std': float(np.std(vals, ddof=1))}
            pred_all = fit_probe(F_non, yn, F_te)
            pred_lda = fit_lda(F_non, yn, F_te)
            pred_mlp = fit_mlp(F_non, yn, F_te)
            head['all_non_test'] = {'mean': float((pred_all == yt).mean())}
            head['all_non_test_lda'] = {'mean': float((pred_lda == yt).mean())}
            head['all_non_test_mlp'] = {'mean': float((pred_mlp == yt).mean())}
            for alpha in (0.1, 1.0, 100.0):
                pred_a = fit_probe(F_non, yn, F_te, alpha=alpha)
                head['all_non_test_ridge_%g' % alpha] = {
                    'mean': float((pred_a == yt).mean())}
            out['runs'][seed][sub_name] = {
                'n_cal': int(len(yc)), 'n_test': int(len(yt)),
                'shared_model': acc_shared, 'frozen_head': head,
                'within_csp': float(ref[sub_name]['csp_within']),
                'within_tangent_lda': float(ref[sub_name]['tangent_lda_within'])}
        del net

    out['summary'] = {}
    for sub_name in SUBJECTS:
        summary = {}
        for metric in ('shared_model', 'within_csp', 'within_tangent_lda'):
            vals = [out['runs'][seed][sub_name][metric] for seed in RUNS]
            summary[metric] = float(np.mean(vals))
        summary['frozen_head'] = {}
        for key in ('2', '5', '10', '20', 'all_non_test',
                    'all_non_test_lda', 'all_non_test_mlp'):
            vals = [out['runs'][seed][sub_name]['frozen_head'][key]['mean']
                    for seed in RUNS]
            summary['frozen_head'][key] = {
                'mean': float(np.mean(vals)),
                'seed_std': float(np.std(vals, ddof=1))}
        out['summary'][sub_name] = summary

    outdir = BASE / 'results' / 'low_erd_frozen_probe'
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'low_erd_frozen_probe.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
