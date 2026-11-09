// 三个方向上的试次间一致性: 相位 / 幅值(包络) / 整体波形, 各自带零假设
const fs = require("fs"), zlib = require("zlib");
function load(f) {
  const buf = fs.readFileSync(f); let e = buf.length - 22;
  while (buf.readUInt32LE(e) !== 0x06054b50) e--;
  const n = buf.readUInt16LE(e + 10); let p = buf.readUInt32LE(e + 16); const T = {};
  for (let i = 0; i < n; i++) {
    const nl = buf.readUInt16LE(p + 28), el = buf.readUInt16LE(p + 30), cl = buf.readUInt16LE(p + 32), lho = buf.readUInt32LE(p + 42);
    const name = buf.toString("utf8", p + 46, p + 46 + nl); let q = lho;
    const meth = buf.readUInt16LE(q + 8), cs = buf.readUInt32LE(q + 18), qn = buf.readUInt16LE(q + 26), qe = buf.readUInt16LE(q + 28);
    let d = buf.slice(q + 30 + qn + qe, q + 30 + qn + qe + cs);
    T[name] = meth === 8 ? zlib.inflateRawSync(d) : d; p += 46 + nl + el + cl;
  }
  return T;
}
function meta(raw) {
  const v = raw[6], hl = v === 1 ? raw.readUInt16LE(8) : raw.readUInt32LE(8);
  const h = raw.toString("ascii", v === 1 ? 10 : 12, v === 1 ? 10 + hl : 12 + hl);
  return { shape: h.match(/'shape':\s*\(([^)]*)\)/)[1].split(",").map(x => x.trim()).filter(Boolean).map(Number), off: v === 1 ? 10 + hl : 12 + hl };
}
function trials(f, cls, ch) {
  const F = load(`D:/EEG_Data/三分类数据集/Weibo2014_3class/Weibo2014_MI_3class_${f}.npz`);
  const L = meta(F["labels.npy"]), D = meta(F["data.npy"]);
  const [N, NC, NT] = D.shape, out = [];
  for (let i = 0; i < N; i++) if (F["labels.npy"].readInt32LE(L.off + i * 8) === cls) {
    const a = new Float64Array(NT);
    for (let t = 0; t < NT; t++) a[t] = F["data.npy"].readDoubleLE(D.off + ((i * NC + ch) * NT + t) * 8);
    out.push(a);
  }
  return { x: out, NT };
}
const FS = 200, CYC = 7, STEP = 10;
const bands = { mu: [], beta: [] };
for (let f = 8; f <= 13.01; f += 0.5) bands.mu.push(f);
for (let f = 15; f <= 30.01; f += 0.5) bands.beta.push(f);
function wave(sig, fr) {
  const NT = sig.length, nT = Math.floor(NT / STEP);
  return fr.map(f => {
    const sg = CYC / (2 * Math.PI * f), hw = Math.ceil(3.5 * sg * FS);
    const K = []; for (let d = -hw; d <= hw; d++) { const g = Math.exp(-d * d / (2 * (sg * FS) ** 2)); K.push([g * Math.cos(2 * Math.PI * f * d / FS), g * Math.sin(2 * Math.PI * f * d / FS)]); }
    const row = new Array(nT);
    for (let ti = 0; ti < nT; ti++) { const t = ti * STEP; let re = 0, im = 0; for (let j = 0; j < K.length; j++) { const tt = t + j - hw; if (tt < 0 || tt >= NT) continue; re += sig[tt] * K[j][0]; im += sig[tt] * K[j][1]; } row[ti] = [re, im]; }
    return row;
  });
}
const pearson = (a, b) => { const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n; let sa = 0, sb = 0, sp = 0; for (let i = 0; i < n; i++) { const da = a[i] - ma, db = b[i] - mb; sa += da * da; sb += db * db; sp += da * db; } return sa && sb ? sp / Math.sqrt(sa * sb) : NaN; };
const envOf = (Z, fr) => { const n = Z[0].length; return Array.from({ length: n }, (_, ti) => { let s = 0; for (let fi = 0; fi < fr.length; fi++) s += Math.hypot(Z[fi][ti][0], Z[fi][ti][1]); return s / fr.length; }); };
const powOf = (Z, fr) => { const n = Z[0].length; return Array.from({ length: n }, (_, ti) => { let s = 0; for (let fi = 0; fi < fr.length; fi++) s += Z[fi][ti][0] ** 2 + Z[fi][ti][1] ** 2; return s / fr.length; }); };
const itcOf = Z => { const n = Z[0].length; return Array.from({ length: n }, (_, ti) => { let re = 0, im = 0; for (const z of Z) { const m = Math.hypot(z[ti][0], z[ti][1]) || 1e-12; re += z[ti][0] / m; im += z[ti][1] / m; } return Math.hypot(re, im) / Z.length; }); };
const win = (a, t0, t1) => { const s = a.slice(Math.floor(t0 * 20), Math.floor(t1 * 20)); return s.reduce((x, y) => x + y, 0) / s.length; };
const avg = a => a.reduce((x, y) => x + y, 0) / a.length;

const CH = { C3: 0, C4: 2 };
const cells = [["001", "Left", 1, "C4"], ["001", "Right", 2, "C3"], ["001", "Rest", 0, "C3"], ["002", "Left", 1, "C4"], ["002", "Right", 2, "C3"], ["002", "Rest", 0, "C3"], ["003", "Left", 1, "C4"], ["003", "Right", 2, "C3"], ["003", "Rest", 0, "C3"]];
const out = [];
for (const [s, cond, cls, ch] of cells) {
  const tr = trials(s, cls, CH[ch]), NT = tr.NT, X = tr.x;
  const r = { S: s, cond, N: X.length };
  const Zmu = X.map(x => { const m = avg(Array.from(x)); return wave(Float64Array.from(x, v => v - m), bands.mu); });
  const Zbe = X.map(x => { const m = avg(Array.from(x)); return wave(Float64Array.from(x, v => v - m), bands.beta); });
  const Emu = Zmu.map(z => envOf(z, bands.mu)), Pmu = Zmu.map(z => powOf(z, bands.mu));
  const Ebe = Zbe.map(z => envOf(z, bands.beta)), Pbe = Zbe.map(z => powOf(z, bands.beta));
  const h1 = i => i % 2 === 0;
  // 幅值方向: 留一(对半)包络相关 与 随机成对包络相关(零假设)
  for (const [nm, E] of [["mu", Emu], ["beta", Ebe]]) {
    const other = i => E.filter((_, j) => h1(j) !== h1(i));
    r["env_holdout_" + nm] = +avg(E.map((e, i) => pearson(e, other(i).reduce((a, b) => a.map((v, k) => v + b[k]), other(i)[0].map(() => 0)).map(v => v / other(i).length)))).toFixed(2);
    const pairs = []; for (let i = 0; i < 60; i++) { const a = Math.floor(Math.random() * E.length), b = Math.floor(Math.random() * E.length); if (a !== b) pairs.push(pearson(E[a], E[b])); }
    r["env_pair_" + nm] = +avg(pairs).toFixed(2);
    r["erd_" + nm] = null; // 稍后按被试填
    r["pow_" + nm] = +avg(E.map((_, i) => powOf((nm === "mu" ? Zmu : Zbe)[i], nm === "mu" ? bands.mu : bands.beta)).map(p => win(p, 0.5, 3.5))).toFixed(1);
  }
  // 相位方向: ITC 图劈半(整带×时间) 与 mu×0-0.5s 窗口
  const itcHalf = idx => { const Z = Zmu.filter((_, i) => idx(i)); const n = Z[0][0].length; return bands.mu.map((_, fi) => itcOf(Z.map(z => z[fi]))); };
  const A = itcHalf(h1), B = itcHalf(i => !h1(i));
  r["itc_sh_mu"] = +pearson(A.flat(), B.flat()).toFixed(2);
  const itcMu = itcOf(Zmu.map(z => z)).length ? bands.mu.map((_, fi) => itcOf(Zmu.map(z => z[fi]))) : [];
  r["itc_mu_early"] = +avg(itcMu.map(row => win(row, 0, 0.5))).toFixed(3);
  r["itc_mu_late"] = +avg(itcMu.map(row => win(row, 0.5, 3.5))).toFixed(3);
  // 整体波形方向: 试次 vs 对半均值 的相关(留一), 与随机成对相关(零假设)
  const meanOdd = X.filter((_, i) => h1(i)).reduce((a, b) => Array.from(a, (v, k) => v + b[k]).map(v => v / 1), a => a).length ? Array.from({ length: NT }, (_, k) => avg(X.filter((_, i) => h1(i)).map(x => x[k]))) : null;
  const meanEven = Array.from({ length: NT }, (_, k) => avg(X.filter((_, i) => !h1(i)).map(x => x[k])));
  r["wf_holdout"] = +avg(X.map((x, i) => pearson(Array.from(x), i % 2 === 0 ? meanEven : meanOdd))).toFixed(3);
  const wp = []; for (let i = 0; i < 60; i++) { const a = Math.floor(Math.random() * X.length), b = Math.floor(Math.random() * X.length); if (a !== b) wp.push(pearson(Array.from(X[a]), Array.from(X[b]))); }
  r["wf_pair"] = +avg(wp).toFixed(3);
  // 诱发电位解释方差: var(均值)/平均var(试次)
  const m = Array.from({ length: NT }, (_, k) => avg(X.map(x => x[k])));
  const vm = avg(m.map(v => (v - avg(m)) ** 2));
  const vx = avg(X.map(x => { const mu = avg(Array.from(x)); return avg(Array.from(x, v => (v - mu) ** 2)); }));
  r["evoked_var_frac"] = +(vm / vx).toFixed(3);
  out.push(r);
}
for (const r of out) for (const b of ["mu", "beta"]) {
  const rest = out.find(o => o.S === r.S && o.cond === "Rest");
  if (r["erd_" + b] === null && r.cond !== "Rest") r["erd_" + b] = +((r["pow_" + b] / rest["pow_" + b] - 1) * 100).toFixed(1);
  if (r.cond === "Rest") r["erd_" + b] = "参照";
}
out.forEach(r => console.log(JSON.stringify(r)));
fs.writeFileSync("材料/试次间一致性_三方向对比.json", JSON.stringify(out, null, 1));
