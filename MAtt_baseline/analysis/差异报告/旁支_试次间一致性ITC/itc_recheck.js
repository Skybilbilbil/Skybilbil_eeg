// 用真数据复算 ITC, 并与 80 试次的理论/经验零假设对照
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

// Morlet 复小波: 7 周期, 时域直接卷积(去均值), 返回每个时频点的 ITC
const FS = 200, CYC = 7;
const freqs = []; for (let f = 4; f <= 36.01; f += 0.5) freqs.push(+f.toFixed(1));
const STEP = 10;                                   // 时间下采样 20 Hz
function itcMap(tr, shift, subERP) {
  const NT = tr.NT, nT = Math.floor(NT / STEP);
  let erp = null;
  if (subERP) { erp = new Float64Array(NT); tr.x.forEach(x => { for (let t = 0; t < NT; t++) erp[t] += x[t] / tr.x.length; }); }
  const g = freqs.map(() => Array.from({ length: nT }, () => ({ re: 0, im: 0 })));
  tr.x.forEach((x, k) => {
    const mu = x.reduce((a, b) => a + b, 0) / NT;
    const s = new Float64Array(NT);
    for (let t = 0; t < NT; t++) {
      const src = subERP ? x[t] - erp[t] : x[t];
      s[t] = src - mu;
    }
    if (shift) { const o = new Float64Array(NT); for (let t = 0; t < NT; t++) o[t] = s[(t + shift[k]) % NT]; s.set(o); }
    freqs.forEach((f, fi) => {
      const sig = CYC / (2 * Math.PI * f), hw = Math.ceil(3.5 * sig * FS);
      const K = []; for (let d = -hw; d <= hw; d++) K.push([Math.exp(-d * d / (2 * (sig * FS) ** 2)) * Math.cos(2 * Math.PI * f * d / FS), Math.exp(-d * d / (2 * (sig * FS) ** 2)) * Math.sin(2 * Math.PI * f * d / FS)]);
      for (let ti = 0; ti < nT; ti++) {
        const t = ti * STEP; let re = 0, im = 0;
        for (let j = 0; j < K.length; j++) { const tt = t + j - hw; if (tt < 0 || tt >= NT) continue; re += s[tt] * K[j][0]; im += s[tt] * K[j][1]; }
        const m = Math.hypot(re, im) || 1e-12;
        g[fi][ti].re += re / m; g[fi][ti].im += im / m;
      }
    });
  });
  const N = tr.x.length;
  return g.map(b => b.map(c => Math.hypot(c.re, c.im) / N));
}
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const band = (m, f0, f1, t0, t1) => {
  const o = [];
  freqs.forEach((f, fi) => { if (f < f0 || f > f1) return; for (let ti = Math.floor(t0 * 20); ti < Math.floor(t1 * 20); ti++) o.push(m[fi][ti]); });
  return mean(o);
};

const CH = { C3: 0, C4: 2 };
console.log("理论地板 0.886/sqrt(N):  N=80 ->", (0.886 / Math.sqrt(80)).toFixed(4), " N=240 ->", (0.886 / Math.sqrt(240)).toFixed(4), " N=480 ->", (0.886 / Math.sqrt(480)).toFixed(4));
for (const [s, cond, cls, ch] of [["001", "Left", 1, "C4"], ["001", "Right", 2, "C3"], ["001", "Rest", 0, "C3"], ["002", "Left", 1, "C4"], ["002", "Right", 2, "C3"], ["002", "Rest", 0, "C3"], ["003", "Left", 1, "C4"], ["003", "Right", 2, "C3"], ["003", "Rest", 0, "C3"]]) {
  const tr = trials(s, cls, CH[ch]);
  const m = itcMap(tr, null);
  const all = m.flat();
  // 经验零假设: 每试次随机整数循环平移(打乱相位参考), 重复 5 次
  const nulls = [];
  for (let r = 0; r < 5; r++) {
    const sh = tr.x.map(() => 50 + Math.floor(Math.random() * (tr.NT - 100)));
    const mn = itcMap(tr, sh).flat();
    nulls.push(mean(mn));
  }
  const me = itcMap(tr, null, true);
  console.log(`S${s} ${cond.padEnd(5)} ${ch}  N=${String(tr.x.length).padStart(3)} | 全图=${mean(all).toFixed(3)} 零假设=${mean(nulls).toFixed(3)}±${Math.sqrt(nulls.map(v => (v - mean(nulls)) ** 2).reduce((a, b) => a + b, 0) / nulls.length).toFixed(3)} | mu×0-0.5s 原=${band(m, 8, 13, 0, 0.5).toFixed(3)} 去ERP=${band(me, 8, 13, 0, 0.5).toFixed(3)} | beta×0-0.5s 原=${band(m, 15, 30, 0, 0.5).toFixed(3)} 去ERP=${band(me, 15, 30, 0, 0.5).toFixed(3)} | mu后段(0.5-3.5s)=${band(m, 8, 13, 0.5, 3.5).toFixed(3)}`);
}
