// 试次间一致性的四种度量在同一份数据上的对比
//  R1 相位ITC图 的劈半可重复性: 两半试次各算ITC, 在 mu带×时间 上求 Pearson r (零假设=0)
//  R2 mu带功率时间过程 的劈半可重复性: 两半各算平均功率, 沿时间求 r (零假设=0)
//  R3 单试次 mu包络 与 条件均值包络 的平均相关 (逐试次的幅值可重复性)
//  ERD mu带功率相对静息条件的变化百分比 (0.5-3.5s)
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
const MU = []; for (let f = 8; f <= 13.01; f += 0.5) MU.push(f);
// 每试次 mu 带复小波: 返回 [trial][freq][timeBin] 的 {re,im}
function muZ(x, NT) {
  const nT = Math.floor(NT / STEP);
  return x.map(sig0 => {
    const mu = sig0.reduce((a, b) => a + b, 0) / NT;
    const s = Float64Array.from(sig0, v => v - mu);
    return MU.map(f => {
      const sg = CYC / (2 * Math.PI * f), hw = Math.ceil(3.5 * sg * FS);
      const K = []; for (let d = -hw; d <= hw; d++) { const g = Math.exp(-d * d / (2 * (sg * FS) ** 2)); K.push([g * Math.cos(2 * Math.PI * f * d / FS), g * Math.sin(2 * Math.PI * f * d / FS)]); }
      const row = new Array(nT);
      for (let ti = 0; ti < nT; ti++) {
        const t = ti * STEP; let re = 0, im = 0;
        for (let j = 0; j < K.length; j++) { const tt = t + j - hw; if (tt < 0 || tt >= NT) continue; re += s[tt] * K[j][0]; im += s[tt] * K[j][1]; }
        row[ti] = [re, im];
      }
      return row;
    });
  });
}
const itc = Zs => { const nT = Zs[0][0].length; return MU.map((_, fi) => { const r = new Array(nT); for (let ti = 0; ti < nT; ti++) { let re = 0, im = 0; for (const z of Zs) { const m = Math.hypot(z[fi][ti][0], z[fi][ti][1]) || 1e-12; re += z[fi][ti][0] / m; im += z[fi][ti][1] / m; } r[ti] = Math.hypot(re, im) / Zs.length; } return r; }); };
const pow = Zs => { const nT = Zs[0][0].length; const r = new Array(nT).fill(0); for (const z of Zs) for (let ti = 0; ti < nT; ti++) { let s = 0; for (let fi = 0; fi < MU.length; fi++) s += z[fi][ti][0] ** 2 + z[fi][ti][1] ** 2; r[ti] += s / MU.length; } return r.map(v => v / Zs.length); };
const env = z => z[0].map((_, ti) => { let s = 0; for (let fi = 0; fi < MU.length; fi++) s += Math.hypot(z[fi][ti][0], z[fi][ti][1]); return s / MU.length; });
const pearson = (a, b) => { const n = a.length, ma = a.reduce((x, y) => x + y, 0) / n, mb = b.reduce((x, y) => x + y, 0) / n; let sa = 0, sb = 0, sab = 0; for (let i = 0; i < n; i++) { const da = a[i] - ma, db = b[i] - mb; sa += da * da; sb += db * db; sab += da * db; } return sab / Math.sqrt(sa * sb); };
const flat = m => m.flat();

const CH = { C3: 0, C4: 2 };
const rows = [];
for (const [s, cond, cls, ch] of [["001", "Left", 1, "C4"], ["001", "Right", 2, "C3"], ["001", "Rest", 0, "C3"], ["002", "Left", 1, "C4"], ["002", "Right", 2, "C3"], ["002", "Rest", 0, "C3"], ["003", "Left", 1, "C4"], ["003", "Right", 2, "C3"], ["003", "Rest", 0, "C3"]]) {
  const tr = trials(s, cls, CH[ch]);
  const Z = muZ(tr.x, tr.NT);
  const h1 = Z.filter((_, i) => i % 2 === 0), h2 = Z.filter((_, i) => i % 2 === 1);
  const R1 = pearson(flat(itc(h1)), flat(itc(h2)));
  const R2 = pearson(pow(h1), pow(h2));
  const envs = Z.map(env);
  const nT = envs[0].length;
  const meanEnv = Array.from({ length: nT }, (_, ti) => envs.reduce((a, e) => a + e[ti], 0) / envs.length);
  const R3 = envs.reduce((a, e) => a + pearson(e, meanEnv), 0) / envs.length;
  const late = a => { const s0 = Math.floor(0.5 * 20), s1 = Math.floor(3.5 * 20); let t = 0; for (let i = s0; i < s1; i++) t += a[i]; return t / (s1 - s0); };
  rows.push({ S: s, cond, R1_phaseITC: +R1.toFixed(2), R2_power: +R2.toFixed(2), R3_envVsMean: +R3.toFixed(2), muPow_late: +late(pow(Z)).toFixed(1) });
}
// ERD%: 任务 mu 功率相对同被试静息
const rest = {};
for (const r of rows) if (r.cond === "Rest") rest[r.S] = r.muPow_late;
for (const r of rows) if (r.cond !== "Rest") r.ERD_pct_vs_rest = +((r.muPow_late / rest[r.S] - 1) * 100).toFixed(1);
rows.forEach(r => console.log(JSON.stringify(r)));
fs.writeFileSync("材料/试次间一致性_多度量对比.json", JSON.stringify(rows, null, 1));
