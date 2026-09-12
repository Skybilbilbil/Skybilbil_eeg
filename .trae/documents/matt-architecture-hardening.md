# MAtt 架构严谨化重构方案

## Context

`MAtt_baseline/mAtt/mAtt.py` 是从上游 MAtt(NeurIPS 2022)手工拷贝后深度修改的文件,存在五类"补丁式写法":① 构造外覆盖属性(`net.dropout = nn.Dropout(0.3)` 在三处入口重复覆盖类内硬编码 0.5,即"双默认值坑");② 模块级环境变量开关(`ATTN_SCORE_ORIGINAL`)控制注意力打分分支;③ `mAtt_bci`/`mAtt_mamem`/`mAtt_cha` 三份几乎重复的模型骨架;④ `build_matt_model` 与 train_pair 直接建模型的双入口,前者只支持参数子集;⑤ `forward`/`get_features` 重复主干、手动窗口探测、`ChannelPriorWeighting` 双模式分支。

用户已拍板:**连上游遗留模型类一起抽公共基类;只修补丁冲突点(不做顶部常量 argparse 外露);mAtt_bci 默认 dropout=0.3 并在 `_exp_compare.py` 显式保护 0.5。** 硬约束:数值行为逐位不变(150it/seed42 下 A 臂基线可复现)、注意力独立基无锚点、InstanceNorm、评分函数 `1/(1+log(1+d²))`、`--channel_prior_eps` clamp 逻辑保留。

## 改动步骤(每步独立验证)

### Step 1 — mAtt.py:注意力打分参数化
- 删 L9-12 环境变量块与 L2 `import os`(os 仅被它使用;math 保留)。
- `AttentionManifold.__init__` 增加 `score_original=True`;`_head_prob`(L489)内 `if ATTN_SCORE_ORIGINAL:` → `if self.score_original:`。
- 全库仅 L12/L489 两处引用,当前无脚本设置该环境变量 → 行为不变。
- 验证:py_compile + 建 AttentionManifold 断言走 original 打分。

### Step 2 — mAtt.py:私有基类 + 三子类重构(最大改动,数值风险最高)
- 新增 `_SparseAttentionBase(win_samples, step_samples, n_classes, d_attn_in, d_attn_out, local_radius, anchor_stride, n_heads, n_windows, use_hann, use_channel_weight, n_spatial_filters, channel_gaussian, gaussian_sigma, conv1_kernel, conv1_out, conv2_kernel, conv2_pad, conv2_out, conv2_separable, conv2_multiscale, conv2_ms_per_branch, conv2_ms_specs, manifold_change, attn_change, attn_prior, dropout=0.3, score_original=True)`。
- **构造顺序须与现 mAtt_bci 逐位一致**(q/k/v 三套 SPDTransform 的 `orthogonal_` 消耗 RNG 次序不可变):channel_weight(None 则不建)→ conv1 → In1 → conv2 分支(**先 multiscale → 再 separable → 否则默认**)→ In2 → E2R → SPDNorm → ManifoldChange → AttentionManifold → SPDRectified → SPDTangentSpace → Flatten → dropout(None=不建)→ feat_per_window → Linear(LazyLinear 或固定,按 n_windows)。
- 方法:`_shared_backbone(x)`(通道→conv1→conv2→SPD 链)与 `_features(x)`(`_shared_backbone`→att→ract2→tangent→flatten);`forward` 共享,`get_features` 只在 `mAtt_bci` 定义(`return self._features(x)`),mamem/cha 不新增 public API。
- 子类映射(签名保持现状):
  - `mAtt_bci`(17 参 + 末尾新增 `dropout=0.3`):`use_channel_weight=True, conv1_kernel=22, conv1_out=n_spatial_filters, conv2_kernel=12, conv2_pad=6, conv2_out=d_attn_in`,其余透传;删原 `__init__` 主干与 `self.dropout=nn.Dropout(0.5)`(L649)。保留 `self.n_spatial_filters` 语义(基类以 conv1_out 赋值)。
  - `mAtt_mamem`:签名不变;`use_channel_weight=False, conv1_out=d_attn_in, conv1_kernel=8, conv2_kernel=36, conv2_pad=18, d_attn_in=15, d_attn_out=12, n_classes=5, dropout=None`。
  - `mAtt_cha`:签名不变;`use_channel_weight=False, conv1_out=22, conv1_kernel=56, conv2_kernel=64, conv2_pad=32, d_attn_in=16, d_attn_out=8, n_classes=2, dropout=None`。
- 验证:用补全后的 `build_matt_model` 建网,`load_state_dict('MAtt_baseline/results/tune_a_mixbase/mixbase_cross_mix_best_acc.pt', strict=True)` 必须零缺键;小 batch 前向 shape 正确。不通过即回退直至通过。

### Step 3 — utils/functions.py:build_matt_model 补全
- 加全 mAtt_bci 参数映射:`n_classes/d_attn_in/d_attn_out/local_radius/anchor_stride/n_heads/use_hann/conv2_separable/conv2_multiscale/conv2_ms_*/n_spatial_filters/channel_gaussian/gaussian_sigma/attn_change/attn_prior/manifold_change/dropout`,全部 `args.get(..., 现状默认)`;`n_windows=None` 保持(勿改,LazyLinear 自适应)。
- 删 L360 `net.dropout = nn.Dropout(...)`,改 `mAtt_bci(..., dropout=args.get('dropout', 0.3))` 构造传入。
- 验证:py_compile;build/load_matt_model 严格加载既有 bci2a checkpoint 成功。

### Step 4 — train_pair_with_recording.py:统一构建入口 + dropout 构造传参 + M helper
- L426/L449 两处 `mAtt_bci(...)` 改为 `build_matt_model(WIN_SAMPLES, STEP_SAMPLES, model_args)`,model_args 含当前全部生效参数(`d_attn_in=8, d_attn_out=8, local_radius=LOCAL_RADIUS, n_spatial_filters=N_SPATIAL_FILTERS, channel_gaussian, conv2_multiscale, conv2_ms_*, attn_change, attn_prior, manifold_change, dropout=DROP_PROB, anchor_stride=None, n_heads=1, n_windows=None, wd 不在此`);顶部 import build_matt_model。
- 删 L437/L460 `net.dropout = nn.Dropout(DROP_PROB)`。
- L310-324 手动窗口探测抽成 helper `_infer_windows(net, tr_ld) -> int`(逻辑逐字不变;不从 net.linear 推窗数,因为 LazyLinear 首前向前无权重且 M 是记录语义参数)。
- 验证:全部 py_compile + 完整回归(见下)。

### Step 5 — 根目录 _exp_compare.py:保护历史基线(1 行)
- 用户已拍板默认 0.3:`_exp_compare.build_net`(约 L155)显式传 `dropout=0.5`,保住 `_exp_compare_summary.json` 的 win5/win3 数字。

## 不改的文件
- `run_matt_physionetmi.py`、`run_matt_cross_subject.py`:CONFIG 均含 `dropout:0.3`,经补全后的 build_matt_model 自动读取,行为不变。
- `_reproduce_official.py`/`_ablation_official.py`/`_verify_newdata_attn.py`:在 forward 前覆盖 dropout,不受默认影响,不动。
- 已废弃的 `MAtt_baseline/mAtt_mamem.py`、`mAtt_bcicha.py`(上游遗留原始入口,勿碰)。

## 回归验证(金标准 = tune_a_mixbase,只读)

1. py_compile:5 个改动文件 + 依赖链。
2. state_dict 严格加载既有 checkpoint(最强单点检查,见 Step 2)。
3. WSL 完整回归:
   `wsl -e bash -lc 'cd /mnt/d/EEG_Analysis/Riemannian_Trajectories/MAtt_baseline && source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg_analysis && python train_pair_with_recording.py --out_dir tune_refactor_check --tag refchk --wd 0.02 --seed 42'`
4. 对照 `tune_a_mixbase/pair_training_report.txt` 关键锚点必须一致:
   - cross_mix:test_acc 最终 **54.6%** / 峰值 **55.6%(it110)**;silhouette 0.001、attn_conc 0.203;通道scale 0.5596、Conv1 0.4351、Conv2 0.4711、分类器W 1.2599
   - within_S3:test 最终 81.0% / 峰值 82.8%;train 89.0%;通道scale 2.6486
   - within_S9:test 最终 72.4% / train 88.4%
   - 接受标准:同 seed 下位级完全一致为佳,允许 ≤0.1pp 浮点噪声,趋势与锚点逐位对齐。
5. 验证通过后清理 `tune_refactor_check` 产物;`_smoke_test.py` 冒烟兜底(它不训练,仅链路保底)。

## 风险与回滚
- 高危:基类构造顺序(尤其 q/k/v SPDTransform 的 RNG 消耗次序)、state_dict 键名(`channel_weight.scale` 17 维 / `ract1.window` / `att2.*`)、conv2 分支优先级(multiscale→separable→default)、gaussian 模式 buffer 注册方式——任何偏差都导致复现失败或 strict 加载失败,Step 2/3 落地后先过"严格加载 + 构造对比"检查再继续。
- git 单提交落地;`tune_a_mixbase` 的 checkpoint 与 report 是只读金标准,全程不动。