# AGENTS.md

EEG 运动想象(MI)解码研究代码:SPD 流形上的注意力模型(基线为 NeurIPS 2022 MAtt)+ 自研预处理管线。这是研究代码,不是软件工程项目——无 CI、无 lint/format 配置、无测试框架、根目录无依赖清单。改动请最小化、聚焦研究需求。

## 快速定位(新 agent 从这里开始)

- **唯一训练主入口** `MAtt_baseline/train_pair_with_recording.py`(argparse 开关见 `--help`);**唯一数据** = `saved_data/*.pkl` 2 份;**唯一结果根** = `MAtt_baseline/results/`。

| 要做什么 | 去哪 |
| --- | --- |
| 任何训练对比(9 人混合 / 被试内 / Physionet103) | `MAtt_baseline/train_pair_with_recording.py` |
| 纯被试内 BCI-2a 单被试 | `MAtt_baseline/run_matt_physionetmi.py`(改 `CONFIG['dataset']='bci2a'`;无模型开关) |
| 90 seen / 13 unseen | `MAtt_baseline/run_matt_cross_subject.py` |
| 模型本体 + 前后端边界 | `MAtt_baseline/mAtt/mAtt.py`(`mAtt_bci`、`BACKEND_PREFIXES`) |
| 模型统一构建入口 | `MAtt_baseline/utils/functions.py::build_matt_model` |
| 数据加载(所有脚本共用) | `MAtt_baseline/utils/GetPhysionetMI.py::load_preprocessed_dataset` |
| 逐层 / 跨被试诊断 | `MAtt_baseline/analyze_cross_pipeline.py`、`analyze_pair_subjects.py`、`analyze_train_test_layers.py` |
| 重新生成 pkl(预处理) | 根目录 `main.py`(Physionet103)/ `main_bci2a.py`(BCI9),配置在文件顶部 |
| 参考后端实验复现凭据 | `MAtt_baseline/_run_bci_ref_backend.py` + `results/checkpoint/backend_ref_{S3,S7}.pt` |
| 历史结论数字 / 逐实验曲线 | 本文件「项目实时进度」;曲线 = `results/tune_*/pair_training.json` |

- **别去碰**(已废弃,保留仅为不删代码):`visualize_matt.py`、`profile_10_subjects.py`、`analyze_profile_results.py`、`diagnose_cross_subject.py`、`.trae/`、根目录遗留 LSTM 簇(`Classification.py`/`classificability_analysis.py`/`adaptive_alignment.py`/`Data_saving.py`/`data_reading.py`/`csv_loader.py`/`Dynamic_trajectory.py`/`CPU_test.py`)与 `visualization/`。
- **四条硬约束**:①数据只有 2 份 pkl,`load_preprocessed_dataset` 是唯一加载口;②复现必须带 `--seed 42`;③`saved_data/`、`results/` 被 gitignore,只在本地;④跑脚本先 `conda activate eeg_analysis`。

## 用户规则(所有 agent 必须遵守)

- **先思考后编码**:动手前明确陈述假设;不确定就主动提问;存在多种理解时列出来,不默默选一个;发现更简单方案直接指出、必要时提出反对;遇到困惑停下来说明并提问。

- **简单至上**:只写能解决问题的最少代码;不为一次性使用建抽象,不加"灵活性/可配置性";不处理不可能发生的异常;代码超出必要长度就重写简化。

- **外科手术式修改**:只动必须动的地方;不顺手"改进"相邻代码/注释/格式,不重构未损坏的部分;保持与现有代码风格一致;无关死代码可口头提及但不删除(除非用户要求);因改动产生的未使用导入/变量/函数必须清理。

- **目标驱动执行**:模糊任务先转化为可验证的成功标准(先写复现/无效输入测试,再让它通过);多步骤任务先给简短计划,每步带验证检查点。

- **算法与数学优先**:回复用中文;讲算法先讲底层数学原理(公式推导、复杂度、收敛性、几何解释),代码只是验证数学思想的工具;写代码前先确保用户理解核心数学逻辑,代码必须严格对应数学步骤并在关键处注释说明对应哪个公式。

- **完成后清理**:操作完成后清理无用的临时文件。

## 进度同步规则(所有 agent 必须遵守)

- **每完成一个大操作(跑完一组实验 / 代码重构 / 数据变更 / 得出重要结论),必须立即把结果同步进下方「项目实时进度」章节**。

- 只写**现状与问题**(做了什么、结果数字、结论、产物路径、坑);**不写推荐方向/未验证建议/待试方案,不写过程叙述**。

- 新结论推翻旧结论时,直接改写对应条目并更新"最后更新"日期。

- 本文件是所有 agent 工具共享的**唯一进度事实源**;每次 agent 开始工作前先读它,干完活同步回它。

## 项目实时进度(最后更新:2026-09-12)

### 用户决策/硬约束(勿改)

- 只做 BCI Competition IV-2a(9 人,250Hz);**PhysionetMI 曾暂停,2026-09-08 起恢复**(用户要求接入全量 109 人验证 per-subject conv1+conv2 结论,论文清洗规范为 Shuqfa 2024 Data-in-Brief,管线已对齐:剔 6 异常被试/丢基线 run/删 65 列,详见下方数据章节)。

- 不做严格 LOO 跨被试,只做 9 被试混合(9 人全量试次拼接 → 60/20/20 随机切分,random\_state=42,有被试重叠,数字偏乐观)。

- 注意力**只用独立基**(共享基实测更好,但用户强制删掉共享基分支);无锚点(anchor\_stride=None);评分函数 `1/(1+log(1+d²))` + softmax(dim=-2)。

- 全连接注意力(local\_radius 取 max(win/step,3) 量级);InstanceNorm 替代 BN;dropout=0.3;跨被试训练 wd=0.02。

- 方案/B 线(LSTM、旧 7 通道、自研对齐等)已废弃,数据已删、代码保留未删,勿在其上扩展。

### 核心结果(2026-09-02 profile\_final 快照,150it, lr5e-4, wd0.02, M=4)

| 协议              | 峰值        | 最终    | 出处                                                       |
| --------------- | --------- | ----- | -------------------------------------------------------- |
| 被试内 S3          | **82.8%** | 81.0% | `MAtt_baseline/results/profile_final/pair_training.json` |
| 被试内 S9          | 81.0%     | 79.3% | 同上                                                       |
| 9 混合 cross\_mix | 56.5%     | 54.5% | 同上                                                       |

- 被试内已超原文 74.71%(MAtt NeurIPS 2022);9 混合 gap≈24pp = **当前唯一瓶颈**。

- 窗长是框架内最强旋钮(`_exp_compare_summary.json` exp3):win5(2.5s窗) S3=82.8 / 9mix=56.6;win3(3.5s窗) 9mix=59.8(9mix 最佳)。

- 类序 left=0 / right=1 / feet=2 / tongue=3;**feet 是混合塌陷最重的类**(cross per-class 0.339,28% 误判为 tongue)。

### 核心结果 2:PhysionetMI 全量 103 人 per-subject conv1+conv2(2026-09-09,tune_physionet103_p154,seed42,150it,wd0.02,103 组滤波器)

| 协议 | 峰值 | 最终 | 说明 |
| --- | --- | --- | --- |
| 全量混合 per-subject conv1+conv2 | 35.0% | 34.5% | 仅略高于 25% 随机!9 人上的 82.8% 完全不可复现 |
| 全量混合 身份置换(+1 mod N) | 正确 34.52% | 错配 27.18% | **Δ=7.34pp**,远低于 9 人时的 27.6pp → 几乎无身份记忆,也几乎无学习 |
| 被试内 S03 | 55.6% | 55.6% | 远低于 BCI S3 的 82.8%(每类仅 ~22 试次、测试集只有 18 试次) |

- **结论:per-subject conv1+conv2 的 82.8% 是"9 人小池 + 强身份记忆"的假象,全量放大后崩溃**。103 人上 train 72.6% → test 34.5%(强过拟合),silhouette 全程 <0(-0.02),attn 集中度 0.204≈均匀,conv 权重被压缩(channel scale 2.86→0.18、conv1 20.2→0.46、conv2 8.9→0.53),后端未学到可用分离,final 后端已导出 `results/checkpoint/backend_physio103.pt`(8850B,但此后端基于失效前端,勿直接作正式产物)。

- **眼电参考对照实验(2026-09-09,tune_physionet103_eogref,同配置 re-验证)**:给 Physionet loader 接入 Fp1/Fp2 眼电参考(ICA 排除 2 个眼电成分、variance+5.4%)后重跑同款 per-subject 实验,结果[表2]数值逐位同构:test 峰值 35.0→34.0% / 最终 34.5→33.7%(-0.8pp)、silhouette -0.023→-0.026、attn - 集中度 0.204→0.205、身份置换 Δ 7.34→7.61pp(仍几乎无身份记忆)、conv1/conv2 压缩至 0.46/0.53 与清洗前相同;within S3/S9 波动(±1-2 试次)属 18 试次 test 集采样噪声。**结论:眼电伪迹不是混合瓶颈的原因——34.5% 瓶颈在训练机制本身(与"SPD 内容可分性 56.7%"诊断一致,清洗救不了特征内容)**。

- 每类试次量纲:Physionet 全量每被试每类仅 21-24 试次(总 ~2300/类),而 BCI-2a 每人每类 144 试次(总 1296/类×9 人)——**数据稀疏是被试内/混合都大幅回落的主因**。

- 9 人(BCI)82.8% vs 103 人(Physionet)34.5%:9 人混合 60/20/20 随机切分测试集含同池被试+小池过拟合;103 人交叉被试组合爆炸,风险函数无法在共享后端上泛化。

- **框架内后端复用(freeze)机制已在 103 人上完成冒烟验证**(见代码层章节),但当前后端质量不足以支撑"前端改进",后续需先解决"全量混合学习学不动"的根本问题。

### 注意力机制现状(2026-09-03 诊断)

- 当前独立基注意力**无法区分真实想象窗与静息窗**:自聚合权重 self\_im=0.125 与 self\_rest=0.125 相同,权重完全由物理窗位置决定,从未学会选窗。

- **内容可分性才是瓶颈**:SPD 原始特征"静息 vs 想象"最近质心可分性仅 56.7%,且投影到 Q/K 后不再下降 → 特征塌缩在前,注意力不背锅。

- 独立基 vs 共享基对比:共享基 S3=74.1% / S9=65.5%,独立基 S3=60.3% / S9=51.7% —— 独立基在低可分性下聚合更多噪声;但用户强制用独立基。

- 混合塌缩时 attn\_conc 全程 ≈1/M 均匀,注意力从未学会给想象窗加权。

### 已证伪方向(勿重试)

- **E 臂 通道先验锚定微调(2026-09-04,seed=42,150it,wd0.02,`--channel_prior_eps 0.15`**,产物 `MAtt_baseline/results/tune_e_chprior/`):通道 scale 锚定在 先验±15% 邻域,混合 test 峰值 56.0→55.6 / 最终 54.6→54.6(**零增益**);scale 范数保住(-12%,与被试内同水平)但 conv1/conv2 照塌(0.44/0.47,与 A 臂逐位相同)、silhouette≈0、attn 均匀。**证明瓶颈是"方向"自由度(conv1/conv2 空间/频率投影),不是"长度"自由度(通道权重缩放);先验锚定只防通道关闭,救不了 conv**。

- **4-arm 机制定向实验(2026-09-02,seed=42,150it,GPU)**:A 基线 wd0.02→54.6%;B wd0.005→55.5%(train 60.8→64.0,test 仅 +0.9pp);C B+正交正则 ortho0.02→55.5%(零增益);D B+固定 C3/Cz/C4 通道掩膜→55.2%(略降)。产物 `MAtt_baseline/results/tune_{a_mixbase,b_mixwd,c_mixwdortho,d_mixcomb}/`,四臂曲线一致:it5-30 顶到 54-56% 平台后 test 对训练增长无响应。**框架内正则/掩膜微调不能突破 0.56 混合瓶颈**(train 涨 test 不涨 = 过拟合混合池被试组合)。

- silhouette(切空间类间分离度):混合全程 ≈0(-0.010\~+0.002)vs 被试内 0.062-0.079 → **特征 collapse 是瓶颈的直接量纲**。

- **IAF 个体频段(2026-09-03)**:按被试 μ 峰±2Hz 窄带滤波替代固定 8-30Hz → 跨被试 58.9%→46.7%(-12.2pp,丢失 β 信息)。

- **黎曼对齐 RA(2026-09-03,一次性离线探针,非管线改动)**:在被试协方差上做黎曼均值白化 → 跨被试 58.9%→47.9%(-11.0pp,过度抹平个体差异丢类别信息)。**2026-09-12 核实:数据层对齐自始至终只有 EA 一种,两份正式 pkl 都只含 EA,不存在 RA 版本数据;RA 的实验代码已不在仓库(`alignment.py` 仅剩 `compute_ea_matrix_*`/`apply_ea_*`/`riemannian_distance`,全仓 grep 无 RA 白化实现),该 -11.0pp 仅为当时的对照数字。若要翻案 RA,必须在「未 EA」的原始协方差上重跑——在当前已 EA 的 pkl 上再白化属二次对齐,结论不可用。**

- 更早证伪:多尺度卷积(`profile_ms` 无净收益)、EA 用于混合增强(`official_band_ea` 全表最低 47.6%)。

- **per-subject conv1+conv2(2026-09-08 跑,2026-09-09 用户定论:影响很小、开销大,勿再扩展)**:BCI 9 人 `tune_f_per_subject_both` cross_mix 62.6%(峰值63.2%) vs 共享基 54.6% 仅 +8pp;身份置换 correct 62.58%/wrong 35.00%/Δ=27.6pp → **增益大头是身份记忆,剥离后 35% 反而低于共享基**。仅 conv1 per-subject(`tune_f_per_subject_conv1`)59.9%、Δ=23.4pp。103 人全量 34.5%、身份置换 Δ 仅 7.34pp;9 人 wrong_route 35.0% ≈ 103 人 34.5% 高度一致(身份记忆假说闭环)。代价:前端卷积参数量 ×N 被试(22×22 空间核 103 组等),训/存/加载开销大。**结论:per-subject 类别学习增益≈0,不是混合瓶颈的解**;`save/load/freeze_backend` 机制已冒烟验证但**从未正式跑过训练实验,用户明确不要固定后端,勿默认使用**。

- **参考后端初始化微调(2026-09-09 跑,2026-09-09 结论:治标不治本,勿作主攻)**:用参考被试(S3 最佳 + S7 黎曼中心,Log-Euclidean 距离)预训练后端,再配 per-subject conv1+conv2(9 组)微调后端(`tune_g_ps_backend_s3`/`tune_g_ps_backend_s7`,150it,seed42)。cross_mix 最终/峰值:S7=**64.6%**/65.0%、S3=63.0%/63.4%,vs 无后端影初始化 62.6% → **S7 中心后端 +2.0pp、S3 最佳 +0.4pp**。**关键:cross_mix 前端 conv 照样塌缩(conv1 7.19→0.42、conv2 4.55→0.47、channel scale 2.95→0.52,与无后端逐位一致),后端带判别结构也拦不住被试间梯度冲突压缩 conv**;注意力模块 norm 4.8990 全程不变(=加载预训练权重后未再被更新)。**结论:参考后端初始化增益微小(+0.4~2.0pp)、治标不治本,混合瓶颈在前端 conv 塌缩而非后端初始判别结构;本次未跑身份置换,+2pp 是否真实跨被试泛化未定**。

### 混合瓶颈机制结论

- **conv1(空间滤波)**:不同被试空间分布差异 → 梯度方向冲突 → 权重缩至初值 15-25% → 空间特征塌缩。

- **conv2(时序滤波)**:不同被试 μ 峰频率差异(7.8-11.7Hz)→ 梯度方向冲突 → 权重缩至初值 20-30% → 时序特征塌缩。

- **~~数学本质(2026-09-04 确认)~~ → 已被 2026-09-12 的 ② 证伪**:前半段(梯度方向互斥)由 ① 实测证实;但由它推出的"更新方程退化为 `W←(1−ηλ)W`、`|W_t|=(1−ηλ)ᵗ·|W₀|` 指数压缩"**数值上不成立**——ηλ 量级 7350 步只能给 ×0.929,实测 ×0.176。正确机制见下方「wd 塌缩量级复核」与「② wd 对照实测」:**冲突属实,但塌缩 100% 来自 "Adam 把 wd 放大成满步长 η";且关掉 wd 后精度并不上升,说明塌缩本身不是瓶颈。**

- 共同结局:batch 内梯度相消 → conv 权重被优化器(wd)"删除" → conv 输出特征 norm 0.058→0.0085(×7 衰退)→ 注意力输入塌缩 → 分类器无有效信号(分类器 W 范数仅 +15% vs 被试内 +70-80%)。

- 生理依据(`results/diagnosis/diagnosis_report.json`):被试间子空间主角度 60.4°(各自用不同电极组合)、MI 响应潜伏期差 7.54 窗、频谱个体差异(固定 8-30Hz 对各人非最优)。

- **① 跨被试梯度冲突实测(2026-09-12,BCI-9 cross_mix,seed42,150it,产物 `MAtt_baseline/results/probe_grad/`)**:train_pair 新增 `--grad_probe/``--probe_per_subject`(默认48)`/``--skip_within` 三个开关;每个快照点用**固定探针批**(每被试 48 试次)算 g_s=∇_θ L_s,度量非对角 cos 均值与对齐比 |mean_s g_s|/mean_s|g_s|(K=9 完全正交时 =1/√9=**0.333**,完全同向 =1)。实测(150 迭代全程 16 个快照):
  - **conv1**:cos 均值 **-0.008**(范围 -0.065~+0.073)、对齐比均值 **0.323**(范围 0.223~0.438)。
  - **conv2**:cos 均值 **-0.001**、对齐比均值 **0.353**。
  - **backend**:cos 均值 **+0.098**、对齐比均值 **0.493**。
  - 结论:conv1/conv2 的跨被试梯度**全程与"完全正交(纯冲突)"不可区分**,后端却**显著高于**正交基线(存在共同方向)。即**梯度冲突集中在 conv 前端,后端不是冲突源**;且冲突在迭代 0(权重尚未塌缩时)已存在 → 是初始条件而非训练产物。
  - 该 run test 峰值 **56.4%** / 最终 **54.5%**,与 `tune_a_mixbase`(54.6%) 一致 → 探针未扰动训练。

- **wd 塌缩量级复核(2026-09-12,推翻本文档早先的 `|W_t|=(1−ηλ)^t|W₀|`)**:η=5e-4、λ=0.02、~7350 步下乘性衰减只给 ×0.929,而实测 conv1 **2.475→0.436(×0.176)**、channel scale 2.932→0.565(×0.193)、conv2 1.542→0.477(×0.310)。正确机制是 **Adam 归一化**:wd 以 λp 加进梯度后,梯度冲突把数据项抵消掉,Adam 分母 √v̂ 随之变小,wd 项被归一化成满步长 lr 的 −sign(p) 方向步 → 每步绝对收缩 ≈lr,**总降幅 ≈ lr×步数 = 3.68**,与实测降幅 2.04 同量级。**塌缩主因 = "冲突让数据梯度失效" + "Adam 把 wd 放大成满步长",不是乘性衰减。**

- **身份主导假说在共享 conv 配置下未被证实(2026-09-12,部分推翻 2026-09-09 的推断)**:同一 run 把切空间特征换被试标签重算 silhouette:class-sil 均值 **-0.0028**、subject-sil 均值 **-0.0421**。**两者必须各自与随机基线比**(class K=4 基线 −1/3、subject K=9 基线 −1/8,直接互比是错的):class 超出基线 **+0.331**、subject 仅超出 **+0.083** → 该配置下特征对**类别**的编码强于对**身份**的编码。原"身份记忆"证据(Δ=27.6pp)出自 **per-subject conv** 配置,不能外推到共享 conv。

- **② wd 对照实测(2026-09-12,`--wd 0.0` 其余同 ①,产物 `MAtt_baseline/results/probe_wd0/`)**:对塌缩机制的判决性实验。
  - **关掉 wd 后 conv 完全不塌**:conv1 **2.723→2.785(×1.02)**、conv2 1.626→2.022(×1.24)、channel scale 3.025→3.003(×0.99)、In1 ×1.00、In2 ×1.01;对照 wd=0.02 分别为 ×0.176 / ×0.309 / ×0.193 / ×0.112 / ×0.079。
  - **⇒ 塌缩 100% 由 "Adam + wd" 造成,与"共享最优解本身小"无关**。这**证伪了"共享驻点=各被试最优解的精度加权平均、|W*|≈|W_s|/√N"这条几何解释**:若共享解真小,wd=0 下范数仍应显著低于初值;实测 ×1.02 说明**共享解并不小**。两层机制中第一层(平均冲突)被排除,第二层(Adam 归一化放大 wd)是全部原因。
  - **但关掉 wd 并不解决瓶颈**:test 峰值 56.4→**56.9%**(+0.5pp)、最终 54.5→**55.6%**(+1.1pp)、best_val 56.6→**59.7%**;conv1 对齐比 0.323→0.408(仍逼近正交基线 0.333)、class-sil −0.0028→−0.0001、subject-sil −0.0421→+0.0085,**均无质变**。⇒ **"conv 范数不塌" ≠ "学到跨被试特征":冲突依旧,数据梯度本身没有可用的共同方向。**
  - 副产物:**wd=0 时分类器 W 由 1.202 涨到 4.488(×3.73)**,而 wd=0.02 时只 ×1.09 → wd 一边抑制分类器过拟合、一边误伤 conv,而误伤的代价更大(去掉它精度不降反微升)。
  - **对后续步骤的影响**:④(P·V 方向约束)原本"防塌缩"的动机已被 ② 排除,其价值只剩"收缩 argmin 可行集、可能提升梯度对齐";⑤(PCGrad 提升对齐)优先级相对上升。

- **⛔ ⑦【整条作废,以文末「最终对照」为准】留一被试探针(2026-09-12,新脚本 `MAtt_baseline/analyze_holdout_probe.py`,独立于 train_pair、不改其任何开关;超参与 cross_mix 一致)**:只做 1 折——训练 S01–S08(2764 训练 / 922 验证),测试**完全未见过**的 S09 全部 576 试次。这是本仓库**第一次**在"训练/测试被试零重叠"下测共享 conv 模型。
  - **IN(历史默认)= 55.4%**;**BN(不重算)= 47.6%**;**BN + 用 S09 无标签试次重算统计量 = 57.6%**(Δ=**+10.1pp**,净 +58 试次)。
  - **⇒ 结论一:BN 的统计量失配在"真有域差"时是 -7.8pp 的真实损失**(47.6 vs 55.4)。这正是同池协议测不出来的东西:**`--bn_recalib` 在同池协议下 -0.9pp,在留出被试下 +10.1pp,符号相反**。
  - **⇒ 结论二:零标签适配(BN trick)在这份数据上有效**,把失配补回来并略超 IN。
  - **⇒ 结论三(修正旧叙事)**:**共享 conv 模型的同池协议 54.6% 与真跨被试 55.4% 几乎相同** —— "被试重叠导致数字虚高"的担忧在本模型上不成立(S09 一折)。
  - **统计口径**:576 试次下单点 SE≈2.07pp。BN→BN+重算是**同权重配对比较**,+10.1pp 稳;而 **IN 55.4% vs BN+重算 57.6% 仅差 +2.2pp(≈0.8 SE),不显著** —— **不能宣称"适配后超过 IN"**。
  - **⚠ 2026-09-12 补跑三折后:上述结论未能复现,⑦ 的结论需视为待推翻。** 用同一脚本跑**接收性最低的三个被试**(S05/S02/S06,选择依据是 9×9 迁移矩阵的列均值),结果:
    | 留出 | IN-不重算 | BN-不重算 | BN-重算 | Δ(BN重算−不重算) |
    | --- | --- | --- | --- | --- |
    | S05 | 33.5% | 32.8% | 29.9% | **−3.0pp** |
    | S02 | 28.6% | 28.8% | 28.0% | **−0.9pp** |
    | S06 | 34.0% | 34.0% | 26.9% | **−7.1pp** |
    | 均值 | **32.1%** | 31.9% | 28.2% | **−3.6pp** |
    - **绝对准确率从 S09 的 55.4% 塌到 ~32%**(≈固定基严格跨被试 30.5% 的水平,即深度模型在这些折上几乎没学到可迁移的东西);**BN 重算在三个折上全为负**,与 S09 的 +10.1pp 符号相反。
    - **同时暴露一个混杂**:跑 S09 用的**第一版脚本没有设 torch 随机种子**,而这三折的版本加了 `torch.manual_seed(42)`。**S09 的 55.4% 究竟是"被试特殊"还是"未设种子的幸运初始化",必须先排除** —— 已用带 seed 的当前脚本重跑 S09 作对照(产物 `holdout_probe_s9seed.log`)。
    - **✅ 2026-09-12 最终对照(S09 用带 seed 的当前脚本重跑,产物 `holdout_probe_s9seed.log`)—— 结论已定性:**
      - **S09 的"高"是真实被试效应,不是种子伪影**:带 seed 重跑 IN = **60.1%**(无 seed 时 55.4%),两版都远高于其他三折的 28.6~34.0%。
      - **但 "+10.1pp" 是伪影**:带 seed 后 S09 的 BN+重算 = 52.3%,Δ = **−6.7pp**。
      - **四折汇总**:

        | 留出 | IN | BN | BN+重算 | Δ(重算−不重算) |
        | --- | --- | --- | --- | --- |
        | S09 | 60.1% | 59.0% | 52.3% | −6.7pp |
        | S05 | 33.5% | 32.8% | 29.9% | −2.9pp |
        | S02 | 28.6% | 28.8% | 28.0% | −0.8pp |
        | S06 | 34.0% | 34.0% | 26.9% | −7.1pp |
        | **均值** | **39.0%** | 38.7% | 34.3% | **−4.4pp** |

      - **⇒ 定论一:零标签 BN 统计量重算无效,四折全为负**(−0.8 ~ −7.1pp)。同池协议的 −0.9pp 才是它真实的样子,holdout 下并不更好。**"零标签适配有效"整条推翻。**
      - **⇒ 定论二:BN ≈ IN(均值 −0.4pp)**;"BN 统计量失配 −7.8pp"同属无 seed 伪影。**⑥ 里"BN 比 IN +2.6pp(同池)"也仅单 seed,按下面的种子不确定度不可信,需重跑。**
      - **⇒ 定论三:真跨被试 4 折均值 39.0%,跨度 28.6~60.1%(31.5pp),std 12.3pp。同池协议的 54.6% 确实虚高** —— 我基于单折(且无 seed)做出的"同池没有虚高"的修正**是错的**,项目原有的"数字偏乐观"判断成立。
      - **⇒ 定论四:性能由"目标被试是谁"主导**(31.5pp 跨度),远超任何方法差异(<5pp)。与检索到的 2026 文献一致(Subject-Level Heterogeneity / Average Rankings Mask Per-Subject Optimality)。
      - **⇒ 定论五:单 seed 不可信。** 同是 S09,无 seed 与有 seed 两个 run 之间 BN 从 47.6% 变到 59.0%(**11.4pp**)。**本仓库所有单 seed 的跨被试结论都应加此不确定度重审。**

- **⚠ 方法论教训(2026-09-12,写给后续 agent)**:⑦ 连续反转两次——第一次"零标签适配有效(+10.1pp)"来自**单折 + 未设种子**;随后"同池没有虚高"来自**同一个单折**。两次都错。教训:**任何跨被试结论至少要 (a) 多折 且 (b) 多 seed;探针脚本早期版本漏设 `torch.manual_seed` 是直接诱因。**

- **⑥ 归一化层对照与零标签适配(2026-09-12,`train_pair` 新增 `--norm {in,bn}` 与 `--bn_recalib`,mAtt.py 新增 `_norm_layer()`+ `norm_type` 参数;默认仍是 InstanceNorm,历史结果不受影响。产物 `results/probe_bn/`、`results/probe_bnrecal/`)**:动机是 InstanceNorm 逐样本归一化、**没有 running stats**,所以"在目标数据上重算归一化统计量"这个零标签适配旋钮在原模型里根本不存在。
  - **BN vs IN(同 seed42 / wd0.02 / skip_within)**:test 峰值 56.4→**57.6%**、最终 54.5→**57.1%(+2.6pp)**、best_val 56.6→57.3%;但 **train acc 末位相同(60.4 vs 60.5)** → 增益全在泛化侧。**机制完全未动**:conv1 对齐比 0.323→0.323、conv2/backend 0.353/0.493→0.344/0.487、conv1 范数塌缩同为 →0.437、class-sil ≈0。⇒ **BN 改的是"表示质量",不是"跨被试坐标"**。仅 1 个 seed(历史 run 间 spread 0.5pp,2.6pp 超出噪声 5 倍,但未多 seed 确认)。
  - **逐被试 BN 统计量重算(零标签适配,`--bn_recalib`:对每个被试重置 BN running stats、用其无标签试次前向一遍再评估,权重不动)**:**反而略差**——峰值 57.4→56.5%(-0.9pp)、最终 55.4→54.6%(-0.9pp)、全程均值 54.7→54.4%,30 个评估点仅 10 个改善;it0~30 有帮助(+3.7~+1.1pp),it35 之后转为负。
  - **⇒ 关键结论:当前 60/20/20 协议(测试集含同池被试)下,归一化统计量的"域差"本来就是 0**(BN 训练期 running stats 已覆盖全部 9 个被试),重算不仅无可修,还丢掉了跨被试共享统计量 → 变差。**即:任何"目标域适配"方法在当前协议下原理上无法被检验**(需留出被试)。BN 那 +2.6pp 很可能同样依赖该协议(换成未见被试时 running stats 失配,增益未必保住)。

- **⑤ 数据层 SPD 表示可分性诊断(2026-09-12,新脚本 `MAtt_baseline/analyze_spd_separability.py`,纯 CPU 数分钟)**:不看模型、不训练,只用"22 通道 cue 后 [0.5,3.5]s 逐试次协方差 → 切空间(253 维)"这个最朴素表示,回答"特征本身还有没有可学的东西"。
  - **① 被试内 4 类可分性(5 折 CV,LDA+Ledoit-Wolf 收缩)**:各被试 84.0/57.6/88.9/64.4/46.7/55.4/84.7/87.7/83.2%,**平均 72.5%**(随机 25%)。
  - **② 跨被试 9×9 迁移矩阵(i 训→j 测)**:非对角均值 **30.5%**,仅略高于 25% 随机;对角线(in-sample,乐观)91.7%。
  - **⇒ 结论一:表示形式不是瓶颈。** 固定 22 通道空间基、完全不学空间滤波就已到 **72.5%**,而模型被试内 82.8% —— **conv1 只贡献约 +10pp**。旧叙事"SPD 内容可分性 56.7% 是瓶颈"不能支撑"特征没内容"的结论(那是"静息 vs 想象"二分类的另一口径,与 4 类 MI 不可比)。
  - **⇒ 结论二:瓶颈 100% 是"坐标系不同"。** 同一份表示,被试内 72.5% → 跨被试 30.5%,**丢掉 42pp**。这是纯 domain shift,与优化、与 conv 塌缩都无关。
  - **⇒ 对方向的含义**:对齐/坐标变换类方法(alignment、可学习域变换、subject-aware 归一化)的价值被抬高;改表示形式(换 SPD/切空间/多尺度)的价值被降低。

- **④ 前提性检验:各被试"最优空间滤波器"是否正交(2026-09-12,新脚本 `MAtt_baseline/analyze_csp_subspace.py`,纯 CPU 数秒)**:用两条独立口径测这个前提,并给出**基线**(没有基线的角度数字无法解释——旧诊断的"主角度 60.4°"即属此类,且其口径不可复现)。
  - **口径 A:数据驱动 one-vs-rest CSP**(k=8 维子空间,泛化特征问题两端各 1 个/类)。度量 tr(P_iP_j)/k = 平均 cos²(主角)。跨被试 36 对 **0.399 ± 0.052**(平均主角度 **53.1°**);随机 k=8/d=22 子空间基线 **0.364 ± 0.041**(平均主角度 55.4°);**被试内 test-retest 上限(同一被试劈半各算一次 CSP)0.552 ± 0.057**。
  - **⇒ 可共享成分 =(0.399−0.364)/(0.552−0.364)= 18.6%,被试特异成分约 81%**。跨被试重叠显著高于随机(差 0.035,SEM 0.0087,≈4σ)但**远低于被试内上限**。
  - **口径 B:被试内训练得到的 conv1 权重**(S3 vs S9,各 22 个 22 维通道向量;满秩故子空间重叠恒为 1,只能逐滤波器比最佳匹配 |cos|):实测 **0.450**,随机滤波器组基线 **0.455 ± 0.018** → **与随机完全无差别,零对齐**。
  - **结论:①②③ 依赖的"各被试最优方向正交"前提成立,且在两个独立口径上都成立**;精确表述是"可共享成分仅约 19%、其余 81% 被试特异",而非"完全正交"。
  - **对 ④ 的含义**:CSP 是数据驱动估计(自带估计噪声,test-retest 仅 0.552),而 ④ 用的是**解剖先验基**(不是估计出来的),理论上仍可能捕获 CSP 估不准的那部分共享结构——这是 ④ 唯一还站得住的立论;但**上限很低(≈19%),不能指望 ④ 把 54.6% 拉回被试内 80% 量级**。

- **③ N-标度律实测(2026-09-12,PhysionetMI,`--wd 0.0 --max_subjects N`,N=3/9/27/103,产物 `results/probe_n{3,9,27,103}/`;新增 `--max_subjects` 开关,并修掉它与 `PAIR_SUBJECTS=[3,9]` 的越界冲突)**:在 wd=0(排除 wd 干扰)下测对齐比随 N 的曲线。
  - **conv1 对齐比 0.514 / 0.393 / 0.218 / 0.108(N=3/9/27/103)**,独立随机方向的理论值 $1/\sqrt N$ = 0.577 / 0.333 / 0.192 / 0.099 —— **几乎逐点吻合**(拟合指数 α≈0.44,理论 0.5);conv2 同形(0.491/0.385/0.208/0.108)。非对角 cos 在四个 N 上均为 ≈0(−0.110/+0.039/−0.002/+0.003)。
  - **⇒ 跨被试梯度是"独立随机方向",既不是结构化冲突,也不是可对齐方向**。这解释了"全量混合学不动"的数学根因:**每加一个被试,共享梯度幅值按 $1/\sqrt N$ 衰减**,N=103 时只剩单被试梯度的 1/10。backend 明显偏高(0.860/0.618/0.301/0.128),再次印证 ① 的"后端有共同方向"。
  - wd=0 下 conv 范数比在所有 N 上均为 1.005~1.034 → **独立确认 ② 的"不塌缩"结论与 N 无关**。
  - 精度:test 峰值 44.4/42.6/39.1/40.3%(N=3/9/27/103),class-sil 全程 ≈0,subject-sil 仅 N=3 为正(+0.055)。
  - **数学推论(已用 numpy 仿真验证,200 次重复)**:①把 conv1 的可行集收缩到 k 维子空间**不会改变 $1/\sqrt N$ 律**——d=1000/64/8/2 下 align 分别 0.578/0.575/0.565/0.528(N=3)、0.099/0.099/0.096/0.090(N=103),与 $1/\sqrt N$ 基本无关。**唯一例外是子空间与被试最优方向相关**(即生理上有共同结构),那时 align 才不衰减。②**PCGrad 对正交梯度几乎不起作用**:两两随机方向下投影后向量保留比例 **0.9995**(PCGrad 只在负内积时触发,而实测 cos≈0 恰好正交)→ ⑤ 按原设计预期是 no-op。

### 代码层状态与坑

- `_smoke_test.py` 通过;BCI-2a 单被试被试内训练跑通(`run_matt_physionetmi.py` 改 `CONFIG['dataset']='bci2a'`,sub=1,120it → test\_acc 71.55%)。

- BCI2a 实测 T=1126、E2R M=4 窗(win1.5s/step0.75s);训练默认无 seed(dataloader shuffle 未固定;仅 `train_pair_with_recording.py` 支持 `--seed`,复现实验务必带上)。

- 模型结构锚点(2026-09-04 确认):conv1=`Conv2d(1,22,(22,1))` 全通道空间混合,conv2=`Conv2d(22,20,(1,12),pad6)` 12 点时序带通(默认非 separable/multiscale);`n_spatial_filters` 默认 22(trai\_pair 有 `--n_spatial_filters`)。

- **注意力模块范数恒为 4.8990 不是 bug(2026-09-12 核实)**:`SPDTransform.weight` 是 `StiefelParameter`,`MixOptimizer` 每步先做 `orthogonal_projection`(投到切空间)再 `retraction`(拉回 Stiefel 流形),**范数按构造守恒**。所以 ①/② 与历史所有实验里"注意力模块 4.8990 → 4.8990"只说明**范数**不变,**不代表 attention 没在学**(它只在流形上旋转)。推论:`param_snapshot` 的范数**无法**用来判断 attention 是否学到东西,要判断只能看 `att2.attention_prob` 的输出(现有证据:self_im=self_rest=0.125,权重由物理窗位置决定)。

- **dropout 双默认值坑**:`mAtt_bci` 类内 `self.dropout=nn.Dropout(0.5)` 是硬编码默认;训练入口会覆盖为 0.3(train_pair `DROP_PROB=0.3`、`build_matt_model` 默认 `dropout=0.3`)。裸建 `mAtt_bci(...)` 不经训练入口会拿到 0.5,改 dropout 必须走入口。

- **架构严谨化重构(2026-09-06,git 已提交 943dd3c)**:mAtt.py 抽 `_SparseAttentionBase` 私有基类,`mAtt_bci` 配置化派生(构造顺序保持旧语义、state_dict 严格兼容);dropout 改为构造参数(默认 0.3,None=不建)取代"类内硬编码 0.5 + 入口覆盖";`ATTN_SCORE_ORIGINAL` 环境变量 → `AttentionManifold(score_original=True)` 参数;build_matt_model 补全全参数,三个训练入口统一经它构建;train_pair 窗口探测抽 `_infer_windows` helper。回归:150it/seed42 两轮独立验证,与 tune_a_mixbase 的 silhouette/参数范数/attn 逐位一致,cross_mix test_acc 三独立 run 落点 54.6/54.8/55.1%(spread 0.5pp = 1-2 测试样本,GPU 浮点 run 间噪声,验证目录已清理);`dropout=0.5` 历史仅 `_exp_compare.py` 依赖,已显式 `dropout=0.5` 保护。新坑:**analyze_\*/profile_\* 诊断脚本裸建 mAtt_bci 会随默认翻到 0.3(非历史基线来源,如需 0.5 请显式传)**。后续:数值级对照(重构前 `943dd3c^` vs 当前,同 seed)确认非 linear 主干参数与 get_features 逐位 0,仅 LazyLinear 初始化有 ~5e-2 RNG 起点差(非算法);并删除死类 `mAtt_mamem`/`mAtt_cha`(git 未提交,已 rm 文件)。

- Windows 侧曾出现 25 个假性 M 文件 = CRLF 噪声(`git diff -w` 内容等价),无损还原,勿重复"修复"。

- **后端统一模型机制(2026-09-08 新增,train_pair 两个开关;2026-09-09 补 `--freeze_backend`)**:`--save_backend <path>` 在最优 val_acc 处把后端权重(=E2R 之后 att2/ract2/tangent/flat/dropout/linear/spdnorm/manifold_change 全部张量,含 tangent 索引掩码/spdnorm running stats 等 Buffer)独立落盘;`--load_backend <path>` 只加载后端权重、前端(conv1/In1/conv2/In2/channel_weight)保持随机,用于"换 conv 前端、复用后端";`--freeze_backend`(须配 `--load_backend`)加载后把全部后端参数 requires_grad=False——优化器用 `filter(requires_grad)` 构建,冻结参数不进优化器,训练只更新前端,实现"改进 conv 前端时后端完全不动、无需重训"。后端输入是 SPD 协方差,维度仅依赖 d_attn_in/d_attn_out/M(窗口数),不依赖前端卷积实现——换前端只要保持 c2_out=d_attn_in、M 一致即可复用。已验证:save 产物 12 键(9 人 per-subject 冒烟),load 后 q_trans 数值与 ckpt 逐位一致、missing=0/unexpected=0、forward 正常(GPU);freeze 后后端 7 个可学习参数全 False、前端 9 个保持可训练、forward 正常。首次 save 冒烟输出 `/tmp/backend_unified.pt`(临时,勿作正式产物)。**2026-09-09 架构内聚**:边界常量 `BACKEND_PREFIXES`(含前后端划分约束注释)从 train_pair 移入 mAtt.py 模块级,`train_pair` 改 `from mAtt.mAtt import BACKEND_PREFIXES` 统一引用;functions.py 重复的 `BCI2A_CHANNELS` 已删,统一从 mAtt.mAtt 导入。回归:d_in=8 配置加载 `backend_physio103.pt` missing=0/unexpected=0、冻结 7 参数,与重构前一致;后端前缀集合逐字未变,已有 backend_*.pt 兼容。

- **训练提速改造(2026-09-09,只改性能不改数学结果,数值已 PASS)**:
  ① per-subject 卷积向量化(`mAtt.py::_PerSubjectConv2d`/`_PerSubjectConv2dTemporal` forward):103 组逐被试 `F.conv2d` 循环 → gather 权重 + 单次 einsum(空间核 22→1 次 kernel;时序核 12 次固定 einsum),与 F.conv2d 老实现输出逐位一致(spatial 7e-7 / temporal 1.2e-6,输出长 T+1 不变)。
  ② `train_pair` 新增 4 个 argparse:`--iterations`(默认 150)、`--batch_size`(默认 64)、`--snap_every`(默认 10,快照减频;历史 SNAP=5 已由默认值改 10)、`--sil_max`(默认 2000,silhouette 子采样上限,O(N²·d) 复杂度控制)。**注意:快速档(bs128 等)改变训练动力学,结果与历史 bs64 严格不可比,复现已证伪方向勿用;耗时权衡实验才用**。
  ③ 幅度级:103 人 cross_mix 150it 历史 44min(旧 SNAP5),~分钟后级预算依赖上两项 + bs128≈减半。

### 运行速查

- profile 生成器(13 个 argparse 开关全有效):`wsl -e bash -lc 'cd /mnt/d/EEG_Analysis/Riemannian_Trajectories/MAtt_baseline && source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg_analysis && python train_pair_with_recording.py --out_dir <NAME> --tag <TAG> --seed 42 [--wd 0.005 | --ortho_lambda 0.02 | --channel_gaussian | ...]'`

- `run_matt_physionetmi.py`(CONFIG 字典)跑 BCI-2a:改 `CONFIG['dataset']='bci2a'` 即可(auto 选 pkl + sfreq250);其 CONFIG **无模型开关**(channel\_gaussian/ortho\_lambda/manifold\_change/attn\_\* 经 build\_matt\_model 硬接线默认关),要翻这些开关只能走 train\_pair。

## 代码结构

- **根目录** = 自研预处理管线 + 官方 MAtt 复现脚本:

  - 预处理入口 `main.py`(PhysionetMI,109 剔 6 → 103 人)与 `main_bci2a.py`(BCI IV-2a,9 人):MOABB 加载 → 22 通道 → Picard ICA 去噪 → 8-30Hz → \[-1,3.5]s epoch → 逐被试独立 EA → 存 `saved_data/*.pkl`。无 CLI 参数,配置全在文件顶部。

  - 复现脚本(有 CLI,只打 stdout 不落盘):`_reproduce_official.py --mode within|cross|both`(官方数据跨 session,对照原文 74.71%)、`_reproduce_seeds.py`、`_exp_compare.py --exp 1,2,3`、`_ablation_official.py`、`_smoke_test.py`。

- **`MAtt_baseline/`** = 上游 MAtt(NeurIPS 2022)手工目录拷贝(非 git clone),已深度修改(中文注释、InstanceNorm 替 BN、多尺度卷积),是当前研究主战场。

  - 训练入口:`train_pair_with_recording.py`(有 argparse)、`run_matt_physionetmi.py`(CONFIG 字典)、`run_matt_cross_subject.py`(90 seen/13 unseen)。

- **已废弃(未删除,勿在其上扩展)**:`visualize_matt.py`(需训练 npz)、`profile_10_subjects.py`/`analyze_profile_results.py`(早期 profile)、`diagnose_cross_subject.py`;遗留 LSTM 簇(Classification.py、classificability\_analysis.py、Dynamic\_trajectory.py、Data\_saving.py、csv\_loader.py、data\_reading.py、CPU\_test.py、adaptive\_alignment.py 等)及 `visualization/` 下失效脚本。**已删除(2026-09-06)**:上游原版入口 `mAtt_bci.py`/`mAtt_bcicha.py`/`mAtt_mamem.py`(期望 `data/*.mat` 从未跑过)、`utils/GetMamem.py`/`utils/GetBCIcha.py`、mAtt.py 内死类 `mAtt_mamem`/`mAtt_cha`、functions.py 的 `testNetwork_auc` 及其 `roc_auc_score` alias。**已删除(2026-09-12 全面清理)**:一次性查询/监控探针 `_check_results.py`/`_check_identity.py`/`_check_subjects.py`/`_read_results.py`/`_watch_final.py`/`_verify_newdata_attn.py`/`_download_physionet_mp.py`(后者数据已落盘、重跑可重建);`MAtt_baseline/results/` 下过时日志与冒烟产物(tune_g_s3/s7.log、\_runb\_bci2a\_sub1.log、bci2a\_sub001\_best\_\*.json、figures/\*.npz、checkpoint/bci2a\_repeat1\_sub1\_\*.pt)及 `MAtt_baseline/tune_{a,b,c,d}_mix*_run.log` 4 份 4-arm 运行日志(结论已在 tune\_\* 目录 pair_training.json + AGENTS 进度,日志不可再生但无信息量故删);全部 `__pycache__` 编译缓存已删。保留:官方复现脚本 `_reproduce_official.py`/`_reproduce_seeds.py`/`_ablation_official.py`/`_exp_compare.py`/`_smoke_test.py` 与对照数据 `_exp_compare_summary.json`(官方 pkl 已删,官方分支失效但代码保留勿扩展);`_run_bci_ref_backend.py` 与 `results/checkpoint/backend_ref_{S3,S7}.pt` 作参考后端实验复现凭据。

- `.dsh/` 是 agent 工具的任务板/回滚状态,忽略即可。

## 环境(WSL 为唯一运行时)

- 仓库 `D:\EEG_Analysis\Riemannian_Trajectories` 与 `/mnt/d/EEG_Analysis/Riemannian_Trajectories` 是**同一物理目录**(drvfs 挂载),git HEAD 两侧一致;所有研究脚本在 WSL 里跑。

- WSL2 Ubuntu-24.04.4 LTS(用户 lenovo,系统上限 16核/10GB);GPU:RTX 4060 Laptop 8GB,`torch.cuda.is_available()`=True。训练 `cuda if available` 不强制 GPU;预处理纯 CPU,`main*.py` 顶部设 OMP/OPENBLAS/MKL 线程为 8。

- **研究环境 =** **`/home/lenovo/miniconda3/envs/eeg_analysis`(Python 3.10.14)**:跑研究脚本前需手动 `conda activate eeg_analysis`(交互 shell 默认激活 base)。关键包:torch 2.6.0+cu124、mne 1.11.0、moabb 1.5.0、numpy 2.2.6、scikit-learn 1.7.2、pyriemann 0.10。condarc 走清华镜像。

- 不要用 `MAtt_baseline/mAtt_env.yml` 建环境(上游 macOS 遗留,py3.6.9+torch 1.10.2 CPU)。

- 校验命令模板:`wsl -e bash -lc 'cd /mnt/d/EEG_Analysis/Riemannian_Trajectories && source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg_analysis && <cmd>'`。

- 注意:`~/` 下有 3 个以 Windows 路径命名的空占位目录(含 `D:\EEG_Analysis\...`),是 setup 副作用,勿 cd 进去。

- **GPU 与 DSH 沙箱(2026-09-12,必读)**:agent 的 bash 默认跑在 DSH `workspace-write` 沙箱内,该沙箱**禁止以写方式打开工作区外的任何路径**;而 CUDA/NVML 必须以 **O_RDWR 打开 `/dev/dxg`** → 于是 `torch.cuda.is_available()=False`、NVML 报 `GPU access blocked by the operating system`、torch 报 `Found no NVIDIA driver`——**这三条报错全部是误导,GPU 与 Windows 驱动完全正常**(Windows 侧 `nvidia-smi.exe -L` 正常列出 RTX 4060)。判别方法:`python -c "import os; os.open('/dev/dxg', os.O_RDONLY)"` 成功而 `os.O_RDWR` 返回 EACCES 即为沙箱所致。**解决:该条命令加 `sandbox_permissions: "danger-full-access"` 即可,GPU 立即可用**(本会话已验证)。不要据此判断需要重装驱动或 `wsl --shutdown`。

## 数据(最易踩坑)

- `saved_data/`、`results/`、`MAtt_baseline/results/` 均被 gitignore,大文件只在本地。

- 现存 pkl:`preprocessed_data_bci2a_raw_ica.pkl`(\~490MB,已验收:9 人×4 类×144 试次=每类 1296,T=1126/22ch/250Hz、EA 齐全,格式与 physionet 共用 `load_preprocessed_dataset`)。**`official_bciciv2a_matt_raw.pkl`(官方 mat 组装)已删(2026-09-09,用户决定:自建 BCI 已超原文 82.8%>74.71%,官方对照不再需要;依赖它的官方复现脚本 `_reproduce_official.py`/`_reproduce_seeds.py`/`_ablation_official.py`/`_exp_compare.py`/`_smoke_test.py` 官方分支随之失效,代码保留未删勿在其上扩展)**。`preprocessed_data_moabb_raw_ica.pkl` **已完成(2026-09-09,103/103 人,None 阈值;第二次重跑 2026-09-09 接入 Fp1/Fp2 眼电参考)**:历经 physionet.org 网络不稳(SSL EOF/超时)多轮重试收敛 + 断点续跑合并;最终发现部分被试(如 S048/S109)原始幅度大被 `reject_amp_uv=250µV` 整类剔光(属记录噪声特性非文件损坏),遂全量改用 **None(不剔除)** 从零重跑;**2026-09-09 对照 BCI 对齐补充**:BCI loader 保留真实 EOG1/2/3 供 `ica.find_bads_eog` 做眼电参考检测(对齐官方 EOG 回归清洗),Physionet 原 loader `pick_types(eeg=True)` 把额部通道砍光 → ICA 退化为峰度统计兜底;现已在 `moabb_loader.py::_load_run_raw` 额外保留 **Fp1/Fp2 并标记 eog 类型**作伪眼电参考,`select_channels_physionetmi`/`ica_denoise` 均改按 eog 类型(而非名字)识别,ICA 完成后 eog 被 drop,最终 pkl 仍只含 22 EEG 通道(已验证 S001 排除 2 个眼电成分、方差去除 0.14%→5.41%、幅度去除 0.06%→12.74%,BCI S1 回归 144×4 不受影响),每人每类 21-24 试次、总 ~2300/类、形状 (N,22,721) 均匀;**2026-09-09 用户决定:预处理环节固定为本版(Fp1/Fp2 眼电参考),旧版备份 `backup_moabb_preeogref_v34.5.pkl` 已删除;对比确认两版试次数逐类完全一致、新/旧 abs 幅度比 0.725(ICA 清洗所致能量差),后续全部实验以此 pkl 为准**。main.py 已支持 `REJECT_AMP_UV`(float/None) 与 `PHYSIONET_SUBJECTS` 环境变量;断点续跑 bug 已修(`_load_existing` 合并已有被试,勿回退)。**2026-09-09 已清理**:下载/补齐临时脚本 `_run_preprocess_loop.sh`/`_run_preprocess_remain.sh` 与旧管线质量报告 `saved_data/_physionet_quality_report.json` 均已删除(数据已落盘,重跑可重建)。

- physionetmi 的 train_pair 实验入口:`--dataset physionetmi`(auto 切 160Hz→win=240/step=120→M=5 窗,混合人数按 pkl 实际被试数动态化,per-subject 滤波器组数=n 被试)。论文清洗规范已验证全部对齐,无需另加步骤。

- WSL 原始数据缓存 `~/mne_data/`:**2026-09-09 已删 `MNE-eegbci-data`(Physionet 109 人原始 EDF,1.8GB,用户决定暂不需要;数据已固定落盘 `saved_data/*.pkl`,未来重跑预处理需重新下载);`MNE-bnci-data`(BCI-2a 9 人原始 GDF,位于 `~bci/database/001-2014`)保留勿删**。

- pkl 统一格式(加载器 `MAtt_baseline/utils/GetPhysionetMI.py: load_preprocessed_dataset`,所有复现脚本共用):dict 键 `aligned_{left,right,feet,hands}_epochs_list` + `subject_metadata`;数组 (n\_trials, 22ch, T);类序 left=0/right=1/feet=2/tongue=3。

- 官方与自建数据不通用:官方=\[0.5,2.25]s 切片、EOG 回归清洗、无 EA;自建=4.5s 全段、ICA+独立 EA。对比见 `_exp_compare_summary.json`。

- MAtt 复现脚本从仓库根目录直接跑即可,脚本内部会把 `MAtt_baseline/` 插入 sys.path。

## 公共 helper(复用优先)

- `MAtt_baseline/utils/functions.py`:`build_matt_model(win_samples, step_samples, args)` 与 `load_matt_model(...)`;注意 `sfreq` 为条件式 `250 if dataset=='bci2a' else 160`(physionetmi=160Hz 硬编码 bug 已修)。

- 根目录 `preprocessing.py`:`epochs_to_arrays(epochs_map, class_names, n_channels)`,两 loader 的 class\_names 第 4 类不同:physionetmi=`'hands'`,bci2a=`'tongue'`。

- `GetPhysionetMI.py` 私有 `_load_subject(sub, subject_data, sample_ratio, random_state)` = seen/unseen 共用分层采样。

## 命名约定

- 下划线前缀 `_*.py` = 官方复现/对照脚本(非一次性探针,可长期保留);已废弃的一次性探针/数据手术脚本已删除。

- 结果命名 `{实验名}_best_{acc,loss}.pt` + 同名 json/png,输出在 `MAtt_baseline/results/{profile_*,tune_*,checkpoint,figures,diagnosis}/`(tune\_\* 每目录含 pair\_training.json 全过程曲线)。

## Git

- remote `github.com/Skybilbilbil/Skybilbil_eeg`,单分支 `master`,HEAD=`51db6fe`(2026-09-12 核实);**工作区另有 8 个已改未提交文件**(AGENTS.md、`MAtt_baseline/mAtt/mAtt.py`、`MAtt_baseline/train_pair_with_recording.py`、`MAtt_baseline/utils/GetPhysionetMI.py`、`MAtt_baseline/utils/functions.py`、`main.py`、`moabb_loader.py`、`preprocessing.py`)与未跟踪的 `_*.py` 官方脚本 + `_exp_compare_summary.json` + `_run_bci_ref_backend.py` + `.trae/`;AGENTS.md 本身已被跟踪;实验产物(results/、\*.log)均被 gitignore;提交信息中文 + Conventional Commits(feat/fix/docs/refactor/chore)。

## 文档入口

- `MAtt_baseline/mAtt/理解脑电算法流程.md`(\~10000 行中文):全流程讲解、A1-A22 bug 修复清单、调参手册——理解本项目数学与设计决策的最快入口。

- `MAtt_baseline/README.md`:上游 MAtt 官方说明。

