# AGENTS.md

EEG 运动想象(MI)解码研究代码:SPD 流形上的注意力模型(基线为 NeurIPS 2022 MAtt)+ 自研预处理管线。这是研究代码,不是软件工程项目——无 CI、无 lint/format 配置、无测试框架、根目录无依赖清单。改动请最小化、聚焦研究需求。

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

## 项目实时进度(最后更新:2026-09-04)

### 用户决策/硬约束(勿改)

- 只做 BCI Competition IV-2a(9 人,250Hz);**PhysionetMI 暂停**。

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

### 注意力机制现状(2026-09-03 诊断)

- 当前独立基注意力**无法区分真实想象窗与静息窗**:自聚合权重 self\_im=0.125 与 self\_rest=0.125 相同,权重完全由物理窗位置决定,从未学会选窗。

- **内容可分性才是瓶颈**:SPD 原始特征"静息 vs 想象"最近质心可分性仅 56.7%,且投影到 Q/K 后不再下降 → 特征塌缩在前,注意力不背锅。

- 独立基 vs 共享基对比:共享基 S3=74.1% / S9=65.5%,独立基 S3=60.3% / S9=51.7% —— 独立基在低可分性下聚合更多噪声;但用户强制用独立基。

- 混合塌缩时 attn\_conc 全程 ≈1/M 均匀,注意力从未学会给想象窗加权。

### 已证伪方向(勿重试)

- **E 臂 通道先验锚定微调(2026-09-04,seed=42,150it,wd0.02,`--channel_prior_eps 0.15`** **新开关,产物** **`MAtt_baseline/results/tune_e_chprior/`)**:通道 scale 锚定在 先验±15% 邻域,混合 test 峰值 56.0→55.6 / 最终 54.6→54.6(**零增益**);scale 范数保住(-12%,与被试内同水平)但 conv1/conv2 照塌(0.44/0.47,与 A 臂逐位相同)、silhouette≈0、attn 均匀。**证明瓶颈是"方向"自由度(conv1/conv2 空间/频率投影),不是"长度"自由度(通道权重缩放);先验锚定只防通道关闭,救不了 conv**。

- **4-arm 机制定向实验(2026-09-02,seed=42,150it,GPU)**:A 基线 wd0.02→54.6%;B wd0.005→55.5%(train 60.8→64.0,test 仅 +0.9pp);C B+正交正则 ortho0.02→55.5%(零增益);D B+固定 C3/Cz/C4 通道掩膜→55.2%(略降)。产物 `MAtt_baseline/results/tune_{a_mixbase,b_mixwd,c_mixwdortho,d_mixcomb}/`,四臂曲线一致:it5-30 顶到 54-56% 平台后 test 对训练增长无响应。**框架内正则/掩膜微调不能突破 0.56 混合瓶颈**(train 涨 test 不涨 = 过拟合混合池被试组合)。

- silhouette(切空间类间分离度):混合全程 ≈0(-0.010\~+0.002)vs 被试内 0.062-0.079 → **特征 collapse 是瓶颈的直接量纲**。

- **IAF 个体频段(2026-09-03)**:按被试 μ 峰±2Hz 窄带滤波替代固定 8-30Hz → 跨被试 58.9%→46.7%(-12.2pp,丢失 β 信息)。

- **黎曼对齐 RA(2026-09-03)**:被试协方差黎曼均值白化替代 EA → 跨被试 58.9%→47.9%(-11.0pp,过度抹平个体差异丢类别信息)。

- 更早证伪:多尺度卷积(`profile_ms` 无净收益)、EA 用于混合增强(`official_band_ea` 全表最低 47.6%)。

### 混合瓶颈机制结论

- **conv1(空间滤波)**:不同被试空间分布差异 → 梯度方向冲突 → 权重缩至初值 15-25% → 空间特征塌缩。

- **conv2(时序滤波)**:不同被试 μ 峰频率差异(7.8-11.7Hz)→ 梯度方向冲突 → 权重缩至初值 20-30% → 时序特征塌缩。

- 数学本质(2026-09-04 确认):batch 内不同被试的梯度方向互斥 → 期望梯度 𝔼\[∂L/∂W]→0 → 更新方程退化为 `W←(1−ηλ)W`,解 `|W_t|=(1−ηλ)ᵗ·|W₀|` 指数压缩。正则/掩膜/wd 微调都不改变"期望梯度归零"这个拓扑事实,因此框架内旋钮全部无效(与 4-arm 结论互相印证)。

- 共同结局:batch 内梯度相消 → conv 权重被优化器(wd)"删除" → conv 输出特征 norm 0.058→0.0085(×7 衰退)→ 注意力输入塌缩 → 分类器无有效信号(分类器 W 范数仅 +15% vs 被试内 +70-80%)。

- 生理依据(`results/diagnosis/diagnosis_report.json`):被试间子空间主角度 60.4°(各自用不同电极组合)、MI 响应潜伏期差 7.54 窗、频谱个体差异(固定 8-30Hz 对各人非最优)。

### 代码层状态与坑

- `_smoke_test.py` 通过;BCI-2a 单被试被试内训练跑通(`run_matt_physionetmi.py` 改 `CONFIG['dataset']='bci2a'`,sub=1,120it → test\_acc 71.55%)。

- BCI2a 实测 T=1126、E2R M=4 窗(win1.5s/step0.75s);训练默认无 seed(dataloader shuffle 未固定;仅 `train_pair_with_recording.py` 支持 `--seed`,复现实验务必带上)。

- 模型结构锚点(2026-09-04 确认):conv1=`Conv2d(1,22,(22,1))` 全通道空间混合,conv2=`Conv2d(22,20,(1,12),pad6)` 12 点时序带通(默认非 separable/multiscale);`n_spatial_filters` 默认 22(trai\_pair 有 `--n_spatial_filters`)。

- **dropout 双默认值坑**:`mAtt_bci` 类内 `self.dropout=nn.Dropout(0.5)` 是硬编码默认;训练入口会覆盖为 0.3(train_pair `DROP_PROB=0.3`、`build_matt_model` 默认 `dropout=0.3`)。裸建 `mAtt_bci(...)` 不经训练入口会拿到 0.5,改 dropout 必须走入口。

- **架构严谨化重构(2026-09-06,git 待提交)**:mAtt.py 抽 `_SparseAttentionBase` 私有基类,mAtt_bci/mAtt_mamem/mAtt_cha 三子类配置化派生(构造顺序保持旧语义、state_dict 严格兼容);dropout 改为构造参数(默认 0.3,None=不建)取代"类内硬编码 0.5 + 入口覆盖";`ATTN_SCORE_ORIGINAL` 环境变量 → `AttentionManifold(score_original=True)` 参数;build_matt_model 补全全参数,三个训练入口统一经它构建;train_pair 窗口探测抽 `_infer_windows` helper。回归:150it/seed42 两轮独立验证,与 tune_a_mixbase 的 silhouette/参数范数/attn 逐位一致,cross_mix test_acc 三独立 run 落点 54.6/54.8/55.1%(spread 0.5pp = 1-2 测试样本,GPU 浮点 run 间噪声,验证目录已清理);`dropout=0.5` 历史仅 `_exp_compare.py` 依赖,已显式 `dropout=0.5` 保护。新坑:**analyze_\*/profile_\* 诊断脚本裸建 mAtt_bci 会随默认翻到 0.3(非历史基线来源,如需 0.5 请显式传)**。

- Windows 侧曾出现 25 个假性 M 文件 = CRLF 噪声(`git diff -w` 内容等价),无损还原,勿重复"修复"。

### 运行速查

- profile 生成器(13 个 argparse 开关全有效):`wsl -e bash -lc 'cd /mnt/d/EEG_Analysis/Riemannian_Trajectories/MAtt_baseline && source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg_analysis && python train_pair_with_recording.py --out_dir <NAME> --tag <TAG> --seed 42 [--wd 0.005 | --ortho_lambda 0.02 | --channel_gaussian | ...]'`

- `run_matt_physionetmi.py`(CONFIG 字典)跑 BCI-2a:改 `CONFIG['dataset']='bci2a'` 即可(auto 选 pkl + sfreq250);其 CONFIG **无模型开关**(channel\_gaussian/ortho\_lambda/manifold\_change/attn\_\* 经 build\_matt\_model 硬接线默认关),要翻这些开关只能走 train\_pair。

## 代码结构

- **根目录** = 自研预处理管线 + 官方 MAtt 复现脚本:

  - 预处理入口 `main.py`(PhysionetMI,109 剔 6 → 103 人)与 `main_bci2a.py`(BCI IV-2a,9 人):MOABB 加载 → 22 通道 → Picard ICA 去噪 → 8-30Hz → \[-1,3.5]s epoch → 逐被试独立 EA → 存 `saved_data/*.pkl`。无 CLI 参数,配置全在文件顶部。

  - 复现脚本(有 CLI,只打 stdout 不落盘):`_reproduce_official.py --mode within|cross|both`(官方数据跨 session,对照原文 74.71%)、`_reproduce_seeds.py`、`_exp_compare.py --exp 1,2,3`、`_ablation_official.py`、`_smoke_test.py`。

- **`MAtt_baseline/`** = 上游 MAtt(NeurIPS 2022)手工目录拷贝(非 git clone),已深度修改(中文注释、InstanceNorm 替 BN、多尺度卷积),是当前研究主战场。

  - 训练入口:`train_pair_with_recording.py`(有 argparse)、`run_matt_physionetmi.py`(CONFIG 字典)、`run_matt_cross_subject.py`(90 seen/13 unseen)。

- **已废弃(未删除,勿在其上扩展)**:上游原版入口 `mAtt_bci.py`/`mAtt_bcicha.py`/`mAtt_mamem.py`(期望本地 `data/*.mat`,从未跑过)、`visualize_matt.py`(需训练 npz)、`profile_10_subjects.py`/`analyze_profile_results.py`(早期 profile)、`diagnose_cross_subject.py`;遗留 LSTM 簇(Classification.py、classificability\_analysis.py、Dynamic\_trajectory.py、Data\_saving.py、csv\_loader.py、data\_reading.py、CPU\_test.py、adaptive\_alignment.py 等)及 `visualization/` 下失效脚本。

- `.dsh/` 是 agent 工具的任务板/回滚状态,忽略即可。

## 环境(WSL 为唯一运行时)

- 仓库 `D:\EEG_Analysis\Riemannian_Trajectories` 与 `/mnt/d/EEG_Analysis/Riemannian_Trajectories` 是**同一物理目录**(drvfs 挂载),git HEAD 两侧一致;所有研究脚本在 WSL 里跑。

- WSL2 Ubuntu-24.04.4 LTS(用户 lenovo,系统上限 16核/10GB);GPU:RTX 4060 Laptop 8GB,`torch.cuda.is_available()`=True。训练 `cuda if available` 不强制 GPU;预处理纯 CPU,`main*.py` 顶部设 OMP/OPENBLAS/MKL 线程为 8。

- **研究环境 =** **`/home/lenovo/miniconda3/envs/eeg_analysis`(Python 3.10.14)**:跑研究脚本前需手动 `conda activate eeg_analysis`(交互 shell 默认激活 base)。关键包:torch 2.6.0+cu124、mne 1.11.0、moabb 1.5.0、numpy 2.2.6、scikit-learn 1.7.2、pyriemann 0.10。condarc 走清华镜像。

- 不要用 `MAtt_baseline/mAtt_env.yml` 建环境(上游 macOS 遗留,py3.6.9+torch 1.10.2 CPU)。

- 校验命令模板:`wsl -e bash -lc 'cd /mnt/d/EEG_Analysis/Riemannian_Trajectories && source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg_analysis && <cmd>'`。

- 注意:`~/` 下有 3 个以 Windows 路径命名的空占位目录(含 `D:\EEG_Analysis\...`),是 setup 副作用,勿 cd 进去。

## 数据(最易踩坑)

- `saved_data/`、`results/`、`MAtt_baseline/results/` 均被 gitignore,大文件只在本地。

- 现存 pkl:`preprocessed_data_bci2a_raw_ica.pkl`(\~490MB)、`official_bciciv2a_matt_raw.pkl`(\~733MB,官方 mat 组装,meta 含 T\_split/E\_split 供跨 session 协议)。**`preprocessed_data_moabb_raw_ica.pkl`** **当前缺失**——默认跑 physionetmi 的脚本会直接失败,要么先跑 `main.py` 生成,要么 dataset 改 `'bci2a'`。

- WSL 原始数据缓存在 `~/mne_data/`:MNE-eegbci-data 仅 27/109 人(跑全量 main.py 会现场下载 \~5GB);MNE-bnci-data 9 人齐全,位于遗留临时目录 `~/bci/database/001-2014/`。

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

- remote `github.com/Skybilbilbil/Skybilbil_eeg`,单分支 `master`,HEAD=`ab8f64e`;AGENTS.md 本身已被跟踪;实验产物(results/、\*.log)均被 gitignore;提交信息中文 + Conventional Commits(feat/fix/docs/refactor/chore)。

## 文档入口

- `MAtt_baseline/mAtt/理解脑电算法流程.md`(\~10000 行中文):全流程讲解、A1-A22 bug 修复清单、调参手册——理解本项目数学与设计决策的最快入口。

- `MAtt_baseline/README.md`:上游 MAtt 官方说明。

