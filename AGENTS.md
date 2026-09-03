# AGENTS.md

EEG 运动想象(MI)解码研究代码:SPD 流形上的注意力模型(基线为 NeurIPS 2022 MAtt)+ 自研预处理管线。这是研究代码,不是软件工程项目——无 CI、无 lint/format 配置、无测试框架、根目录无依赖清单。改动请最小化、聚焦研究需求。

## 进度同步规则(所有 agent 必须遵守)

**每完成一个大操作(跑完一组实验 / 代码重构 / 数据变更 / 得出重要结论),必须立即把结果同步进下方「项目实时进度」章节**:只写事实(做了什么、结果数字、结论、产物路径),不写过程叙述;新结论推翻旧结论时直接改写对应条目并更新"最后更新"日期。本文件是所有 agent 工具共享的**唯一进度事实源**。

## 项目实时进度(最后更新:2026-09-03)

### 研究焦点与用户决策(2026-09-02/03 确定)
- **只做 BCI Competition IV-2a(9 人,250Hz);PhysionetMI 暂停**。
- **不做严格 LOO 跨被试,只做 9 被试混合**(9 人全量试次拼接→60/20/20 随机切分,random_state=42,有被试重叠,数字偏乐观)。
- 路线:被试内超原文(已达成)→ 攻 9 混合;遇到必须改算法框架的问题时,先全面复盘分析再给建议。

### 代码层状态(2026-09-02 全链路验证通过,无代码问题)
- `_smoke_test.py` 通过;BCI-2a 单被试被试内训练跑通(`run_matt_physionetmi.py` 改 `CONFIG['dataset']='bci2a'`,sub=1,120it → test_acc 71.55%)。
- 已知事实:BCI2a 实测 T=1126、E2R M=4 窗(win1.5s/step0.75s);训练默认无 seed(dataloader shuffle 未固定;仅 `train_pair_with_recording.py` 支持 `--seed`,复现实验务必带上);Windows 侧曾出现 25 个假性 M 文件 = CRLF 噪声(`git diff -w` 内容等价),无损还原。

### 核心结果(截至 2026-09-02)
| 协议 | 峰值 | 最终 | 出处 |
|---|---|---|---|
| 被试内 S3 | **82.8%** | 81.0% | `MAtt_baseline/results/profile_final/pair_training.json`(150it, lr5e-4, wd0.02, M=4) |
| 被试内 S9 | 81.0% | 79.3% | 同上 |
| 9 混合 cross_mix | 56.5% | 54.5% | 同上 |

- **被试内已超原文 74.71%**(MAtt NeurIPS 2022);9 混合 gap≈24pp = 当前唯一瓶颈。
- 窗长是框架内最强旋钮(`_exp_compare_summary.json` exp3):win5(2.5s窗) S3=82.8/9mix=56.6;win3(3.5s窗) 9mix=59.8(9mix 最佳)。
- 类序 left=0/right=1/feet=2/tongue=3;**feet 是混合塌陷最重的类**(cross per-class 0.339,28% 误判为 tongue)。

### 已证伪方向(勿重试)
- **4-arm 机制定向实验(2026-09-02,seed=42,150it,GPU)**:A 基线 wd0.02→54.6%;B wd0.005→55.5%(train 60.8→64.0 但 test 仅 +0.9pp);C B+正交正则 ortho0.02→55.5%(零增益);D B+固定 C3/Cz/C4 通道掩膜→55.2%(略降)。产物 `MAtt_baseline/results/tune_{a_mixbase,b_mixwd,c_mixwdortho,d_mixcomb}/`,曲线形状四臂一致:it5-30 顶到 54-56% 平台后 test 对训练增长无响应。**框架内正则/掩膜微调不能突破 0.56 混合瓶颈**(train 涨 test 不涨 = 过拟合混合池被试组合)。
- silhouette(切空间类间分离度):混合全程 ≈0(-0.010~+0.002)vs 被试内 0.062-0.079 → **特征 collapse 是瓶颈的直接量纲**。
- 更早证伪:多尺度卷积(`profile_ms` 无净收益)、EA 用于混合增强(`official_band_ea` 全表最低 47.6%)。

### 混合瓶颈机制结论
- 跨被试训练时,不同被试把空间/时间滤波器往相反方向推 → batch 内梯度相消 → conv/IN 权重缩到初值 15-25%(被优化器"删除")→ 注意力输入塌缩;注意力本身不背锅(attn_conc 全程≈1/M 均匀,从未学会选窗)。
- 生理依据(`results/diagnosis/diagnosis_report.json`):被试间子空间主角度 60.4°(各自用不同电极组合)、MI 响应潜伏期差 7.54 窗、频谱个体差异(固定 8-30Hz 对各人非最优)。

### 待拍板的候选方向(按证据强度排序)
1. **黎曼域对齐 RA/切空间对齐**(预处理级,零架构改动;Zanini2018、CovMix +4.5~9.4pp;pyriemann 0.10 已装)——默认推荐。
2. AdaBN 式输入统计对齐(需小改 `build_matt_model`;文献 +7.4pp)。
3. 逐被试个性化频带 IAF(需重跑预处理;Gaur2021)。

### 运行速查
- profile 生成器(13 个 argparse 开关全有效):`wsl -e bash -lc 'cd /mnt/d/EEG_Analysis/Riemannian_Trajectories/MAtt_baseline && source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg_analysis && python train_pair_with_recording.py --out_dir <NAME> --tag <TAG> --seed 42 [--wd 0.005 | --ortho_lambda 0.02 | --channel_gaussian | ...]'`
- `run_matt_physionetmi.py`(CONFIG 字典)跑 BCI-2a:改 `CONFIG['dataset']='bci2a'` 即可(自动选 pkl + sfreq250);但其 CONFIG **无模型开关**(channel_gaussian/ortho_lambda/manifold_change/attn_* 经 build_matt_model 硬接线默认关),要翻这些开关只能走 train_pair。

## 结构:两条主线

- **根目录** = 自研预处理管线 + 官方 MAtt 论文复现脚本
  - 入口 `main.py`(PhysionetMI,109 剔 6 → 103 人)与 `main_bci2a.py`(BCI Competition IV-2a,9 人):MOABB 加载 → 22 通道 → Picard ICA 去噪 → 8-30Hz → [-1,3.5]s epoch → 逐被试独立 EA → 存 `saved_data/*.pkl`。**无 CLI 参数**,配置全在文件顶部常量。
  - 复现脚本(有 CLI):`_reproduce_official.py --mode within|cross|both`(官方数据跨 session,对照原文 74.71%)、`_reproduce_seeds.py`、`_exp_compare.py --exp 1,2,3`、`_ablation_official.py`。
- **`MAtt_baseline/`** = 上游 MAtt(NeurIPS 2022)的**手工目录拷贝(非 git clone)**,已被深度修改(中文注释、InstanceNorm 替换 BN、多尺度卷积等),是当前研究主战场
  - 训练入口:`run_matt_physionetmi.py`(CONFIG 字典改参,无 argparse)、`run_matt_cross_subject.py`(90 seen / 13 unseen)、`train_pair_with_recording.py`(有 argparse,`--out_dir`)。
  - **已废弃(未删除,勿在其上扩展)**:`mAtt_bci.py` / `mAtt_bcicha.py` / `mAtt_mamem.py` 是上游原版入口,期望本地 `data/*.mat`(从未在本机跑过);`visualize_matt.py`(需训练 npz);`profile_10_subjects.py` / `analyze_profile_results.py`(早期 profile,已由 profile_final/profile_ms 取代);`diagnose_cross_subject.py`。
- **遗留 LSTM 簇(已废弃(未删除),早期实验,当前主线不用,勿在其上扩展)**:`Classification.py`、`classificability_analysis.py`、`Dynamic_trajectory.py`、`Data_saving.py`、`csv_loader.py`、`data_reading.py`、`CPU_test.py`、`adaptive_alignment.py`,以及 `visualization/` 下的失效脚本。
- `.dsh/` 是 agent 工具的任务板/回滚状态,忽略即可。

## 环境(WSL 为唯一运行时)

- **仓库位置**:`D:\EEG_Analysis\Riemannian_Trajectories`(Windows)与 `/mnt/d/EEG_Analysis/Riemannian_Trajectories`(WSL)是**同一物理目录**(drvfs 挂载),git HEAD 两侧一致;WSL 文件系统内无副本。所有研究脚本在 WSL 里跑。
- **发行版**:WSL2 Ubuntu-24.04.4 LTS,用户 `lenovo`,systemd 开启,Mirrored 网络模式,内存上限 10GB。GPU:NVIDIA RTX 4060 Laptop 8GB,WSL 内 `torch.cuda.is_available()` = True。
- **Python 三套**:系统 `/usr/bin/python3`(3.12.3);conda base(`/home/lenovo/miniconda3`,3.14.6,交互 shell 默认激活);**研究环境 `/home/lenovo/miniconda3/envs/eeg_analysis` = Python 3.10.14**——跑研究脚本前需手动 `conda activate eeg_analysis`。
- **eeg_analysis 关键包**:torch 2.6.0+cu124、mne 1.11.0、moabb 1.5.0、numpy 2.2.6、scipy 1.15.3、scikit-learn 1.7.2、**pyriemann 0.10**(已安装,黎曼运算可用);~/.condarc 走清华 TUNA 镜像。
- **不要用 `MAtt_baseline/mAtt_env.yml` 建环境**——那是上游 macOS 遗留(py 3.6.9 + torch 1.10.2 CPU),本机从未用过。
- 训练一律 `cuda if available`,不强制 GPU;预处理纯 CPU,`main.py`/`main_bci2a.py` 顶部把 OMP/OPENBLAS/MKL 线程设为 8。
- 校验命令模板(在 Windows 侧跑 WSL):`wsl -e bash -lc 'cd /mnt/d/EEG_Analysis/Riemannian_Trajectories && source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg_analysis && <cmd>'`。
- 注意:交互 shell 默认激活 base,`~/` 下还有 3 个以 Windows 路径命名的**空占位目录**(含 `D:\EEG_Analysis\Riemannian_Trajectories`),是 setup 副作用,勿 cd 进去。

## 数据(最易踩坑)

- `saved_data/`、`results/`、`MAtt_baseline/results/` 均被 gitignore——大文件只在本地,换机器/重新 clone 后没有。
- 现存 pkl:`preprocessed_data_bci2a_raw_ica.pkl`(~490MB)、`official_bciciv2a_matt_raw.pkl`(~733MB,官方 mat 组装,meta 内含 `T_split`/`E_split` 供跨 session 协议);另有 `_physionet_quality_report.json`。
- **`preprocessed_data_moabb_raw_ica.pkl` 当前缺失**——默认跑 physionetmi 的脚本(`run_matt_physionetmi.py`、`run_matt_cross_subject.py` 等)会直接失败。要么先跑 `main.py` 生成,要么把 dataset 改为 `'bci2a'`。
- WSL 内原始数据缓存在 `~/mne_data/`:MNE-eegbci-data 仅 27/109 人(跑全量 `main.py` 会现场下载约 5GB);MNE-bnci-data 9 人齐全但位于遗留临时目录 `~/bci/database/001-2014/`。
- pkl 统一格式(加载器 `MAtt_baseline/utils/GetPhysionetMI.py: load_preprocessed_dataset`,所有复现脚本共用):dict 键 `aligned_{left,right,feet,hands}_epochs_list` + `subject_metadata`;数组 (n_trials, 22ch, T);类序 left=0 / right=1 / feet=2 / hands(tongue)=3。
- 官方与自建数据不通用:官方 = [0.5,2.25]s 切片(`[124:562]`)、EOG 回归清洗、无 EA;自建 = 4.5s 全段、ICA + 独立 EA。对比结论见 `_exp_compare_summary.json`。
- MAtt 复现脚本从仓库根目录直接跑即可,脚本内部已把 `MAtt_baseline/` 插入 sys.path。

## 公共 helper(重构后,复用优先)

- `MAtt_baseline/utils/functions.py`:`build_matt_model(win_samples, step_samples, args)` 与 `load_matt_model(path, ...)`——供 `run_matt_physionetmi.py` 与 `run_matt_cross_subject.py` 复用(原各自内联的 load_model 已抽取)。另注意 `sfreq` 为条件式 `250 if args['dataset']=='bci2a' else 160`(修复 physionetmi=160Hz 硬编码 bug)。
- 根目录 `preprocessing.py`:`epochs_to_arrays(epochs_map, class_names, n_channels)`——供 `moabb_loader.py` 与 `moabb_loader_bci2a.py` 复用。注意两 loader 传的 class_names 第 4 类不同:physionetmi = `'hands'`,bci2a = `'tongue'`。
- `MAtt_baseline/utils/GetPhysionetMI.py` 私有 helper `_load_subject(sub, subject_data, sample_ratio, random_state)`:提取 seen/unseen 两循环体共用的分层采样。
- `train_pair_with_recording.py` 的双次 `split_train_valid_test` 调用已合并为单次。

## 命名约定与陷阱

- **下划线前缀 `_*.py` = 官方 MAtt 复现/对照脚本**(非一次性探针):`_reproduce_official.py`、`_reproduce_seeds.py`、`_ablation_official.py`、`_exp_compare.py`(+ `_exp_compare_summary.json`)、`_verify_newdata_attn.py`、`_smoke_test.py`(无害冒烟)。早期的一次性探针/数据手术脚本已全部删除并提交。
- 复现/消融脚本只打 stdout 不落盘,需保留数值请自行重定向。
- 结果命名:`{实验名}_best_{acc,loss}.pt` + 同名 json/png;输出在 `MAtt_baseline/results/{profile_*,tune_*,checkpoint,figures,diagnosis}/`(tune_* 为参数扫描实验,每目录含 pair_training.json 全过程曲线)。

## Git

- remote `github.com/Skybilbilbil/Skybilbil_eeg`,单分支 `master`;HEAD=`ab8f64e`,共 9 个提交(初始 d7924b8 → 研究快照 134fc09 → 清理 f514015 → 死代码 e94d6ac → sfreq 修复 513b4e3 → helper 抽取 bc18efa/bdfc118 → AGENTS.md 重写 ab8f64e)。AGENTS.md 本身已被 git 跟踪。工作树仅剩 7 个 untracked 官方复现脚本(`_ablation_official.py`、`_exp_compare.py`、`_exp_compare_summary.json`、`_reproduce_official.py`、`_reproduce_seeds.py`、`_smoke_test.py`、`_verify_newdata_attn.py`);实验产物(results/、*.log)均被 gitignore。
- 提交信息中文 + Conventional Commits(feat/fix/docs/refactor/chore)。

## 文档入口

- `MAtt_baseline/mAtt/理解脑电算法流程.md`(~10000 行中文):全流程讲解、A1-A22 bug 修复清单、调参手册——理解本项目数学与设计决策的最快入口。
- `MAtt_baseline/README.md`:上游 MAtt 官方说明。
