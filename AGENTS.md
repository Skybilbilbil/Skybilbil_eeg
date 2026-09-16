# AGENTS.md

EEG 运动想象(MI)解码研究代码:SPD 流形上的注意力模型(基线为 NeurIPS 2022 MAtt)+ 自研预处理管线。这是研究代码,不是软件工程项目——无 CI、无 lint/format 配置、无测试框架、根目录无依赖清单。改动请最小化、聚焦研究需求。

> **历史结论、实验数字、坑、代码层状态、环境/数据细节,全部在 `实验记录.md`。**
> 本文件只放**长期有效**的东西:定位、用户规则、硬约束、当前进展摘要。不要往本文件堆实验细节。

## 快速定位(新 agent 从这里开始)

- **唯一训练主入口** `MAtt_baseline/train_pair_with_recording.py`(argparse 开关见 `--help`);**唯一数据** = `saved_data/*.pkl` 2 份;**唯一结果根** = `MAtt_baseline/results/`。

| 要做什么 | 去哪 |
| --- | --- |
| 任何训练对比(9 人混合 / 被试内 / Physionet103 / 留出被试) | `MAtt_baseline/train_pair_with_recording.py` |
| 纯被试内 BCI-2a 单被试 | `MAtt_baseline/run_matt_physionetmi.py`(改 `CONFIG['dataset']='bci2a'`;无模型开关) |
| 90 seen / 13 unseen | `MAtt_baseline/run_matt_cross_subject.py` |
| 模型本体 + 前后端边界 | `MAtt_baseline/mAtt/mAtt.py`(`mAtt_bci`、`BACKEND_PREFIXES`) |
| 模型统一构建入口 | `MAtt_baseline/utils/functions.py::build_matt_model` |
| 数据加载(所有脚本共用) | `MAtt_baseline/utils/GetPhysionetMI.py::load_preprocessed_dataset` |
| 逐层 / 跨被试诊断 | `MAtt_baseline/analyze_*.py`(纯 CPU 不训练,见 `实验记录.md`) |
| 重新生成 pkl(预处理) | 根目录 `main.py`(Physionet103)/ `main_bci2a.py`(BCI9),配置在文件顶部 |
| 历史结论数字 / 逐实验曲线 | `实验记录.md`;曲线 = `results/tune_*/pair_training.json` |

- **已删除的废弃分支**(2026-09-16):早期 MAtt profile/可视化脚本、根目录 LSTM 簇、`visualization/` 与 `.trae/`；历史结论保留在 `实验记录.md`，不要按旧路径恢复或扩展。
- **四条硬约束**:①数据只有 2 份 pkl,`load_preprocessed_dataset` 是唯一加载口;②复现必须带 `--seed 42`;③`saved_data/`、`results/` 被 gitignore,只在本地;④跑脚本先 `conda activate eeg_analysis`。

## 用户规则(所有 agent 必须遵守)

- **先思考后编码**:动手前明确陈述假设;不确定就主动提问;存在多种理解时列出来,不默默选一个;发现更简单方案直接指出、必要时提出反对;遇到困惑停下来说明并提问。

- **简单至上**:只写能解决问题的最少代码;不为一次性使用建抽象,不加"灵活性/可配置性";不处理不可能发生的异常;代码超出必要长度就重写简化。

- **外科手术式修改**:只动必须动的地方;不顺手"改进"相邻代码/注释/格式,不重构未损坏的部分;保持与现有代码风格一致;无关死代码可口头提及但不删除(除非用户要求);因改动产生的未使用导入/变量/函数必须清理。

- **目标驱动执行**:模糊任务先转化为可验证的成功标准(先写复现/无效输入测试,再让它通过);多步骤任务先给简短计划,每步带验证检查点。

- **算法与数学优先**:回复用中文;讲算法先讲底层数学原理(公式推导、复杂度、收敛性、几何解释),代码只是验证数学思想的工具;写代码前先确保用户理解核心数学逻辑,代码必须严格对应数学步骤并在关键处注释说明对应哪个公式。

- **完成后清理**:操作完成后清理无用的临时文件。

## 进度同步规则(所有 agent 必须遵守)

- **每完成一个大操作(跑完一组实验 / 代码重构 / 数据变更 / 得出重要结论),立即把结果写进 `实验记录.md`**,不是本文件。

- **本文件不承载实验细节。** 只有"长期有效、每次动手前都该知道"的内容才配进 AGENTS.md(定位、规则、硬约束、当前进展摘要一行);结论/数字/坑/产物路径一律进 `实验记录.md`。

- 只写**现状与问题**(做了什么、结果数字、结论、产物路径、坑);**不写推荐方向/未验证建议/待试方案,不写过程叙述**。

- 新结论推翻旧结论时,直接改写对应条目并更新"最后更新"日期。

- `实验记录.md` 是所有 agent 工具共享的**唯一进度事实源**;每次开始工作前先读它,干完活同步回它。

## 环境与数据速查(完整版见 `实验记录.md`)

- **WSL2 是唯一运行时。** 仓库 `D:\EEG_Analysis\Riemannian_Trajectories` 与 `/mnt/d/EEG_Analysis/Riemannian_Trajectories` 是同一物理目录(drvfs 挂载)。所有研究脚本在 WSL 里跑。
- **研究环境** = `/home/lenovo/miniconda3/envs/eeg_analysis`(Python 3.10.14,torch 2.6.0+cu124)。校验命令模板:`wsl -e bash -lc 'cd /mnt/d/EEG_Analysis/Riemannian_Trajectories && source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg_analysis && <cmd>'`。**不要**用 `MAtt_baseline/mAtt_env.yml` 建环境(上游 macOS 遗留)。
- **GPU = RTX 4060 Laptop 8GB。** 若 `torch.cuda.is_available()=False`,绝大多数情况是 DSH 文件沙箱拒绝以 O_RDWR 打开 `/dev/dxg`(**不是驱动坏了**,Windows 侧 `nvidia-smi.exe -L` 正常)。判别:`python -c "import os; os.open('/dev/dxg', os.O_RDONLY)"` 成功、`os.O_RDWR` 返回 EACCES 即为沙箱。**该命令单独提权重试,不要默认退回 CPU**(CPU 慢 2-3 倍,且 CPU/GPU 单折可差 10pp)。
- **数据**:只有 `saved_data/*.pkl` 2 份,统一经 `load_preprocessed_dataset` 加载;格式 = dict `aligned_{left,right,feet,hands}_epochs_list` + `subject_metadata`,数组 (n_trials, 22ch, T),类序 left=0/right=1/feet=2/tongue=3。
- `~` 下有 3 个以 Windows 路径命名的**空**占位目录(如 `~/D:\EEG_Analysis\...`),是 setup 副作用,**勿 cd 进去**(用它当工作目录会让 `@` 文件检索和组织工具索引全空)。

## 当前进展摘要(数字与依据见 `实验记录.md`)

- **被试内**:BCI-2a S3 峰值 82.8%,已超 MAtt 原文 74.71%。
- **真跨被试(留出被试)**:6 折均值 **46.3%**,范围 28.6–62.2%;同池 60/20/20 协议的 54.6% 相对它**虚高 8–10pp**。
- **瓶颈是域偏移,不是表示形式**:固定 22 通道 SPD 切空间表示被试内 5 折 **72.5%**,跨被试 9×9 迁移只有 **30.5%** → 丢掉 42pp 全在"坐标系不同"。
- **性能由"目标被试是谁"主导**:用 10 秒的 CSP+LDA "接收性"排序,前 3 名深度模型 **60.5%** vs 后 3 名 **32.0%**,两组**零重叠**。报告必须给 per-subject 分布。
- **已判死方向很多**(PCGrad、解剖先验基、零标签归一化适配、per-subject conv、参考后端初始化、IAF 窄带、RA、FBCSP 频带共享前端、域增广/虚拟被试、wd 调参……)——**动手前先查 `实验记录.md` 的「已证伪方向」,不要重试。**
- **主协议**已定为 subject-dependent：全部公开 trial 按类别分层随机 **70/15/15** 切分；模型不得接收 `subject_id`。原 A0T/A0E 仅为数据来源；跨 session 与严格留出被试只作外部泛化参照。

## 文档入口

- **`实验记录.md`(根目录)**:所有历史结论、实验数字、已证伪方向、机制推导、代码层状态与坑、数据/环境细节。**开工前必读。**
- `MAtt_baseline/mAtt/理解脑电算法流程.md`(~10000 行中文):全流程讲解、A1-A22 bug 修复清单、调参手册——理解本项目数学与设计决策的最快入口。
- `MAtt_baseline/README.md`:上游 MAtt 官方说明。
