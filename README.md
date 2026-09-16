# Riemannian Trajectories

BCI Competition IV-2a 运动想象 EEG 研究：以 SPD 流形注意力模型（MAtt）为主线，研究个体差异与被试内解码。

## 当前研究协议

主协议为 **subject-dependent 的 trial-level 70/15/15 分层切分**：每位被试的全部已公开 trial 合并后，按运动想象类别随机分层为 70% 训练、15% 验证、15% 最终测试。模型只接收 EEG，**不接收被试身份**。

原始 `A0T/A0E` 仅是数据来源，并不约束本研究的数据边界。`cross_session` 和严格留出被试实验仅作为额外诊断，不替代主协议结果。

## 结构

```text
.
├── main.py / main_bci2a.py          # PhysionetMI / BCI-2a 预处理入口
├── preprocessing.py / alignment.py  # 公共预处理与 EA
├── moabb_loader*.py                 # 数据集加载
├── saved_data/                      # 本地数据（gitignore）
├── 实验记录.md                       # 唯一实验事实源，动手前必读
├── AGENTS.md                        # 约束、定位与运行环境
├── _*.py                            # 官方复现与历史对照
└── MAtt_baseline/
    ├── train_pair_with_recording.py # 唯一通用训练入口
    ├── mAtt/                        # MAtt、SPD 与优化器实现
    ├── utils/                       # 统一数据加载与模型构建
    ├── analyze_*.py                 # 不训练的诊断脚本
    ├── make_individual_diff_report.py
    ├── 个体化差异研究报告.md
    └── results/                     # 本地训练产物、图、日志与断点结果（gitignore）
```

## 运行

唯一运行环境是 WSL 的 `eeg_analysis` Conda 环境。复现实验必须固定随机种子：

```sh
cd /mnt/d/EEG_Analysis/Riemannian_Trajectories/MAtt_baseline
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eeg_analysis
python train_pair_with_recording.py --seed 42 --split samepool
```

`train_pair_with_recording.py --help` 给出全部开关。历史结果与已证伪路线见 `实验记录.md`；算法与实现细节见 `MAtt_baseline/mAtt/理解脑电算法流程.md`。
