
import os

import mne
import numpy as np
import scipy.stats

# 2026-10-24 适配实验: 与 moabb_physionetmi.PHYSIONET_ALL_EEG 同名开关 ——
# 1 时 preprocess_physionetmi_raw 跳过 22 通道截取, 保留 loader 给的全部通道(62 EEG+2 eog)
_ALL_EEG_MODE = os.environ.get("PHYSIONET_ALL_EEG", "0") == "1"

# 平均参考处理

# 滤波处理
def filter(raw) :
    raw_copy = raw.copy()
    raw_copy.filter(l_freq=8, h_freq=30, picks=['eeg'])
    return raw_copy
 
# 1. BCI IV-2a 完整的 22 个 EEG 通道 -> 标准 10-20 系统名称映射系
def _normalize_ch_name(name):
    """统一通道名格式，忽略大小写、点号、前后空格"""
    return name.strip().upper().replace('.', '')


def select_channels_physionetmi(raw):
    """
    PhysionetMI 数据集通道选择。
    PhysionetMI 有 64 个 EEG 通道，我们选择与 BCI IV-2a 类似的中央区域通道，
    并保留可用的 EOG 通道。
    """
    raw_copy = raw.copy()

    # 选择与 BCI IV-2a 对应的 22 个 EEG 通道（从 PhysionetMI 的 64 个通道中）
    eeg_channels = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
    ]

    # 建立规范化后的通道名映射，兼容 'Fz', 'FZ', 'Fz.', 'fz.' 等写法
    norm_to_orig = {_normalize_ch_name(ch): ch for ch in raw_copy.ch_names}

    channels_to_keep = []
    for ch in eeg_channels:
        norm_ch = _normalize_ch_name(ch)
        if norm_ch in norm_to_orig:
            channels_to_keep.append(norm_to_orig[norm_ch])

    if len(channels_to_keep) != len(eeg_channels):
        missing = set(_normalize_ch_name(ch) for ch in eeg_channels) - set(norm_to_orig.keys())
        raise ValueError(
            f"PhysionetMI 22 通道选择失败，只找到 {len(channels_to_keep)}/22 个通道。"
            f"缺失: {sorted(missing)}。原始通道名示例: {raw_copy.ch_names[:10]}"
        )

    # 保留 eog 类型通道:BCI 的 EOG1/2/3 名字含 EOG,Physionet loader 标记的
    # Fp1/Fp2 伪参考名字不含 EOG,统一按 eog 类型(而非名字)识别保留
    eog_channels = [raw_copy.ch_names[i] for i in mne.pick_types(raw_copy.info, eog=True, exclude=[])]
    channels_to_keep.extend(eog_channels)

    raw_copy.pick_channels(channels_to_keep)

    # 设置 EOG 通道类型
    eog_types = {ch: "eog" for ch in eog_channels if ch in raw_copy.ch_names}
    if eog_types:
        raw_copy.set_channel_types(eog_types)

    # 设置电极坐标
    raw_copy.set_montage("standard_1020", match_case=False, on_missing="warn")

    # 断言：必须保留 22 个 EEG 通道
    n_eeg = len(mne.pick_types(raw_copy.info, eeg=True, exclude=[]))
    if n_eeg != len(eeg_channels):
        raise ValueError(f"选择后 EEG 通道数为 {n_eeg}，期望 {len(eeg_channels)}")

    return raw_copy


def ica_denoise(raw, threshold=2.5, max_exclude=2, adaptive_threshold=True):
    """
    改进的高效ICA去噪算法（无打印、无可视化版本）
    
    参数:
        raw: MNE Raw，已选好 EEG+EOG 通道
        threshold: 眼电检测阈值 (Z-score 阈值，通常 2.0~3.0)
        max_exclude: 最多排除的ICA成分数
        adaptive_threshold: 是否使用自适应阈值
        
    返回:
        raw_clean: 去除伪迹后，且仅包含 EEG 通道的 Raw 数据
        ica_metrics: 效果评估指标字典
    """
    raw_clean = raw.copy()
    
    picks_eeg = mne.pick_types(raw_clean.info, eeg=True, exclude=[])
    picks_eog = mne.pick_types(raw_clean.info, eog=True, exclude=[])
    
    if len(picks_eeg) == 0:
        return raw_clean, {}
        
    # 核心改进 1: 创建专用于拟合ICA的 1Hz 高通滤波数据（极大地提高ICA分离质量）
    #  EOG 通道一并高通（避免直流漂移污染成分），EEG 与 EOG 统一进入 ICA。
    #  降低计算量：成分数 = EEG 通道数（Physionet 已先选 22 通道）、max_iter 200→100。
    #  ICA 复杂度 ≈ n_comp²·iter，成分 40→22 约省 70%，通道先选 22 再 ICA 另省 24 倍。
    picks_ica = mne.pick_types(raw_clean.info, eeg=True, eog=True, exclude=[])
    raw_for_ica = raw_clean.copy().filter(l_freq=1.0, h_freq=None, picks=picks_ica, verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=len(picks_eeg),   # 成分数 = EEG 通道数（22 通道→22 个成分）
        method='picard',
        random_state=42,
        max_iter=100,
    )
    
    # 在 1Hz 滤波后的数据上拟合，但最终应用在未过度滤波的数据上
    ica.fit(raw_for_ica, verbose=False)
    
    # 按 eog 类型识别参考通道(BCI 真实 EOG / Physionet Fp1/Fp2 伪参考均为此类型)
    eog_chs = [raw_clean.ch_names[i] for i in mne.pick_types(raw_clean.info, eog=True, exclude=[])]
    detected_components =[]
    eog_detection_used = False

    if len(eog_chs) > 0:
        # 基于 EOG 通道检测
        eog_detection_used = True
        for eog_ch in eog_chs:
            try:
                indices, scores = ica.find_bads_eog(
                    raw_clean, ch_name=eog_ch, threshold=threshold, measure='zscore', verbose=False
                )
                if adaptive_threshold and len(scores) > 0:
                    abs_scores = np.abs(scores)
                    median = np.median(abs_scores)
                    mad = np.median(np.abs(abs_scores - median))
                    adaptive_thresh = median + 3 * mad
                    filtered_indices =[i for i, score in enumerate(abs_scores) if score >= adaptive_thresh]
                    detected_components.extend(filtered_indices)
                else:
                    detected_components.extend(indices)
            except Exception as e:
                print(f'warning: 无法检测 EOG 成分: {e}')
    else:
        # 无 EOG 通道：先尝试用额部 EEG 通道 (Fpz/AFz) 作为"伪 EOG"辅助检测
        pseudo_eog_candidates = [ch for ch in raw_clean.ch_names if ch.upper() in ['FPZ', 'AFZ', 'FP1', 'FP2']]
        for pseudo_ch in pseudo_eog_candidates:
            try:
                indices, scores = ica.find_bads_eog(
                    raw_clean, ch_name=pseudo_ch, threshold=threshold * 1.5, measure='zscore', verbose=False
                )
                if indices:
                    detected_components.extend(indices)
                    eog_detection_used = True
            except Exception as e:
                print(f'warning: 无法检测 EOG 成分: {e}')

        # 如果伪 EOG 也没检测到，再用统计降级方案
        if not detected_components:
            detected_components = _detect_artifacts_by_statistics(ica, raw_clean, adaptive=adaptive_threshold)

    # 去重并限制最大排除数量
    unique_components = list(dict.fromkeys(detected_components))
    final_exclude = unique_components[:max_exclude]

    metrics = {}
    if final_exclude:
        ica.exclude = final_exclude
        ica.apply(raw_clean, verbose=False)
        # 注意：必须在 pick_types 前计算 metrics，否则 picks_eeg 索引会越界
        metrics = _calculate_ica_metrics(raw, raw_clean, picks_eeg, final_exclude)
        
    # =========================================================================
    # 🌟 核心修改 5：在所有处理（包括指标计算）完成后，仅保留 EEG 通道 🌟
    # =========================================================================
    raw_clean.pick_types(eeg=True)
    
    return raw_clean, metrics


def _detect_artifacts_by_statistics(ica, raw_data, adaptive=True):
    """基于非高斯性（峰度）的统计学伪迹检测"""
    sources = ica.get_sources(raw_data).get_data()
    n_components = sources.shape[0]

    kurtosis_scores =[]
    for i in range(n_components):
        # 眼电和心电等伪迹具有极高的非高斯性，峰度(Kurtosis)是最佳指标
        kurt_val = abs(scipy.stats.kurtosis(sources[i]))
        kurtosis_scores.append(kurt_val)

    kurtosis_scores = np.array(kurtosis_scores)

    if adaptive:
        # 使用 IQR 自适应阈值，比固定 Z-score 更稳健
        q1 = np.percentile(kurtosis_scores, 25)
        q3 = np.percentile(kurtosis_scores, 75)
        iqr = q3 - q1
        threshold_val = q3 + 1.5 * iqr
        artifact_indices = np.where(kurtosis_scores > threshold_val)[0].tolist()
    else:
        # 固定 Z-score
        z_scores = (kurtosis_scores - np.mean(kurtosis_scores)) / (np.std(kurtosis_scores) + 1e-8)
        artifact_indices = np.where(z_scores > 2.0)[0].tolist()

    # 按异常程度排序
    artifact_indices.sort(key=lambda idx: kurtosis_scores[idx], reverse=True)

    return artifact_indices


def _calculate_ica_metrics(raw_before, raw_after, picks_eeg, excluded_components):
    """计算纯粹的统计学改善指标"""
    data_before = raw_before.get_data(picks=picks_eeg)
    data_after = raw_after.get_data(picks=picks_eeg)
    
    # 核心改进 4: 移除了毫无意义的脑电 SNR 计算，改为 Variance Reduction
    var_before = np.var(data_before, axis=1)
    var_after = np.var(data_after, axis=1)
    # 零保护必须用【相对】量:MNE 的 Raw 以伏特存储,22 通道方差实测中位数仅 1.17e-10,
    # 原先的绝对 1e-8 比它大 ~82 倍,会把"方差去除百分比"整体压低约 80 倍(见 实验记录.md)。
    var_reduction = np.mean((var_before - var_after) / np.maximum(var_before, 1e-30) * 100)
    
    rms_before = np.sqrt(np.mean(data_before**2, axis=1))
    rms_after = np.sqrt(np.mean(data_after**2, axis=1))
    # 幅度(rms ~1e-5)受同一常数的影响仅 0.09%,但同样改用相对保护以保持一致
    amp_reduction = np.mean((rms_before - rms_after) / np.maximum(rms_before, 1e-30) * 100)
    
    kurtosis_before = scipy.stats.kurtosis(data_before.flatten())
    kurtosis_after = scipy.stats.kurtosis(data_after.flatten())
    kurt_reduction = kurtosis_before - kurtosis_after
    
    metrics = {
        'variance_reduction_percent': float(var_reduction),
        'amplitude_reduction_percent': float(amp_reduction),
        'kurtosis_reduction': float(kurt_reduction),
        'components_excluded': len(excluded_components)
    }
    
    return metrics

# PhysionetMI 四分类运动想象事件配置 (left_hand / right_hand / feet / hands)
_EPOCH_EVENT_CONFIG = {
    "names": ("left_hand", "right_hand", "feet", "hands"),
    "event_id": {
        "left_hand": 1, "right_hand": 2, "feet": 3, "hands": 4,
    },
    "left": "left_hand", "right": "right_hand",
    "feet": "feet", "hands": "hands",
}

# BCIC-IV-2a 四分类事件配置 (left_hand / right_hand / feet / tongue)
_BCI2A_EVENT_CONFIG = {
    "names": ("left_hand", "right_hand", "feet", "tongue"),
    "event_id": {
        "left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4,
    },
    "left": "left_hand", "right": "right_hand",
    "feet": "feet", "hands": "tongue",
}


def _detect_epoch_event_config(raw, event_config=None):
    """检查 annotations 是否包含指定四分类事件。"""
    if event_config is None:
        event_config = _EPOCH_EVENT_CONFIG

    if raw.annotations is None or len(raw.annotations) == 0:
        raise ValueError("当前被试没有 annotations，无法创建 epochs")

    descriptions = set(raw.annotations.description.astype(str))
    if set(event_config["names"]).issubset(descriptions):
        return event_config

    raise ValueError(
        f"未识别的运动想象事件标注，期望 {event_config['names']}；"
        f"实际标注: {sorted(descriptions)}"
    )


def epochs(raw, tmin=-1.0, tmax=5.0, event_config=None, reject_amp_uv=None):
    """
    切分运动想象 Epochs，并支持按幅度剔除坏 trial。

    参数:
        tmin: 事件前截取时长（秒）。负值表示包含事件前的静息期。
        tmax: 事件后截取时长（秒）。建议大于想象时长，以包含想象结束后的静息期。
        event_config: 事件配置，默认 PhysionetMI；可传入 BCI-IV-2a 配置。
        reject_amp_uv: peak-to-peak 幅度阈值（µV）。不为 None 时，任一 EEG 通道
            在 epoch 内峰峰值超过该阈值的 trial 会被剔除（MNE auto_reject）。
            推荐 100µV（运动伪迹/眨眼），在 8-30Hz 滤波后的干净信号上使用。

    默认 tmin=-1.0, tmax=5.0 覆盖：
        -1.0~0.0s : 事件前静息（基线）
         0.0~4.0s : 完整运动想象时段
         4.0~5.0s : 想象结束后回到静息
    """
    raw_copy = raw.copy()
    config = _detect_epoch_event_config(raw_copy, event_config)
    event_id = config["event_id"]
    code_values = list(event_id.values())

    events, _ = mne.events_from_annotations(
        raw_copy,
        event_id=event_id,
    )

    events = events[np.isin(events[:, 2], code_values)]
    if events.size == 0:
        raise ValueError(
            f"当前被试 annotations 中没有四类运动想象事件: {config['names']}"
        )

    reject_dict = None
    if reject_amp_uv is not None:
        # MNE reject 参数按通道类型（'eeg': 阈值）给出，不按通道名
        # 量纲自适应：MNE Raw 数据可能以 V 存储（Physionet/BCI2a EDF 原始单位）
        picks_eeg = mne.pick_types(raw_copy.info, eeg=True, exclude=[])
        _d = raw_copy.get_data(picks=picks_eeg[:1])
        scale = 1e-6 if (_d.size and float(np.abs(_d).max()) < 0.01) else 1.0
        reject_dict = {'eeg': reject_amp_uv * scale}

    epochs_obj = mne.Epochs(
        raw_copy,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        event_repeated="drop",
        reject=reject_dict,
    )

    left_epochs = epochs_obj[config["left"]]
    right_epochs = epochs_obj[config["right"]]
    feet_epochs = epochs_obj[config["feet"]]
    hands_epochs = epochs_obj[config["hands"]]

    # 检查各类别 trial 数量，防止空类别导致后续对齐/拼接失败
    epoch_counts = {
        config["left"]: len(left_epochs),
        config["right"]: len(right_epochs),
        config["feet"]: len(feet_epochs),
        config["hands"]: len(hands_epochs),
    }
    for name, count in epoch_counts.items():
        if count == 0:
            raise ValueError(
                f"当前被试缺少类别 '{name}' 的 epochs，无法构建四分类数据。"
                f"各类别数量: {epoch_counts}"
            )

    return left_epochs, right_epochs, feet_epochs, hands_epochs


def epochs_to_arrays(epochs_map, class_names, n_channels):
    """把按类别名索引的 epochs 字典转成 (n_trials, n_channels, n_times) 数组元组。

    共享逻辑: 先用第一个非空类别探出采样点数 n_times；空类别或无数据类别
    返回形状 (0, n_channels, n_times) 的空数组；非空数组转 float32。
    """
    n_times = 0
    for ep in epochs_map.values():
        if len(ep) > 0:
            n_times = ep.get_data(copy=False).shape[-1]
            break

    result = []
    for name in class_names:
        ep = epochs_map[name]
        if len(ep) == 0 or n_times == 0:
            arr = np.empty((0, n_channels, n_times))
        else:
            arr = ep.get_data(copy=False).astype(np.float32)
        result.append(arr)

    return tuple(result)


def preprocess_physionetmi_raw(raw, tmin=-1.0, tmax=3.5, reject_amp_uv=None):
    """
    PhysionetMI 原始 EDF 数据完整预处理管线。

    该流程针对“时序流形轨迹”特征工程做了调整：
      1. 64 通道 1-45 Hz 宽带带通滤波（为 ICA 提供足够频宽）
      2. 全脑 64 通道 ICA，自动去除 1-2 个眼动/肌电伪迹成分
      3. 截取 22 个中央区 EEG 通道（与 BCI IV-2a 对应）
      4. 22 通道 8-30 Hz 精细带通滤波（μ/β 节律）
      5. 以任务提示 onset 为 t=0，切分 [-1.0, 3.5]s epoch
         （包含 1s 静息期和 3.5s 想象期，保留静息→想象的过渡动力学）

    参数:
        raw: 已加载并拼接的 MNE Raw，annotation 应包含
             left_hand / right_hand / feet / hands / rest 等
        tmin: epoch 起始时间（秒），默认 -1.0
        tmax: epoch 结束时间（秒），默认  3.5
        reject_amp_uv: 峰峰值幅度阈值（µV），剔除超幅坏 trial；None=不剔除

    返回:
        left_epochs, right_epochs, feet_epochs, hands_epochs
    """
    raw_copy = raw.copy()

    # Step 1: 先截取 22 个目标中央区通道（Physionet 64 通道，其中大量枕叶/额区
    #   通道对 MI 任务无贡献；先降通道再做 ICA，ICA 计算量按通道数立方下降，
    #   22³ vs 64³ ≈ 1/24，显著提速）
    #   ALL_EEG 模式(适配实验)跳过截取, 保留 loader 给的 62 EEG + 2 eog
    raw_sel = raw_copy if _ALL_EEG_MODE else select_channels_physionetmi(raw_copy)

    # Step 2: 22 通道 1-45 Hz 宽带滤波
    picks_eeg = mne.pick_types(raw_sel.info, eeg=True, exclude=[])
    raw_sel.filter(l_freq=1.0, h_freq=45.0, picks=picks_eeg, verbose=False)

    # Step 3: 22 通道 ICA 去噪
    raw_ica, _ = ica_denoise(raw_sel)

    # Step 4: 8-30 Hz 精细滤波
    raw_filt = filter(raw_ica)

    # Step 5: 切分四类 MI epoch（可选幅度剔除）
    return epochs(raw_filt, tmin=tmin, tmax=tmax, reject_amp_uv=reject_amp_uv)


def preprocess_bci2a_raw(raw, tmin=-1.0, tmax=3.5, reject_amp_uv=None):
    """
    BCIC-IV-2a 原始数据完整预处理管线。

    BCIC-IV-2a 原生 22 个 EEG 通道 + 3 个 EOG 通道（加载器已保留 EOG）：
      1. 22 通道 1-45 Hz 宽带带通滤波（EOG 不参与滤波）
      2. 22 通道 ICA，用真实 EOG 通道自动检测
         并去除 1-2 个眼电伪迹成分（对齐官方 desc_2a 的 EOG 回归清洗
         A01-A05；ICA 完成后自动 drop EOG，输出纯 EEG）
      3. 校验并重排 22 个通道顺序
      4. 22 通道 8-30 Hz 精细带通滤波
      5. 以任务提示 onset 为 t=0，切分 [-1.0, 3.5]s epoch

    返回:
        left_epochs, right_epochs, feet_epochs, tongue_epochs
    """
    raw_copy = raw.copy()

    # Step 0: 确保 EOG 通道类型正确（供 ica.find_bads_eog 使用）
    eog_chs = [ch for ch in raw_copy.ch_names if 'EOG' in ch.upper()]
    if eog_chs:
        raw_copy.set_channel_types({ch: 'eog' for ch in eog_chs})

    # Step 1: 22 通道 1-45 Hz 宽带滤波
    picks_eeg = mne.pick_types(raw_copy.info, eeg=True, exclude=[])
    raw_copy.filter(l_freq=1.0, h_freq=45.0, picks=picks_eeg, verbose=False)

    # Step 2: 22 通道 ICA 去噪（真实 EOG 参考检测眼电）
    raw_ica, _ = ica_denoise(raw_copy)

    # Step 3: 校验并重排 22 个目标通道
    raw_sel = select_channels_physionetmi(raw_ica)

    # Step 4: 8-30 Hz 精细滤波
    raw_filt = filter(raw_sel)

    # Step 5: 切分四类 MI epoch（第 4 类为 tongue，可选幅度剔除）
    return epochs(raw_filt, tmin=tmin, tmax=tmax, event_config=_BCI2A_EVENT_CONFIG,
                  reject_amp_uv=reject_amp_uv)