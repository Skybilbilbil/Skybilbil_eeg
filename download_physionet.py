# -*- coding: utf-8 -*-
"""
PhysioNet EEGMMIDB 数据集下载脚本
用于提前下载所有数据，避免预处理时网络错误
"""

import warnings
import time
import os
from pathlib import Path

# 忽略警告
warnings.filterwarnings("ignore")

import moabb
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

# 与 moabb_loader.py 保持一致：排除 6 名异常被试
EXCLUDED_SUBJECTS = {88, 89, 92, 100, 104, 106}

# 设置日志级别
moabb.set_log_level("info")

print("=" * 70)
print("🚀 PhysioNet EEGMMIDB 数据集下载脚本")
print("=" * 70)
print()

# 实例化数据集：只需要 MI run（executed=False）
print("📦 初始化 PhysionetMI 数据集...")
dataset = PhysionetMI(imagined=True, executed=False)

# 过滤掉 6 名异常被试
subjects_to_download = [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS]

print(f"✅ 发现被试数量: {len(subjects_to_download)} (1-109, 除{sorted(EXCLUDED_SUBJECTS)})")
print(f"📋 被试列表: {subjects_to_download}")
print()

# 实例化范式（用于触发数据下载）
paradigm = MotorImagery(n_classes=4)

print("=" * 70)
print("💡 开始下载数据...")
print("💡 这可能需要较长时间（约 5.5GB），请耐心等待")
print("💡 如果下载中断，重新运行脚本会继续下载剩余数据")
print("=" * 70)
print()

# 逐个下载被试数据（这样可以显示进度）
success_count = 0
failed_subjects = []

for idx, subject in enumerate(subjects_to_download):
    print(f"\n[{idx+1}/{len(subjects_to_download)}] 正在下载被试 {subject:03d}...")
    
    max_retries = 3
    retry_count = 0
    success = False
    
    while retry_count < max_retries and not success:
        try:
            # 尝试获取数据（这会触发下载）
            X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
            
            print(f"    ✅ 下载成功! trials={len(X)}, labels={set(y)}")
            success = True
            success_count += 1
            
            # 短暂暂停，避免服务器压力
            time.sleep(0.5)
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"    ⚠️ 下载失败，重试 {retry_count}/{max_retries}... 错误: {e}")
                time.sleep(2)  # 等待后重试
            else:
                print(f"    ❌ 下载失败 (已重试 {max_retries}次): {e}")
                failed_subjects.append(subject)
    
    # 每 10 个被试打印进度总结
    if (idx + 1) % 10 == 0:
        print(f"\n📊 进度: {idx+1}/{len(subjects_to_download)} 已完成, 成功: {success_count}, 失败: {len(failed_subjects)}")

print("\n" + "=" * 70)
print("🏆 下载完成!")
print("=" * 70)
print(f"✅ 成功下载: {success_count} 个被试")

if failed_subjects:
    print(f"❌ 下载失败: {len(failed_subjects)} 个被试")
    print(f"   失败列表: {failed_subjects}")
    print("\n💡 提示: 可以重新运行脚本尝试下载失败的被试")
else:
    print("✅ 所有被试下载成功!")

print("\n💡 现在可以运行 main.py 或 Data_saving.py 进行预处理了!")
print("=" * 70)