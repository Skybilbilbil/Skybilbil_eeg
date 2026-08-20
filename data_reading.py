# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 15:09:01 2026

@author: Lenovo
"""

import mne
import os
import numpy as np

def read_data (path):
    
    
    #读取文件
    raw = mne.io.read_raw_gdf(
    path,
    preload=True,  # 数据读取
    verbose=False,  # 关闭日志
)
    
    #print(raw.info)
    
    eog_channels = ['EOG-left', 'EOG-right', 'EOG-central']
    channels_to_set = {ch: 'eog' for ch in eog_channels if ch in raw.ch_names}
    if channels_to_set:
        raw.set_channel_types(channels_to_set)
    
    return raw