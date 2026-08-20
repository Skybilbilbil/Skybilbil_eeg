import multiprocessing as mp
import time
import math
import os

def cpu_worker():
    """死循环进行浮点运算，持续占用CPU核心"""
    while True:
        # 不断计算正弦和余弦，制造持续的CPU负载
        for _ in range(1000):
            math.sin(0.5) * math.cos(0.3) + math.tan(0.1)
        # 稍微让出一点点，防止系统卡死（但基本保持满载）
        # time.sleep(0.0001)  # 如果需要降低单核占满时的系统响应，可取消注释

def main():
    print("🚀 即将开始逐步增加CPU负载...")
    print("按 Ctrl+C 随时停止。")
    print("请同时打开任务管理器或 Core Temp 观察温度和占用率。\n")

    processes = []
    target_cores = mp.cpu_count()          # 总核心数
    print(f"检测到 {target_cores} 个逻辑核心，最多将启动 {target_cores} 个进程。\n")

    try:
        for i in range(1, target_cores + 1):
            # 启动一个新进程
            p = mp.Process(target=cpu_worker, daemon=True)
            p.start()
            processes.append(p)
            print(f"[{i}] 已启动 {i} 个CPU负载进程，当前占用率约 {i / target_cores * 100:.0f}%")

            # 每次增加后等待 5 秒，让你观察温度变化
            time.sleep(5)

        print("\n✅ 已启动所有核心，CPU已满载。按 Ctrl+C 停止。")
        # 保持主进程存活，等待用户中断
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 收到停止信号，正在终止所有工作进程...")
        for p in processes:
            p.terminate()
        print("✅ 已清理完毕，退出。")

if __name__ == "__main__":
    # Windows下多进程必须放在 if __name__ == '__main__' 保护下
    main()