#!/usr/bin/env python3
"""
TacGlove IPC 测试脚本

用于测试 Python 端与 C++ 端之间的共享内存通信
"""

import sys
import time
import numpy as np
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tac_ipc import TacGloveIPCSender, VECTOR_DIM, MISSING_VALUE


def test_sender():
    """测试 IPC 发送器"""
    print("=" * 60)
    print("TacGlove IPC Sender Test")
    print("=" * 60)
    
    sender = TacGloveIPCSender()
    
    if not sender.initialize():
        print("[ERROR] Failed to initialize IPC sender")
        return False
    
    print(f"[OK] IPC sender initialized")
    print(f"     Vector dimension: {VECTOR_DIM}")
    print(f"     Missing value: {MISSING_VALUE}")
    print()
    
    # 发送测试数据
    print("Sending test frames...")
    for i in range(10):
        # 模拟左手数据
        left_data = np.random.rand(VECTOR_DIM).astype(np.float32) * 100
        # 模拟右手数据
        right_data = np.random.rand(VECTOR_DIM).astype(np.float32) * 100
        
        timestamp_ms = int(time.time() * 1000)
        
        if sender.send_dual_frame(left_data, right_data, timestamp_ms):
            print(f"  Frame {i}: sent (left[0]={left_data[0]:.2f}, right[0]={right_data[0]:.2f})")
        else:
            print(f"  Frame {i}: FAILED")
        
        time.sleep(0.1)  # 10 Hz
    
    print()
    print("Sending frames with missing data...")
    
    # 测试缺失数据
    for i in range(3):
        timestamp_ms = int(time.time() * 1000)
        
        # 只有左手数据
        left_data = np.random.rand(VECTOR_DIM).astype(np.float32) * 100
        sender.send_dual_frame(left_data, None, timestamp_ms)
        print(f"  Frame {i}: left only")
        time.sleep(0.1)
    
    for i in range(3):
        timestamp_ms = int(time.time() * 1000)
        
        # 只有右手数据
        right_data = np.random.rand(VECTOR_DIM).astype(np.float32) * 100
        sender.send_dual_frame(None, right_data, timestamp_ms)
        print(f"  Frame {i}: right only")
        time.sleep(0.1)
    
    sender.close()
    print()
    print("[OK] Test complete")
    return True


def continuous_sender(fps=30, duration=None):
    """
    持续发送测试数据
    
    Args:
        fps: 发送频率
        duration: 持续时间（秒），None 表示无限
    """
    print("=" * 60)
    print(f"TacGlove IPC Continuous Sender ({fps} Hz)")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    sender = TacGloveIPCSender()
    
    if not sender.initialize():
        print("[ERROR] Failed to initialize IPC sender")
        return
    
    print("[OK] Sender ready, start sending...")
    
    interval = 1.0 / fps
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            loop_start = time.time()
            
            # 检查持续时间
            if duration is not None and (loop_start - start_time) >= duration:
                break
            
            # 生成模拟数据
            left_data = np.sin(np.arange(VECTOR_DIM) * 0.1 + frame_count * 0.1).astype(np.float32)
            right_data = np.cos(np.arange(VECTOR_DIM) * 0.1 + frame_count * 0.1).astype(np.float32)
            
            timestamp_ms = int(time.time() * 1000)
            sender.send_dual_frame(left_data, right_data, timestamp_ms)
            
            frame_count += 1
            if frame_count % fps == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"  Sent {frame_count} frames, actual FPS: {actual_fps:.1f}")
            
            # 等待下一帧
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
    
    sender.close()
    print(f"[OK] Total frames sent: {frame_count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TacGlove IPC Test")
    parser.add_argument("--mode", choices=["test", "continuous"], default="test",
                        help="Test mode: 'test' for basic test, 'continuous' for streaming")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for continuous mode")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration in seconds for continuous mode (None for infinite)")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        test_sender()
    else:
        continuous_sender(fps=args.fps, duration=args.duration)
