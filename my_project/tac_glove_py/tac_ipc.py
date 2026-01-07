#!/usr/bin/env python3
"""
TacGlove IPC (Inter-Process Communication) Module

This module provides shared memory based IPC for communicating 137-dimensional
tactile glove vectors between Python and C++ processes.

Protocol:
- Uses POSIX shared memory (/dev/shm/tacglove_ipc)
- Uses named semaphores for synchronization
- Ring buffer design for left and right hand data queues

Shared Memory Layout (total: 8 + 2 * (8 + QUEUE_SIZE * FRAME_SIZE) bytes):
    [Header: 8 bytes]
        - magic: uint32 (0x54414347 = "TACG")
        - version: uint16
        - queue_size: uint16
    [Left Hand Queue]
        - write_index: uint32
        - read_index: uint32
        - frames[QUEUE_SIZE]: each frame is (8 + 137*4) = 556 bytes
            - timestamp_ms: int64
            - data[137]: float32 array
    [Right Hand Queue]
        - (same structure as left hand queue)
"""

import os
import struct
import time
import numpy as np
from typing import Optional, Tuple
import mmap
import posix_ipc

# Constants
VECTOR_DIM = 137
QUEUE_SIZE = 64  # Ring buffer size
FRAME_SIZE = 8 + VECTOR_DIM * 4  # timestamp_ms (8) + data (137 * 4)
QUEUE_DATA_SIZE = 8 + QUEUE_SIZE * FRAME_SIZE  # indices (8) + frames
HEADER_SIZE = 8  # magic (4) + version (2) + queue_size (2)
SHM_SIZE = HEADER_SIZE + 2 * QUEUE_DATA_SIZE

SHM_NAME = "/tacglove_ipc"
SEM_LEFT_NAME = "/tacglove_left_sem"
SEM_RIGHT_NAME = "/tacglove_right_sem"
MAGIC = 0x54414347  # "TACG" in ASCII
VERSION = 1
MISSING_VALUE = -1.0


class TacGloveIPCSender:
    """
    Python端的 TacGlove IPC 发送器
    
    用于将 137 维向量数据发送到共享内存，供 C++ 端读取
    """
    
    def __init__(self):
        self._shm: Optional[posix_ipc.SharedMemory] = None
        self._mmap: Optional[mmap.mmap] = None
        self._sem_left: Optional[posix_ipc.Semaphore] = None
        self._sem_right: Optional[posix_ipc.Semaphore] = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """初始化共享内存和信号量"""
        try:
            # 创建或打开共享内存
            try:
                self._shm = posix_ipc.SharedMemory(
                    SHM_NAME, 
                    posix_ipc.O_CREAT | posix_ipc.O_RDWR,
                    size=SHM_SIZE
                )
            except posix_ipc.ExistentialError:
                # 如果已存在，先删除再创建
                posix_ipc.unlink_shared_memory(SHM_NAME)
                self._shm = posix_ipc.SharedMemory(
                    SHM_NAME,
                    posix_ipc.O_CREAT | posix_ipc.O_RDWR,
                    size=SHM_SIZE
                )
            
            # 映射共享内存
            self._mmap = mmap.mmap(self._shm.fd, SHM_SIZE)
            self._shm.close_fd()  # mmap 后可以关闭 fd
            
            # 创建信号量
            try:
                self._sem_left = posix_ipc.Semaphore(
                    SEM_LEFT_NAME, posix_ipc.O_CREAT, initial_value=1
                )
            except posix_ipc.ExistentialError:
                posix_ipc.unlink_semaphore(SEM_LEFT_NAME)
                self._sem_left = posix_ipc.Semaphore(
                    SEM_LEFT_NAME, posix_ipc.O_CREAT, initial_value=1
                )
                
            try:
                self._sem_right = posix_ipc.Semaphore(
                    SEM_RIGHT_NAME, posix_ipc.O_CREAT, initial_value=1
                )
            except posix_ipc.ExistentialError:
                posix_ipc.unlink_semaphore(SEM_RIGHT_NAME)
                self._sem_right = posix_ipc.Semaphore(
                    SEM_RIGHT_NAME, posix_ipc.O_CREAT, initial_value=1
                )
            
            # 初始化共享内存头部
            self._write_header()
            
            # 初始化队列索引
            self._init_queue_indices()
            
            self._initialized = True
            print(f"[TacGloveIPC] Sender initialized: shm={SHM_NAME}, size={SHM_SIZE}")
            return True
            
        except Exception as e:
            print(f"[TacGloveIPC] Failed to initialize sender: {e}")
            self.close()
            return False
    
    def _write_header(self):
        """写入共享内存头部"""
        header = struct.pack('<IHH', MAGIC, VERSION, QUEUE_SIZE)
        self._mmap.seek(0)
        self._mmap.write(header)
        
    def _init_queue_indices(self):
        """初始化队列读写索引为 0"""
        # 左手队列索引
        self._mmap.seek(HEADER_SIZE)
        self._mmap.write(struct.pack('<II', 0, 0))  # write_index, read_index
        
        # 右手队列索引
        self._mmap.seek(HEADER_SIZE + QUEUE_DATA_SIZE)
        self._mmap.write(struct.pack('<II', 0, 0))
        
    def _get_queue_offset(self, is_left: bool) -> int:
        """获取队列在共享内存中的偏移量"""
        if is_left:
            return HEADER_SIZE
        else:
            return HEADER_SIZE + QUEUE_DATA_SIZE
    
    def send_frame(self, data: np.ndarray, is_left: bool, timestamp_ms: Optional[int] = None) -> bool:
        """
        发送一帧数据到共享内存队列
        
        Args:
            data: 137维 numpy 数组
            is_left: True 表示左手，False 表示右手
            timestamp_ms: 时间戳（毫秒），如果为 None 则使用当前时间
            
        Returns:
            是否发送成功
        """
        if not self._initialized:
            return False
            
        if data.shape != (VECTOR_DIM,):
            print(f"[TacGloveIPC] Invalid data shape: {data.shape}, expected ({VECTOR_DIM},)")
            return False
        
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
            
        sem = self._sem_left if is_left else self._sem_right
        queue_offset = self._get_queue_offset(is_left)
        
        try:
            sem.acquire(timeout=0.01)  # 10ms timeout
            try:
                # 读取当前写索引
                self._mmap.seek(queue_offset)
                write_idx, read_idx = struct.unpack('<II', self._mmap.read(8))
                
                # 计算帧在队列中的位置
                frame_offset = queue_offset + 8 + (write_idx % QUEUE_SIZE) * FRAME_SIZE
                
                # 写入帧数据
                self._mmap.seek(frame_offset)
                self._mmap.write(struct.pack('<q', timestamp_ms))  # timestamp
                self._mmap.write(data.astype(np.float32).tobytes())  # data
                
                # 更新写索引
                new_write_idx = write_idx + 1
                self._mmap.seek(queue_offset)
                self._mmap.write(struct.pack('<II', new_write_idx, read_idx))
                
                return True
            finally:
                sem.release()
        except posix_ipc.BusyError:
            # 信号量获取超时
            return False
        except Exception as e:
            print(f"[TacGloveIPC] Error sending frame: {e}")
            return False
    
    def send_dual_frame(self, left_data: Optional[np.ndarray], right_data: Optional[np.ndarray],
                        timestamp_ms: Optional[int] = None) -> bool:
        """
        同时发送左右手数据
        
        Args:
            left_data: 左手 137 维数组，None 表示缺失
            right_data: 右手 137 维数组，None 表示缺失
            timestamp_ms: 共享时间戳
            
        Returns:
            是否全部发送成功
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
            
        success = True
        
        if left_data is not None:
            if not self.send_frame(left_data, is_left=True, timestamp_ms=timestamp_ms):
                success = False
        else:
            # 发送缺失数据
            missing_data = np.full(VECTOR_DIM, MISSING_VALUE, dtype=np.float32)
            if not self.send_frame(missing_data, is_left=True, timestamp_ms=timestamp_ms):
                success = False
                
        if right_data is not None:
            if not self.send_frame(right_data, is_left=False, timestamp_ms=timestamp_ms):
                success = False
        else:
            missing_data = np.full(VECTOR_DIM, MISSING_VALUE, dtype=np.float32)
            if not self.send_frame(missing_data, is_left=False, timestamp_ms=timestamp_ms):
                success = False
                
        return success
    
    def close(self):
        """关闭并清理资源"""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
            
        if self._shm is not None:
            try:
                self._shm.unlink()
            except:
                pass
            self._shm = None
            
        if self._sem_left is not None:
            try:
                self._sem_left.unlink()
            except:
                pass
            self._sem_left = None
            
        if self._sem_right is not None:
            try:
                self._sem_right.unlink()
            except:
                pass
            self._sem_right = None
            
        self._initialized = False
        print("[TacGloveIPC] Sender closed")
        
    def __del__(self):
        self.close()
        
    def __enter__(self):
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# 全局单例实例，方便在 TacDataCollector 中使用
_global_sender: Optional[TacGloveIPCSender] = None

def get_global_sender() -> TacGloveIPCSender:
    """获取全局 IPC 发送器实例"""
    global _global_sender
    if _global_sender is None:
        _global_sender = TacGloveIPCSender()
        _global_sender.initialize()
    return _global_sender

def send_tactile_data(left_data: Optional[np.ndarray], right_data: Optional[np.ndarray],
                      timestamp_ms: Optional[int] = None) -> bool:
    """
    便捷函数：发送触觉数据到 C++ 进程
    
    Args:
        left_data: 左手 137 维数组，None 表示缺失（会填充 -1）
        right_data: 右手 137 维数组，None 表示缺失（会填充 -1）
        timestamp_ms: 时间戳毫秒，None 使用当前时间
        
    Returns:
        是否发送成功
    """
    sender = get_global_sender()
    return sender.send_dual_frame(left_data, right_data, timestamp_ms)

def cleanup_ipc():
    """清理全局 IPC 资源"""
    global _global_sender
    if _global_sender is not None:
        _global_sender.close()
        _global_sender = None


if __name__ == "__main__":
    # 测试代码
    print("Testing TacGlove IPC Sender...")
    
    with TacGloveIPCSender() as sender:
        for i in range(10):
            left = np.random.rand(137).astype(np.float32)
            right = np.random.rand(137).astype(np.float32)
            
            if sender.send_dual_frame(left, right):
                print(f"Sent frame {i}")
            else:
                print(f"Failed to send frame {i}")
                
            time.sleep(0.1)
    
    print("Test complete")
