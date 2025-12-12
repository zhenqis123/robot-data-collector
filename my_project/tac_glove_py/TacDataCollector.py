import os, sys
import time, threading, queue, sys, gc, os, argparse
import warnings, json, termios, pickle
from copy import deepcopy
import numpy as np
from glove_hand import HandType, HandConfig, TacGloveFrame, PacketAssembler, SerialReader, StatsMonitor, GloveVisualizer, frame2finger_and_palm
from tac_utils import flat1d_from_frame
from tac_ipc import TacGloveIPCSender, send_tactile_data, cleanup_ipc

class TacDataCollector:
    def __init__(self, args):
        self.args = args
        TARGET_RENDER_FPS = 2  # 渲染帧率
        RENDER_INTERVAL = 1.0 / TARGET_RENDER_FPS
        SAMPLE_RATE_HZ = 200 # tac glove sample rate
        TOTAL_SAMPLE_RATE = 200 # total sample number to collect
        self.inference_empty_tac = False
        self.glove_mode = "right" if args.mode=="right" else ("left" if args.mode=="left" else "both")
        '''init glove hand'''
        self.selected_hands = []
        if self.glove_mode == 'left':
            self.selected_hands.append(HandType.LEFT)
        elif self.glove_mode == 'right':
            self.selected_hands.append(HandType.RIGHT)
        elif self.glove_mode == 'both':
            self.selected_hands.append(HandType.LEFT)
            self.selected_hands.append(HandType.RIGHT)
        # Create configurations and data structures for selected hands
        self.hand_configs = {hand: HandConfig(hand) for hand in self.selected_hands}
        self.data_queues = {hand: queue.Queue(maxsize=3) for hand in self.selected_hands}
        self.stats_monitors = {hand: StatsMonitor(expect_fps=SAMPLE_RATE_HZ) for hand in self.selected_hands}
        self.readers = [SerialReader(config, self.data_queues[config.hand]) for config in self.hand_configs.values()]
        for r in self.readers:
            r.start()
            time.sleep(0.1) # Stagger start slightly
        if not any(r.connected for r in self.readers):
            print("\n[FATAL] No serial ports could be opened.")
            for r in self.readers: r.stop()
        if self.glove_mode != "none":
            color_limit = 10 if self.inference_empty_tac else 50
            self.glove_viz = GloveVisualizer(self.hand_configs,VIS_FPS=TARGET_RENDER_FPS, TXT_FPS=TARGET_RENDER_FPS, color_limit=color_limit)
        self.latest_frames = {} # Store the most recent frame for each hand
        self.init_glove_tac_frame = {
            HandType.RIGHT:None,
            HandType.LEFT:None
        }
        self.update_init_glove_frame()
        gc.disable()
        
        # 初始化 IPC 发送器（用于与 C++ 进程通信）
        self.ipc_sender = TacGloveIPCSender()
        self.ipc_enabled = self.ipc_sender.initialize()
        if self.ipc_enabled:
            print("[TacDataCollector] IPC sender initialized for C++ communication")
        else:
            print("[TacDataCollector] IPC sender not available (C++ process not running?)")
        
        self.initialize_start()
        # self.save_dir = args.save_dir
    
    def initialize_start(self):
         self.episode_start = True
         self.episode_done = False
        
    def update_init_glove_frame(self):
        print("Update init glove tac frame")
        # global self.init_glove_tac_frame
        for hand, q in self.data_queues.items():
            latest_frame_this_hand = None
            while latest_frame_this_hand is None:
                try:
                    while True:
                        latest_frame_this_hand = q.get_nowait()
                except queue.Empty:
                    self.init_glove_tac_frame[hand] = latest_frame_this_hand
                    
    def main_loop(self):                
        args = self.args
        timestep = 0
        episode = self.args.start_episode
        
        while True:
            # Press S and a new episode record starts
            if self.episode_start and not self.episode_done:
                self.episode_start = False
                print("Start recording episode {}".format(episode))
                timestep = 0
                data = {
                    'timestep': [], 
                    'done': [], 
                }
                data['right_glove_tac_frame'] = []    # list of TacGloveFrame
                data['left_glove_tac_frame'] = []    # list of TacGloveFrame
                    
            # Press D and the episode record ends
            elif self.episode_done:
                self.episode_done = False
                data['done'][-1] = True
                termios.tcflush(sys.stdin, termios.TCIOFLUSH) # flush the input buffer
                print("Saving episode {} data...".format(episode))
                    
                for k in list(data.keys()):
                    data[k] = np.asarray(data[k]) # save other data as numpy arrays
                
                with open(os.path.join(self.save_dir, 'episode_'+str(episode)+'.pkl'), 'wb') as f:
                    pickle.dump(data, f)
                episode += 1
                print("Episode done.")
                del data
                continue
            
            start_time = time.time()
            latest_frames={
                hand:None for hand in self.selected_hands
            }
            first_read = True
            for hand, q in self.data_queues.items():
                if first_read:
                    latest_frame_this_hand = None
                    while latest_frame_this_hand is None or (time.time()-latest_frame_this_hand.timestamp) > 0.004:
                        try: 
                            while True:
                                latest_frame_this_hand = q.get_nowait()
                        except queue.Empty:
                            latest_frames[hand] = latest_frame_this_hand

                    first_read = False
                else:
                    latest_frame_this_hand = None
                    while latest_frame_this_hand is None:
                        try:
                            while True:
                                latest_frame_this_hand = q.get_nowait()
                        except queue.Empty:
                            latest_frames[hand] = latest_frame_this_hand

            for temp_hand_type in self.selected_hands:
                latest_frames[temp_hand_type] -= self.init_glove_tac_frame[temp_hand_type]
            
            right_glove_frame = latest_frames[HandType.RIGHT] if HandType.RIGHT in self.selected_hands else None
            # self.glove_viz.update(right_glove_frame)
            if not self.episode_done and 'data' in locals():
                data['right_glove_tac_frame'].append(deepcopy(right_glove_frame))
                if right_glove_frame is not None:
                    right_glove_137 = flat1d_from_frame(right_glove_frame, hand_type=HandType.RIGHT)  # just for test
                else:
                    right_glove_137 = np.zeros(137) - 1.0  # just for test
            else:
                right_glove_137 = np.zeros(137) - 1.0  # just for test
                    
            left_glove_frame = latest_frames[HandType.LEFT] if HandType.LEFT in self.selected_hands else None
            self.glove_viz.update(left_glove_frame)
            if not self.episode_done and 'data' in locals():
                data['left_glove_tac_frame'].append(deepcopy(left_glove_frame))
                if left_glove_frame is not None:
                    left_glove_137 = flat1d_from_frame(left_glove_frame, hand_type=HandType.LEFT)  # just for test
                else:
                    left_glove_137 = np.zeros(137) - 1.0  # just for test
            else:
                left_glove_137 = np.zeros(137) - 1.0  # just for test

            # 通过 IPC 发送数据到 C++ 进程
            if self.ipc_enabled:
                timestamp_ms = int(time.time() * 1000)
                # 将 numpy 数组转换为 float32 并发送
                left_data = left_glove_137.astype(np.float32) if left_glove_frame is not None else None
                right_data = right_glove_137.astype(np.float32) if right_glove_frame is not None else None
                self.ipc_sender.send_dual_frame(left_data, right_data, timestamp_ms)

            sleep_time = max(args.policy_control_dt - time.time() + start_time, 0)
            if sleep_time == 0:
                warnings.warn("Exceed policy control dt with {}s.".format(time.time() - start_time))
            # print(sleep_time)
            time.sleep(sleep_time)
            timestep += 1
            
        
        
        

if __name__ == "__main__":
    parse = argparse.ArgumentParser() 
    parse.add_argument('--mode', default="left", type=str) # left, right, both
    parse.add_argument('--policy_control_dt', default=0.1, type=float) # policy control frequency
    parse.add_argument('--start_episode', '-s', default=0, type=int) # 中途中断后继续记录
    args = parse.parse_args()
    
    data_collector = TacDataCollector(args)
    data_collector.main_loop()
    
    
    