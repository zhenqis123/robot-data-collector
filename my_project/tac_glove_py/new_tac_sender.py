import os, sys
import time, threading, queue, sys, gc, os, argparse
import warnings, json, termios, pickle
from copy import deepcopy
import numpy as np
from glove_hand import HandType, HandConfig, TacGloveFrame, PacketAssembler, SerialReader, StatsMonitor, GloveVisualizer, frame2finger_and_palm
from tac_utils import flat1d_from_frame
from tac_ipc import TacGloveIPCSender, send_tactile_data, cleanup_ipc


class DirectQueue:
    """
    Simulates a Queue for SerialReader but handles data directly 
    by updating a latest_frame buffer and sending to IPC immediately.
    """
    def __init__(self, hand_type, collector):
        self.hand_type = hand_type
        self.collector = collector
        self.latest_frame = None
        self.lock = threading.Lock()

    def put_nowait(self, frame):
        # 1. Update latest frame directly
        with self.lock:
            self.latest_frame = frame
        
        # 2. Send to IPC immediately (Direct Sending)
        if self.collector.ipc_enabled:
            self.collector.send_ipc_direct(frame)

    def put(self, item, block=True, timeout=None):
        """Compatibility wrapper for queue.Queue interface"""
        self.put_nowait(item)

    def get_nowait(self):
        # SerialReader only calls this if queue is Full.
        # since put_nowait never raises Full, this is effectively unused
        # but kept for interface compatibility.
        raise queue.Empty

    def get_latest(self):
        with self.lock:
            return self.latest_frame


class NewTacDataCollector:
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
        
        # Use DirectQueue for direct processing/sending instead of queue.Queue
        self.ipc_lock = threading.Lock()

        # 初始化 IPC 发送器（用于与 C++ 进程通信）
        self.ipc_sender = TacGloveIPCSender()
        self.ipc_enabled = self.ipc_sender.initialize()
        if self.ipc_enabled:
            print("[NewTacDataCollector] IPC sender initialized for C++ communication")
        else:
            print("[NewTacDataCollector] IPC sender not available (C++ process not running?)")

        # Initialize this BEFORE starting readers, so send_ipc_direct doesn't crash when accessed by threads
        # self.init_glove_tac_frame = {
        #     HandType.RIGHT: None,
        #     HandType.LEFT: None
        # }

        self.data_queues = {hand: DirectQueue(hand, self) for hand in self.selected_hands}
        
        self.stats_monitors = {hand: StatsMonitor(expect_fps=SAMPLE_RATE_HZ) for hand in self.selected_hands}
        self.readers = [SerialReader(config, self.data_queues[config.hand]) for config in self.hand_configs.values()]
        for r in self.readers:
            r.start()
            time.sleep(0.1) # Stagger start slightly
        if not any(r.connected for r in self.readers):
            print("\n[FATAL] No serial ports could be opened.")
            for r in self.readers: r.stop()
        # if self.glove_mode != "none":
        #     color_limit = 10 if self.inference_empty_tac else 50
        #     self.glove_viz = GloveVisualizer(self.hand_configs,VIS_FPS=TARGET_RENDER_FPS, TXT_FPS=TARGET_RENDER_FPS, color_limit=color_limit)
        self.latest_frames = {} # Store the most recent frame for each hand
        # self.init_glove_tac_frame was initialized above
        # self.update_init_glove_frame()
        gc.disable()
        
        self.initialize_start()
        self.save_dir = args.save_dir if hasattr(args, 'save_dir') else "." # Fix potential error
    
    def initialize_start(self):
         self.episode_start = True
         self.episode_done = False
        
    def update_init_glove_frame(self):
        print("Update init glove tac frame (DISABLED)")
        # for hand, q in self.data_queues.items():
        #     latest_frame_this_hand = None
        #     # Wait until we get at least one frame
        #     while latest_frame_this_hand is None:
        #         latest_frame_this_hand = q.get_latest()
        #         if latest_frame_this_hand is None:
        #             time.sleep(0.01)
        #     self.init_glove_tac_frame[hand] = latest_frame_this_hand
                    
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
            # Direct access to the latest frame (no queue draining)
            for hand, q in self.data_queues.items():
                latest_frames[hand] = q.get_latest()

            for temp_hand_type in self.selected_hands:
                if latest_frames.get(temp_hand_type) is not None:
                     # latest_frames[temp_hand_type] -= self.init_glove_tac_frame[temp_hand_type]
                     pass
            
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
            # self.glove_viz.update(left_glove_frame)
            if not self.episode_done and 'data' in locals():
                data['left_glove_tac_frame'].append(deepcopy(left_glove_frame))
                if left_glove_frame is not None:
                    left_glove_137 = flat1d_from_frame(left_glove_frame, hand_type=HandType.LEFT)  # just for test
                else:
                    left_glove_137 = np.zeros(137) - 1.0  # just for test
            else:
                left_glove_137 = np.zeros(137) - 1.0  # just for test

            # IPC Sending is now handled directly in DirectQueue.put_nowait
            # The following block is removed / commented out

            sleep_time = max(args.policy_control_dt - time.time() + start_time, 0)
            if sleep_time == 0:
                warnings.warn("Exceed policy control dt with {}s.".format(time.time() - start_time))
            # print(sleep_time)
            time.sleep(sleep_time)
            timestep += 1
            
    def send_ipc_direct(self, frame):
        """
        Subprocess logic moved from Main Loop:
        Process frame and send directly to IPC
        """
        # init_frame = self.init_glove_tac_frame.get(frame.hand)
        # if init_frame is None:
        #     return

        try:
            # Subtract init frame
            # Use deepcopy and -= to ensure thread safety and compatibility (if only __isub__ is defined)
            processed_frame = deepcopy(frame)
            # processed_frame -= init_frame
            
            # Convert to 137d array
            data_137 = flat1d_from_frame(processed_frame, hand_type=frame.hand)
            
            # Send to IPC (thread-safe)
            # Use frame.timestamp for precision
            timestamp_ms = int(frame.timestamp * 1000) 
            
            with self.ipc_lock:
                self.ipc_sender.send_frame(
                    data_137.astype(np.float32), 
                    is_left=(frame.hand == HandType.LEFT),
                    timestamp_ms=timestamp_ms
                )
        except Exception as e:
            print(f"IPC Send Error: {e}")
            import traceback
            traceback.print_exc()
        

if __name__ == "__main__":
    parse = argparse.ArgumentParser() 
    parse.add_argument('--mode', default="left", type=str) # left, right, both
    parse.add_argument('--policy_control_dt', default=0.005, type=float) # policy control frequency
    parse.add_argument('--start_episode', '-s', default=0, type=int) # 中途中断后继续记录
    args = parse.parse_args()
    
    data_collector = NewTacDataCollector(args)
    data_collector.main_loop()