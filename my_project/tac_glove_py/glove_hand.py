import serial, time, threading, queue, sys, gc, os, argparse
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Deque, List, Tuple, Any
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

PORT  = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port1"
# #on titan
# LEFT_PORT  = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:14:1.0-port0"
# RIGHT_PORT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:6:1.0-port0"

# on 4090
LEFT_PORT  = "/dev/serial/by-path/pci-0000:14:00.3-usb-0:2:1.0"
LEFT_PORT  = "/dev/serial/by-path/pci-0000:11:00.0-usb-0:3:1.0"
LEFT_PORT  = "/dev/serial/by-path/pci-0000:11:00.0-usb-0:3:1.0"
LEFT_PORT  = "/dev/serial/by-path/pci-0000:0e:00.0-usb-0:3:1.0"
# LEFT_PORT = "/dev/serial/by-path/pci-0000:11:00.0-usb-0:3:1.0"

RIGHT_PORT = "/dev/serial/by-path/pci-0000:0e:00.0-usb-0:2:1.0"
RIGHT_PORT = "/dev/serial/by-path/pci-0000:0e:00.0-usb-0:2:1.0"
RIGHT_PORT = "/dev/ttyACM0"
# RIGHT_PORT = "/dev/serial/by-path/pci-0000:11:00.0-usb-0:3:1.0"
RIGHT_PORT = "/dev/serial/by-path/pci-0000:11:00.0-usb-0:3:1.0"
RIGHT_PORT = "/dev/serial/by-path/pci-0000:0e:00.0-usb-0:3:1.0"

# BAUD = 921_600
# BAUD = 460_800	
BAUD = 3_000_000
FRAME_HEADER   = b'\xAA\x55\x03\x99'
HEADER_LEN     = 4
SEQ_LEN, TYPE_LEN = 1, 1
DATA_LEN_1, DATA_LEN_2 = 128, 144
PKT_LEN_1, PKT_LEN_2   = SEQ_LEN + TYPE_LEN + DATA_LEN_1, SEQ_LEN + TYPE_LEN + DATA_LEN_2

SAMPLE_RATE_HZ  = 300
SAMPLE_PERIOD_S = 1.0 / SAMPLE_RATE_HZ

# --- HandType-Specific Sensor ID Mappings ---


THUMB_P_RIGHT = [240, 239, 238, 256, 255, 254, 16, 15, 14, 32, 31, 30]
INDEX_P_RIGHT = [237, 236, 235, 253, 252, 251, 13, 12, 11, 29, 28, 27]
MIDDLE_P_RIGHT = [234, 233, 232, 250, 249, 248, 10, 9, 8, 26, 25, 24]
RING_P_RIGHT = [231, 230, 229, 247, 246, 245, 7, 6, 5, 23, 22, 21]
PINKY_P_RIGHT = [228, 227, 226, 244, 243, 242, 4, 3, 2, 20, 19, 18]
BEND_ID_RIGHT = [47, 44, 41, 38, 35]
PALM_P_RIGHT = [
    61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50,
    80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66,
    96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82,
    112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98,
    128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114,
]


THUMB_P_LEFT  = [19, 18, 17, 3, 2, 1, 243, 242, 241, 227, 226, 225]
INDEX_P_LEFT  = [22, 21, 20, 6, 5, 4, 246, 245, 244, 230, 229, 228]
MIDDLE_P_LEFT = [25, 24, 23, 9, 8, 7, 249, 248, 247, 233, 232, 231]
RING_P_LEFT   = [28, 27, 26, 12, 11, 10, 252, 251, 250, 236,235, 234]
PINKY_P_LEFT  = [31, 30, 29, 15, 14, 13, 255, 254, 253, 239,238, 237]
BEND_ID_LEFT  = [210, 213, 216, 219, 222]
PALM_P_LEFT = [
207,206,205,204,203,202,201,200,199,198,197,196,
191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,
175,174,173,172,171,170,169,168,167,166,165,164,163,162,161,
159,158,157,156,155,154,153,152,151,150,149,148,147,146,145,
143,142,141,140,139,138,137,136,135,134,133,132,131,130,129,
]

LEFT_ORDER = THUMB_P_LEFT + INDEX_P_LEFT + MIDDLE_P_LEFT + RING_P_LEFT + PINKY_P_LEFT + BEND_ID_LEFT + PALM_P_LEFT

# --- Modular Classes ---





class HandType(Enum):
    """Enumeration for left and right hands."""
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    RIGHT1 = "Right"
    LEFT1 = "Left"

    def __eq__(self, other):
        # 如果比较的是同一个枚举类型，使用默认比较
        if isinstance(other, HandType):
            return super().__eq__(other)
        
        # 如果比较的是第一个 HandType 枚举的实例
        try:
            # 检查是否是第一个 HandType 类型
            if hasattr(other, 'name') and hasattr(other, 'value'):
                # 将第一个枚举的名称与第二个枚举的值进行比较（忽略大小写）
                return self.value.lower() == other.name.lower()
        except:
            pass
        
        # 其他情况返回 False
        return False
    
    def __hash__(self):
        # 确保哈希值与默认实现一致
        return hash(self.value)

    def upper(self):
        return self.name.upper()
    
    def lower(self):
        return self.name.lower()

class HandConfig:
    """Encapsulates all configuration and sensor mapping for a single hand."""
    def __init__(self, hand: HandType, 
                 left_port: str = LEFT_PORT, 
                 right_port: str = RIGHT_PORT):
        self.hand = hand
        # print(hand) #HandType.LEFT
        # print(hand == HandType.LEFT) #False
        # print(f"hand: {hand}")
        # print(f"HandType.LEFT: {HandType.LEFT}")
        # print(f"Type of hand: {type(hand)}")
        # print(f"Type of HandType.LEFT: {type(HandType.LEFT)}")
        # print(f"hand == HandType.LEFT: {hand == HandType.LEFT}")
        # print(f"hand is HandType.LEFT: {hand is HandType.LEFT}")
        # print(f"hand.value: {hand.value}")
        # print(f"HandType.LEFT.value: {HandType.LEFT.value}")
        if hand == HandType.LEFT:
            self.port = left_port
            self.finger_press_ids = [THUMB_P_LEFT, INDEX_P_LEFT, MIDDLE_P_LEFT, RING_P_LEFT, PINKY_P_LEFT]
            self.bend_ids = BEND_ID_LEFT
            self.palm_ids = PALM_P_LEFT
        elif hand == HandType.RIGHT:
            self.port = right_port
            self.finger_press_ids = [THUMB_P_RIGHT, INDEX_P_RIGHT, MIDDLE_P_RIGHT, RING_P_RIGHT, PINKY_P_RIGHT]
            self.bend_ids = BEND_ID_RIGHT
            self.palm_ids = PALM_P_RIGHT
        else:
            raise ValueError("Invalid hand type specified.")

        all_ids = []
        for ids in self.finger_press_ids: all_ids.extend(ids)
        all_ids.extend(self.bend_ids)
        all_ids.extend(self.palm_ids)
        
        # Filter out placeholder IDs (-1) before creating the map
        valid_ids = [sid for sid in all_ids if sid > 0]
        self.id2pos = {sid: sid - 1 for sid in valid_ids}

        # Pre-calculate visualization matrices
        self.id_mats_fingers = self._create_finger_id_mats()
        self.id_mat_palm = self._create_palm_id_mat()
    
    def _create_finger_id_mats(self) -> List[np.ndarray]:
        """Creates a list of 2D numpy arrays for finger sensor IDs."""
        mats = []
        for i, finger_ids in enumerate(self.finger_press_ids):
            m = np.full((5, 3), np.nan)
            for r in range(4):
                for c in range(3):
                    sid = finger_ids[r * 3 + c]
                    m[r, c] = sid if sid > 0 else np.nan
            bend_sid = self.bend_ids[i]
            m[4, 1] = bend_sid if bend_sid > 0 else np.nan
            mats.append(m)
        return mats

    def _create_palm_id_mat(self) -> np.ndarray:
        """Creates a 2D numpy array for palm sensor IDs."""
        m = np.full((5, 15), np.nan)
        # Assuming the structure of 12 + 4*15 for the palm
        if self.hand == HandType.RIGHT:
            init_offset = 3
            for c in range(12):
                if c < len(self.palm_ids):
                    sid = self.palm_ids[c]
                    m[0, c+init_offset] = sid if sid > 0 else np.nan
        else:
            for c in range(12):
                if c < len(self.palm_ids):
                    sid = self.palm_ids[c]
                    m[0, c] = sid if sid > 0 else np.nan
        
        off = 12
        for r in range(1, 5):
            for c in range(15):
                idx = off + (r - 1) * 15 + c
                if idx < len(self.palm_ids):
                    sid = self.palm_ids[idx]
                    m[r, c] = sid if sid > 0 else np.nan
        return m
    
    def extract_finger_val_mat(self, finger_idx: int, all_vals: np.ndarray) -> np.ndarray:
        """Extracts a matrix of values for a specific finger."""
        id_mat = self.id_mats_fingers[finger_idx]
        val_mat = np.full_like(id_mat, np.nan, dtype=float)
        
        for i in range(id_mat.shape[0]):
            for j in range(id_mat.shape[1]):
                sid = id_mat[i, j]
                if not np.isnan(sid) and int(sid) in self.id2pos:
                    val_mat[i, j] = all_vals[self.id2pos[int(sid)]]
        return val_mat

    def extract_palm_val_mat(self, all_vals: np.ndarray) -> np.ndarray:
        """Extracts a matrix of values for the palm."""
        id_mat = self.id_mat_palm
        val_mat = np.full_like(id_mat, np.nan, dtype=float)
        
        for i in range(id_mat.shape[0]):
            for j in range(id_mat.shape[1]):
                sid = id_mat[i, j]
                if not np.isnan(sid) and int(sid) in self.id2pos:
                    val_mat[i, j] = all_vals[self.id2pos[int(sid)]]
        return val_mat

@dataclass
class TacGloveFrame:
    """Represents a single, complete data frame from a glove."""
    hand: HandType
    sensor_type: int
    data: bytes
    build_time: float
    timestamp: float

    def __init__(self, hand: HandType, sensor_type: int, data: bytes, build_time: float = 0.0, timestamp: float = None):
        self.hand = hand
        self.sensor_type = sensor_type
        self.data = data
        self.build_time = build_time
        self.timestamp = timestamp if timestamp is not None else time.time()

    def __sub__(self, other):
        """
        实现 TacGloveFrame 的减法操作 (self - other)。

        - 检查左右手属性是否一致。
        - 对传感器数据进行逐元素相减。
        - [新逻辑] 将所有负数结果置为 0。
        - 返回一个数据格式与原始 Frame 完全相同的新的 TacGloveFrame 实例。
        """
        if not isinstance(other, TacGloveFrame):
            return NotImplemented

        if self.hand != other.hand:
            raise ValueError(f"无法对不同手型的Frame进行相减: {self.hand.value} vs {other.hand.value}")

        # 1. 为了避免 uint8 直接相减导致的回绕错误，先转换为范围更大的有符号整数
        data_A = np.frombuffer(self.data[:256], dtype=np.uint8).astype(np.int32)
        data_B = np.frombuffer(other.data[:256], dtype=np.uint8).astype(np.int32)

        # 2. 执行相减
        result_data_np = data_A - data_B

        # 3. [核心修改] 使用 clip 方法将数组中的所有负值变为 0
        result_data_np = result_data_np.clip(min=0)
        # 或者使用 np.maximum(0, result_data_np) 效果相同

        # 4. 因为所有值都已 >= 0，并且不会超过255，可以安全地转换回 uint8
        final_data_np = result_data_np.astype(np.uint8)

        # 5. 将结果转回 bytes
        result_data_bytes = final_data_np.tobytes()

        # 6. 创建并返回新的 TacGloveFrame
        return TacGloveFrame(
            hand=self.hand,
            sensor_type=self.sensor_type,
            data=result_data_bytes,
            build_time=self.build_time,
            timestamp=self.timestamp
        )
def frame2finger_and_palm(frame: TacGloveFrame, hand_type: HandType) -> Tuple[np.ndarray, np.ndarray]:
    selected_hands = [HandType.LEFT, HandType.RIGHT]
    hand_configs = {hand: HandConfig(hand) for hand in selected_hands}
    config = hand_configs[hand_type]
    # Data is 256 uint8 for sensors, then other data (e.g., IMU)
    sensor_values = np.frombuffer(frame.data[:256], dtype=np.uint8)

    # 1. Calculate and store all value matrices first.
    finger_val_mats = [config.extract_finger_val_mat(i, sensor_values) for i in range(5)]
    # print(finger_val_mats)
    palm_val_mat = config.extract_palm_val_mat(sensor_values)

    return finger_val_mats, palm_val_mat

class PacketAssembler:
    """Assembles two-part packets into a single TacGloveFrame."""
    def __init__(self, hand: HandType):
        self.hand = hand
        self.buffers: Dict[int, Dict[str, Optional[bytes]]] = defaultdict(
            lambda: {"01": None, "02": None, "t0": 0.0}
        )

    def feed_packet(self, seq_hex: str, stype: int, payload: bytes) -> Optional[TacGloveFrame]:
        buf = self.buffers[stype]
        if seq_hex == "01":
            buf["t0"] = time.time()
        buf[seq_hex] = payload[2:]  # Payload contains seq+type, skip them
        
        if buf["01"] and buf["02"]:
            frame = TacGloveFrame(
                hand=self.hand,
                sensor_type=stype,
                data=buf["01"] + buf["02"],
                build_time=time.time() - buf["t0"],
                timestamp=time.time(),
            )
            self.buffers[stype] = {"01": None, "02": None, "t0": 0.0}
            return frame
        return None

class SerialReader(threading.Thread):
    """Reads from a serial port, assembles packets, and puts frames in a queue."""
    READ_SZ = 4096

    def __init__(self, config: HandConfig, out_q: queue.Queue):
        super().__init__(daemon=True, name=f"{config.hand.value}Reader")
        self.ser = serial.Serial()
        self.ser.port = config.port
        self.ser.baudrate = BAUD
        self.ser.timeout = 0.01
        
        self.q = out_q
        self.asm = PacketAssembler(config.hand)
        self.buf = bytearray()
        self.running = True
        self.connected = False

    def stop(self):
        self.running = False
        if self.is_alive():
            self.join()
        if self.ser.is_open:
            self.ser.close()

    def run(self):
        try:
            self.ser.open()
            self.connected = True
            print(f"[INFO] Successfully opened port for {self.asm.hand.value} hand: {self.ser.port}")
            try:
                self.ser.set_buffer_size(rx_size=64_000)
            except Exception:
                pass
        except serial.SerialException as e:
            print(f"[ERROR] Could not open port for {self.asm.hand.value} hand: {self.ser.port}. {e}")
            return
            
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    self.buf += self.ser.read(self.ser.in_waiting)
                else:
                    time.sleep(0.0001) # Avoid busy-waiting
            except (serial.SerialException, OSError) as e:
                print(f"[ERROR] Serial port error on {self.asm.hand.value} hand: {e}. Thread stopping.")
                self.running = False
                break

            for frame in self._extract_frames():
                if frame:
                    try:
                        self.q.put_nowait(frame)
                    except queue.Full:
                        # Drop oldest frame to make space
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                        self.q.put_nowait(frame)

            # Prevent buffer from growing indefinitely
            if len(self.buf) > 100_000:
                self.buf = self.buf[-50_000:]
        
        if self.ser.is_open:
            self.ser.close()
        print(f"[INFO] {self.asm.hand.value} reader thread finished.")


    def _extract_frames(self):
        pos = 0
        while True:
            hdr_idx = self.buf.find(FRAME_HEADER, pos)
            # print(hdr_idx)
            if hdr_idx == -1: 
                break
            # print("FOUND")
            if hdr_idx + HEADER_LEN + 2 > len(self.buf): break
            
            seq = self.buf[hdr_idx + HEADER_LEN]
            stype = self.buf[hdr_idx + HEADER_LEN + 1]
            
            if seq == 0x01: pkt_len = PKT_LEN_1
            elif seq == 0x02: pkt_len = PKT_LEN_2
            else:
                pos = hdr_idx + 1 # Invalid seq, advance past header start
                continue

            if hdr_idx + HEADER_LEN + pkt_len > len(self.buf): break

            payload = bytes(self.buf[hdr_idx + HEADER_LEN : hdr_idx + HEADER_LEN + pkt_len])
            frame = self.asm.feed_packet(f"{seq:02x}", stype, payload)
            if frame:
                yield frame
            
            pos = hdr_idx + HEADER_LEN + pkt_len

        if pos > 0:
            del self.buf[:pos]

class StatsMonitor:
    """Calculates FPS, build time, and packet loss for a data stream."""
    def __init__(self, expect_fps=SAMPLE_RATE_HZ):
        self.expect_period = 1.0 / expect_fps if expect_fps > 0 else 0
        self.stats_per_type: Dict[int, deque] = defaultdict(lambda: deque(maxlen=200))
        self.drops_per_type: Dict[int, int] = defaultdict(int)
        self.last_ts_per_type: Dict[int, float] = {}

    def update(self, frame: TacGloveFrame):
        stype = frame.sensor_type
        self.stats_per_type[stype].append(frame)
        
        if self.expect_period > 0 and stype in self.last_ts_per_type:
            gap = frame.timestamp - self.last_ts_per_type[stype]
            if gap > self.expect_period * 1.5:
                self.drops_per_type[stype] += int(round(gap / self.expect_period)) - 1
        self.last_ts_per_type[stype] = frame.timestamp

    def get_stats_text(self, stype: int) -> Tuple[str, str]:
        window = self.stats_per_type[stype]
        if not window: return "N/A", "black"

        # FPS
        fps = (len(window) - 1) / (window[-1].timestamp - window[0].timestamp) if len(window) > 1 else 0.0
        # Build time
        avg_build_ms = 1e3 * np.mean([f.build_time for f in window])
        # Loss rate
        total = len(window) + self.drops_per_type[stype]
        loss_rate = self.drops_per_type[stype] / total if total > 0 else 0.0

        text = (f'Type {stype}\n'
                f'FPS: {fps:5.1f}\n'
                f'Build: {avg_build_ms:5.1f} ms\n'
                f'Loss: {loss_rate*100:4.1f}%')
        
        color = 'red' if fps < (0.8 * SAMPLE_RATE_HZ) or loss_rate > 0.10 else 'black'
        return text, color

# ----------------------------- Visualization ----------------------------- #

# class GloveVisualizer:

#     def __init__(self, configs: Dict[HandType, HandConfig], VIS_FPS: int  = 10, TXT_FPS: int = 10, color_limit = 255):
#         self.configs = configs
#         self.color_limit = color_limit  # Limit for sensor value visualization

#         self.hands = list(configs.keys())
#         self.num_hands = len(self.hands)

#         self.VIS_FPS = VIS_FPS  # Visuals refresh rate (Hz)
#         self.TXT_FPS = TXT_FPS   # Text labels refresh rate (Hz)

#         plt.ion()
#         self.fig = plt.figure(figsize=(8 * self.num_hands, 8))
#         self.fig.suptitle("Tactile Glove Visualization", fontsize=16)

#         self._setup_plots()
#         self.last_draw_time = 0.0
#         self.last_text_update_time = 0.0
    
#     def _setup_plots(self):
#         """Creates the matplotlib axes and artists for the required hands."""
#         self.plots = {}
        
#         # GridSpec: 3 rows, 5*num_hands columns + space for colorbar
#         num_cols = 5 * self.num_hands + max(0, self.num_hands - 1)
#         gs = self.fig.add_gridspec(3, num_cols, height_ratios=[1, 0.2, 2], wspace=0.5, hspace=0.4)
        
#         finger_titles = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

#         for i, hand in enumerate(self.hands):
#             col_offset = i * (5 + int(self.num_hands > 1)) # 5 cols for plots + 1 for spacing
            
#             hand_plots = {'ims': [], 'txts': [], 'stats_txt': None, 'im_palm': None, 'txt_palm': None}

#             # Finger plots
#             for col, title in enumerate(finger_titles):
#                 ax = self.fig.add_subplot(gs[0, col_offset + col])
#                 im = ax.imshow(np.full((5, 3), np.nan), vmin=-self.color_limit, vmax=self.color_limit, cmap='viridis')
#                 ax.set_xticks([]); ax.set_yticks([])
#                 ax.set_title(f"{hand.value} {title}", fontsize=10)
#                 txt = [[ax.text(j, i, '', ha='center', va='center', fontsize=7, color='white') for j in range(3)] for i in range(5)]
#                 hand_plots['ims'].append(im); hand_plots['txts'].append(txt)
            
#             # Palm plot
#             ax_palm = self.fig.add_subplot(gs[2, col_offset:col_offset+5])
#             im_palm = ax_palm.imshow(np.full((5, 15), np.nan), vmin=-self.color_limit, vmax=self.color_limit, cmap='viridis')
#             ax_palm.set_xticks([]); ax_palm.set_yticks([])
#             txt_palm = [[ax_palm.text(j, i, '', ha='center', va='center', fontsize=7, color='white') for j in range(15)] for i in range(5)]
#             hand_plots['im_palm'] = im_palm
#             hand_plots['txt_palm'] = txt_palm

#             # Stats text
#             stats_ax = self.fig.add_subplot(gs[0:3, col_offset:col_offset+5])
#             stats_ax.axis('off')
#             hand_plots['stats_txt'] = stats_ax.text(0.5, 0.45, '', va='top', ha='center', fontsize=9,
#                                         bbox=dict(facecolor='white', alpha=0.7, pad=4))

#             self.plots[hand] = hand_plots

#         # Shared Colorbar
#         cax = self.fig.add_axes([0.92, 0.15, 0.015, 0.7])
#         self.fig.colorbar(self.plots[self.hands[0]]['im_palm'], cax=cax, label='Sensor Value')
        
#         self.fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.05)


    # def update(self, frame, stats_monitor, finger_mats = None, palm_mat = None, vis_monitor=True, hand_type=None):
    #     now = time.time()
    #     if now - self.last_draw_time < 1.0 / self.VIS_FPS:
    #         return
    #     self.last_draw_time = now

    #     if frame is None:
    #         assert not vis_monitor
    #         assert hand_type is not None
    #         assert finger_mats is not None
    #         assert palm_mat is not None

    #         hand = hand_type
    #         finger_val_mats = finger_mats
    #         palm_val_mat = palm_mat
    #         plots = self.plots[hand_type]


    #     else:
    #         assert (hand_type is None) or (hand_type == HandType(frame.hand))
    #         assert finger_mats is None
    #         assert palm_mat is None

    #         hand, stype = frame.hand, frame.sensor_type
    #         config = self.configs[hand]
    #         plots = self.plots[hand]
        
    #     # Data is 256 uint8 for sensors, then other data (e.g., IMU)
    #         sensor_values = np.frombuffer(frame.data[:256], dtype=np.uint8)
    #         finger_val_mats = [config.extract_finger_val_mat(i, sensor_values) for i in range(5)]
    #         palm_val_mat = config.extract_palm_val_mat(sensor_values)


    #     # 2. Update image data using the calculated matrices.
    #     for i in range(5):
    #         plots['ims'][i].set_data(finger_val_mats[i])
    #     # print("im_palm:",palm_val_mat.shape)
    #     if isinstance(palm_val_mat, list):
    #         palm_val_mat = palm_val_mat[0]
    #     plots['im_palm'].set_data(palm_val_mat)

    #     # Rate-limit text updates
    #     if now - self.last_text_update_time > 1.0 / self.TXT_FPS:
    #         self.last_text_update_time = now
            
    #         # 3. Update text labels using the *original* calculated matrices.
    #         for i in range(5):
    #             val_mat = finger_val_mats[i] # Use the stored, original matrix

    #             for r in range(5):
    #                 for c in range(3):
    #                     txt = f'{int(val_mat[r,c]):d}' if not np.isnan(val_mat[r, c]) else ''
    #                     plots['txts'][i][r][c].set_text(txt)

    #         # Update palm text labels
    #         for r in range(5):
    #             for c in range(15):
    #                 txt = f'{int(palm_val_mat[r,c]):d}' if not np.isnan(palm_val_mat[r, c]) else ''
    #                 plots['txt_palm'][r][c].set_text(txt)
            
    #         # Update stats text
    #         if vis_monitor:
    #             stats_text, color = stats_monitor.get_stats_text(stype)
    #             plots['stats_txt'].set_text(f"{hand.value} HandType\n" + stats_text)
    #             plots['stats_txt'].set_color(color)

    #     self.fig.canvas.draw_idle()
    #     self.fig.canvas.flush_events()
#     def close(self):
#         plt.close(self.fig)

# class GloveVisualizer:
#     def __init__(self, configs: Dict[HandType, HandConfig], VIS_FPS: int = 20, TXT_FPS: int = 10, color_limit=255):
#         self.configs = configs
#         self.color_limit = color_limit

#         self.hands = list(configs.keys())
#         self.num_hands = len(self.hands)

#         # 检查是否为双手模式，如果不是，同步逻辑无意义
#         # if not (HandType.LEFT in self.hands and HandType.RIGHT in self.hands):
#         #     raise ValueError("GloveVisualizer in sync mode requires configs for both LEFT and RIGHT hands.")

#         self.VIS_FPS = VIS_FPS
#         self.TXT_FPS = TXT_FPS

#         # --- 新增：用于同步的暂存区和线程锁 ---
#         self.pending_data = {}
#         self.hand_data_lock = threading.Lock()

#         plt.ion()
#         self.fig = plt.figure(figsize=(8 * self.num_hands, 8))
#         self.fig.suptitle("Tactile Glove Visualization (Synchronized)", fontsize=16)

#         self._setup_plots()
#         self.last_draw_time = 0.0
#         self.last_text_update_time = 0.0

#     def _setup_plots(self):
#         """Creates the matplotlib axes and artists for the required hands."""
#         self.plots = {}
        
#         num_cols = 5 * self.num_hands + max(0, self.num_hands - 1)
#         gs = self.fig.add_gridspec(3, num_cols, height_ratios=[1, 0.2, 2], wspace=0.5, hspace=0.4)
        
#         finger_titles = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

#         for i, hand in enumerate(self.hands):
#             col_offset = i * (5 + int(self.num_hands > 1))
            
#             hand_plots = {'ims': [], 'txts': [], 'stats_txt': None, 'im_palm': None, 'txt_palm': None}

#             # Finger plots
#             for col, title in enumerate(finger_titles):
#                 ax = self.fig.add_subplot(gs[0, col_offset + col])
#                 im = ax.imshow(np.full((5, 3), np.nan), vmin=0, vmax=self.color_limit, cmap='viridis') # 传感器值通常为正
#                 ax.set_xticks([]); ax.set_yticks([])
#                 ax.set_title(f"{hand.value.capitalize()} {title}", fontsize=10)
#                 txt = [[ax.text(j, i, '', ha='center', va='center', fontsize=7, color='white') for j in range(3)] for i in range(5)]
#                 hand_plots['ims'].append(im); hand_plots['txts'].append(txt)
            
#             # Palm plot
#             ax_palm = self.fig.add_subplot(gs[2, col_offset:col_offset+5])
#             im_palm = ax_palm.imshow(np.full((5, 15), np.nan), vmin=0, vmax=self.color_limit, cmap='viridis')
#             ax_palm.set_xticks([]); ax_palm.set_yticks([])
#             txt_palm = [[ax_palm.text(j, i, '', ha='center', va='center', fontsize=7, color='white') for j in range(15)] for i in range(5)]
#             hand_plots['im_palm'] = im_palm
#             hand_plots['txt_palm'] = txt_palm

#             # Stats text
#             stats_ax = self.fig.add_subplot(gs[0:3, col_offset:col_offset+5])
#             stats_ax.axis('off')
#             hand_plots['stats_txt'] = stats_ax.text(0.5, 0.45, '', va='top', ha='center', fontsize=9,
#                                         bbox=dict(facecolor='white', alpha=0.7, pad=4))

#             self.plots[hand] = hand_plots

#         cax = self.fig.add_axes([0.92, 0.15, 0.015, 0.7])
#         self.fig.colorbar(self.plots[self.hands[0]]['im_palm'], cax=cax, label='Sensor Value')
#         self.fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.05)


#     def update(self, frame: TacGloveFrame, stats_monitor):
#         """
#         接收单只手的数据，暂存起来。当两只手的数据都到达后，触发一次同步绘图。
#         """
#         hand = HandType(frame.hand)
#         config = self.configs[hand]
        
#         # 1. 提取和处理数据
#         sensor_values = np.frombuffer(frame.data[:256], dtype=np.uint8)
#         finger_val_mats = [config.extract_finger_val_mat(i, sensor_values) for i in range(5)]
#         palm_val_mat = config.extract_palm_val_mat(sensor_values)
#         stats_text, color = stats_monitor.get_stats_text(frame.sensor_type)
        
#         # 将处理好的数据打包
#         processed_data = {
#             'finger_mats': finger_val_mats,
#             'palm_mat': palm_val_mat,
#             'stats_text': f"{hand.value.capitalize()} Hand\n" + stats_text,
#             'stats_color': color
#         }
        
#         # 2. 使用锁安全地更新暂存区
#         with self.hand_data_lock:
#             self.pending_data[hand] = processed_data
            
#             # 3. 检查是否两只手的数据都已准备好
#             if HandType.LEFT in self.pending_data and HandType.RIGHT in self.pending_data:
#                 # 满足条件，执行绘图并清空暂存区
#                 self._draw_hands()
#                 self.pending_data.clear()

#     def _draw_hands(self):
#         """
#         (私有方法) 当左右手数据都准备好后，执行此方法进行同步绘制。
#         """
#         now = time.time()
#         # 视觉帧率限制
#         if now - self.last_draw_time < 1.0 / self.VIS_FPS:
#             return
#         self.last_draw_time = now

#         # 检查是否需要更新文本
#         update_text = now - self.last_text_update_time > 1.0 / self.TXT_FPS
#         if update_text:
#             self.last_text_update_time = now
            
#         # 遍历所有手的数据进行绘图更新 (左和右)
#         for hand, data in self.pending_data.items():
#             plots = self.plots[hand]
#             finger_mats = data['finger_mats']
#             palm_mat = data['palm_mat']
            
#             # 更新图像数据
#             for i in range(5):
#                 plots['ims'][i].set_data(finger_mats[i])
#             plots['im_palm'].set_data(palm_mat)

#             if update_text:
#                 # 更新手指文本
#                 for i in range(5):
#                     val_mat = finger_mats[i]
#                     for r in range(val_mat.shape[0]):
#                         for c in range(val_mat.shape[1]):
#                             txt = f'{int(val_mat[r,c]):d}' if not np.isnan(val_mat[r, c]) else ''
#                             plots['txts'][i][r][c].set_text(txt)
                
#                 # 更新手掌文本
#                 for r in range(palm_mat.shape[0]):
#                     for c in range(palm_mat.shape[1]):
#                         txt = f'{int(palm_mat[r,c]):d}' if not np.isnan(palm_mat[r, c]) else ''
#                         plots['txt_palm'][r][c].set_text(txt)

#                 # 更新统计信息文本
#                 plots['stats_txt'].set_text(data['stats_text'])
#                 plots['stats_txt'].set_color(data['stats_color'])

#         # 所有数据更新完毕后，统一刷新一次画布
#         self.fig.canvas.draw_idle()
#         self.fig.canvas.flush_events()

#     def close(self):
#         plt.ioff()
#         plt.close(self.fig)

class GloveVisualizer:
    def __init__(self, configs: Dict[HandType, HandConfig], VIS_FPS: int = 2, TXT_FPS: int = 2, color_limit=255):
        self.configs = configs
        self.color_limit = color_limit
        self.hands = list(configs.keys())
        self.num_hands = len(self.hands)
        self.VIS_FPS = VIS_FPS
        self.TXT_FPS = TXT_FPS

        # --- 模式检测 ---
        self.is_dual_hand_mode = False
        if self.num_hands == 2:
            if not (HandType.LEFT in self.hands and HandType.RIGHT in self.hands):
                raise ValueError("Dual-hand mode requires configs for both LEFT and RIGHT hands.")
            print("Visualizer initialized in DUAL-HAND sync mode.")
            self.is_dual_hand_mode = True
            # 双手模式需要暂存区和线程锁
            self.pending_data = {}
            self.hand_data_lock = threading.Lock()
        elif self.num_hands == 1:
            print(f"Visualizer initialized in SINGLE-HAND mode for {self.hands[0].value.capitalize()} hand.")
        else:
            raise ValueError(f"Unsupported number of hands: {self.num_hands}. Only 1 or 2 are supported.")

        plt.ion()
        self.fig = plt.figure(figsize=(8 * self.num_hands, 8))
        self.fig.suptitle("Tactile Glove Visualization", fontsize=16)

        self._setup_plots()
        self.last_draw_time = 0.0
        self.last_text_update_time = 0.0

    def _setup_plots(self):
        """(无需修改) 创建 matplotlib 控件。"""
        self.plots = {}
        num_cols = 5 * self.num_hands + max(0, self.num_hands - 1)
        gs = self.fig.add_gridspec(3, num_cols, height_ratios=[1, 0.2, 2], wspace=0.5, hspace=0.4)
        finger_titles = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

        for i, hand in enumerate(self.hands):
            col_offset = i * (5 + int(self.num_hands > 1))
            hand_plots = {'ims': [], 'txts': [], 'stats_txt': None, 'im_palm': None, 'txt_palm': None}

            for col, title in enumerate(finger_titles):
                ax = self.fig.add_subplot(gs[0, col_offset + col])
                im = ax.imshow(np.full((5, 3), np.nan), vmin=-10, vmax=self.color_limit, cmap='viridis')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{hand.value.capitalize()} {title}", fontsize=10)
                txt = [[ax.text(j, i, '', ha='center', va='center', fontsize=7, color='white') for j in range(3)] for i in range(5)]
                hand_plots['ims'].append(im); hand_plots['txts'].append(txt)

            ax_palm = self.fig.add_subplot(gs[2, col_offset:col_offset+5])
            im_palm = ax_palm.imshow(np.full((5, 15), np.nan), vmin=-10, vmax=self.color_limit, cmap='viridis')
            ax_palm.set_xticks([]); ax.set_yticks([])
            txt_palm = [[ax_palm.text(j, i, '', ha='center', va='center', fontsize=7, color='white') for j in range(15)] for i in range(5)]
            hand_plots['im_palm'] = im_palm; hand_plots['txt_palm'] = txt_palm

            stats_ax = self.fig.add_subplot(gs[0:3, col_offset:col_offset+5])
            stats_ax.axis('off')
            hand_plots['stats_txt'] = stats_ax.text(0.5, 0.45, '', va='top', ha='center', fontsize=9,
                                                    bbox=dict(facecolor='white', alpha=0.7, pad=4))
            self.plots[hand] = hand_plots
        
        if self.num_hands > 0:
            cax = self.fig.add_axes([0.92, 0.15, 0.015, 0.7])
            self.fig.colorbar(self.plots[self.hands[0]]['im_palm'], cax=cax, label='Sensor Value')
        self.fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.05)

    def _update_plot_for_hand(self, hand: HandType, data: Dict[str, Any], update_text: bool):
        """
        (新) 实际更新单只手 plot 数据的核心函数。
        """
        plots = self.plots[hand]
        finger_mats = data['finger_mats']
        palm_mat = data['palm_mat']

        # print(finger_mats[0].shape)
        if len(palm_mat) == 1:
            palm_mat = palm_mat[0]
        # print(len(palm_mat))

        # 更新图像
        for i in range(5):
            plots['ims'][i].set_data(finger_mats[i])
        plots['im_palm'].set_data(palm_mat)

        if update_text:
            # 更新文本
            for i in range(5):
                val_mat = finger_mats[i]
                for r in range(val_mat.shape[0]):
                    for c in range(val_mat.shape[1]):
                        txt = f'{int(val_mat[r,c]):d}' if not np.isnan(val_mat[r, c]) else ''
                        plots['txts'][i][r][c].set_text(txt)

            for r in range(palm_mat.shape[0]):
                for c in range(palm_mat.shape[1]):
                    txt = f'{int(palm_mat[r,c]):d}' if not np.isnan(palm_mat[r, c]) else ''
                    plots['txt_palm'][r][c].set_text(txt)
            
            # 更新统计信息
            plots['stats_txt'].set_text(data['stats_text'])
            plots['stats_txt'].set_color(data['stats_color'])


    # def update(self, frame: TacGloveFrame, stats_monitor):
    #     """
    #     根据初始化时确定的模式，分发更新任务。
    #     """
    #     # --- 步骤 1: 数据提取 (通用) ---
    #     hand = HandType(frame.hand)
    #     config = self.configs[hand]
    #     sensor_values = np.frombuffer(frame.data[:256], dtype=np.uint8)
        
    #     processed_data = {
    #         'finger_mats': [config.extract_finger_val_mat(i, sensor_values) for i in range(5)],
    #         'palm_mat': config.extract_palm_val_mat(sensor_values),
    #     }
    #     stats_text, color = stats_monitor.get_stats_text(frame.sensor_type)
    #     processed_data['stats_text'] = f"{hand.value.capitalize()} Hand\n" + stats_text
    #     processed_data['stats_color'] = color

    #     # --- 步骤 2: 根据模式执行不同逻辑 ---
    #     if self.is_dual_hand_mode:
    #         # 双手同步逻辑
    #         with self.hand_data_lock:
    #             self.pending_data[hand] = processed_data
    #             if len(self.pending_data) == 2:
    #                 now = time.time()
    #                 if now - self.last_draw_time < 1.0 / self.VIS_FPS:
    #                     self.pending_data.clear() # 帧率太快，丢弃这对数据
    #                     return
                    
    #                 self.last_draw_time = now
    #                 update_text = now - self.last_text_update_time > 1.0 / self.TXT_FPS
    #                 if update_text:
    #                     self.last_text_update_time = now

    #                 for hand_to_draw, data_to_draw in self.pending_data.items():
    #                     self._update_plot_for_hand(hand_to_draw, data_to_draw, update_text)
                    
    #                 self.pending_data.clear() # 绘制完成后清空
    #     else:
    #         # 单手即时逻辑
    #         now = time.time()
    #         if now - self.last_draw_time < 1.0 / self.VIS_FPS:
    #             return
            
    #         self.last_draw_time = now
    #         update_text = now - self.last_text_update_time > 1.0 / self.TXT_FPS
    #         if update_text:
    #             self.last_text_update_time = now
                
    #         self._update_plot_for_hand(hand, processed_data, update_text)

    #     # --- 步骤 3: 刷新画布 (通用) ---
    #     self.fig.canvas.draw_idle()
    #     self.fig.canvas.flush_events()

    def update(self, frame=None, stats_monitor=None, finger_mats=None, palm_mat=None, vis_monitor=True, hand_type=None):
        """
        更新可视化显示，支持多种数据输入方式，并优化了双手同步性能
        """
        # --- 步骤 1: 数据提取与预处理 ---
        if finger_mats is None:
            # 从帧对象提取数据
            assert (hand_type is None) or (hand_type == HandType(frame.hand))
            assert finger_mats is None
            assert palm_mat is None
            
            hand, stype = HandType(frame.hand), frame.sensor_type
            config = self.configs[hand]
            plots = self.plots[hand]
            
            # 提取传感器数据
            sensor_values = np.frombuffer(frame.data[:256], dtype=np.uint8)
            finger_val_mats = [config.extract_finger_val_mat(i, sensor_values) for i in range(5)]
            palm_val_mat = config.extract_palm_val_mat(sensor_values)
            
            # 准备统计信息
            if vis_monitor and stats_monitor is not None:
                stats_text, color = stats_monitor.get_stats_text(stype)
                stats_display_text = f"{hand.value.capitalize()} Hand\n" + stats_text
                stats_color = color
            else:
                stats_display_text = ""
                stats_color = "black"
                
            processed_data = {
                'hand': hand,
                'finger_mats': finger_val_mats,
                'palm_mat': palm_val_mat,
                'stats_text': stats_display_text,
                'stats_color': stats_color
            }
        else:
            # 直接使用提供的矩阵数据
            # assert not vis_monitor or stats_monitor is None
            assert hand_type is not None
            assert finger_mats is not None
            assert palm_mat is not None
            
            hand = hand_type
            finger_val_mats = finger_mats
            palm_val_mat = palm_mat
            plots = self.plots[hand_type]
            
            processed_data = {
                'hand': hand,
                'finger_mats': finger_val_mats,
                'palm_mat': palm_val_mat,
                'stats_text': "",
                'stats_color': "black",
            }
        
        # --- 步骤 2: 根据模式执行不同更新逻辑 ---
        if self.is_dual_hand_mode:
            # 双手同步逻辑（仅当有帧数据时）
            with self.hand_data_lock:
                
                self.pending_data[hand] = processed_data
                if len(self.pending_data) == 2:
                    now = time.time()
                    if now - self.last_draw_time < 1.0 / self.VIS_FPS:
                        # self.pending_data.clear()  # 帧率太快，丢弃这对数据
                        return
                    
                    self.last_draw_time = now
                    update_text = now - self.last_text_update_time > 1.0 / self.TXT_FPS
                    if update_text:
                        self.last_text_update_time = now
                    
                    for hand_to_draw, data_to_draw in self.pending_data.items():
                        self._update_plot_for_hand(hand_to_draw, data_to_draw, update_text)
                    
                    self.pending_data.clear()  # 绘制完成后清空
        else:
            # 单手即时逻辑或非帧数据模式
            now = time.time()
            if now - self.last_draw_time < 1.0 / self.VIS_FPS:
                return
            
            self.last_draw_time = now
            update_text = now - self.last_text_update_time > 1.0 / self.TXT_FPS
            if update_text:
                self.last_text_update_time = now
                
            self._update_plot_for_hand(hand, processed_data, update_text)
        
        # --- 步骤 3: 刷新画布 ---
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)

# --- Main Execution ---

def original_main():
    parser = argparse.ArgumentParser(description="Tactile Glove Data Visualizer")
    parser.add_argument('hands', type=str, choices=['left', 'right', 'both'],
                        help="Which hand(s) to connect to.")
    args = parser.parse_args()

    # Determine which hands to set up
    selected_hands = []
    if args.hands == 'left':
        selected_hands.append(HandType.LEFT)
    elif args.hands == 'right':
        selected_hands.append(HandType.RIGHT)
    elif args.hands == 'both':
        selected_hands.append(HandType.LEFT)
        selected_hands.append(HandType.RIGHT)

    if not selected_hands:
        print("[ERROR] No hands selected. Use 'left', 'right', or 'both'.")
        return

    # Create configurations and data structures for selected hands
    hand_configs = {hand: HandConfig(hand) for hand in selected_hands}
    data_queue = queue.Queue(maxsize=120)
    stats_monitors = {hand: StatsMonitor(expect_fps=SAMPLE_RATE_HZ) for hand in selected_hands}

    # Initialize and start serial reader threads
    readers = [SerialReader(config, data_queue) for config in hand_configs.values()]
    for r in readers:
        r.start()
        time.sleep(0.1) # Stagger start slightly

    # Check if any reader connected successfully
    if not any(r.connected for r in readers):
        print("\n[FATAL] No serial ports could be opened. Exiting.")
        for r in readers: r.stop()
        return

    # Initialize visualizer
    viz = GloveVisualizer(hand_configs)
    
    latest_frames = {} # Store the most recent frame for each hand

    gc.disable()
    try:
        print("\n[INFO] Visualization running. Press Ctrl+C to exit.")
        while plt.fignum_exists(viz.fig.number):
            try:
                # Process all available frames in the queue
                while True:
                    frame: TacGloveFrame = data_queue.get_nowait()
                    stats_monitors[frame.hand].update(frame)
                    latest_frames[frame.hand] = frame # Overwrite with the newest frame
                    # for hand, frame in latest_frames.items():
                    #     viz.update(frame, stats_monitors[hand])
            except queue.Empty:
                # Once queue is empty, update visualization with the latest data
                for hand, frame in latest_frames.items():
                    viz.update(frame, stats_monitors[hand])
                
                # If no data is coming, still allow GUI events to process
                if not latest_frames:
                    viz.fig.canvas.flush_events()
                
                time.sleep(1 / (3 * viz.VIS_FPS)) # Sleep briefly to yield CPU
    
    except KeyboardInterrupt:
        print("\n[INFO] Quit by Ctrl-C.")
    finally:
        print("[INFO] Shutting down threads...")
        for reader in readers:
            reader.stop()
        viz.close()
        gc.enable()
        print("[INFO] Cleanup complete. Exiting.")


# def main():
#     parser = argparse.ArgumentParser(description="Tactile Glove Data Visualizer")
#     parser.add_argument('hands', type=str, choices=['left', 'right', 'both'],
#                         help="Which hand(s) to connect to.")
#     args = parser.parse_args()

#     # Determine which hands to set up
#     selected_hands = []
#     if args.hands == 'left':
#         selected_hands.append(HandType.LEFT)
#     elif args.hands == 'right':
#         selected_hands.append(HandType.RIGHT)
#     elif args.hands == 'both':
#         selected_hands.append(HandType.LEFT)
#         selected_hands.append(HandType.RIGHT)

#     if not selected_hands:
#         print("[ERROR] No hands selected. Use 'left', 'right', or 'both'.")
#         return

#     # Create configurations and data structures for selected hands
#     hand_configs = {hand: HandConfig(hand) for hand in selected_hands}
    
#     # MODIFICATION 3: Create a separate queue for each hand.
#     data_queues = {hand: queue.Queue(maxsize=2) for hand in selected_hands}
#     # print(data_queues)
#     # breakpoint()
#     stats_monitors = {hand: StatsMonitor(expect_fps=SAMPLE_RATE_HZ) for hand in selected_hands}

#     # Initialize and start serial reader threads, passing the appropriate queue to each.
#     readers = [SerialReader(config, data_queues[config.hand]) for config in hand_configs.values()]
#     for r in readers:
#         r.start()
#         time.sleep(0.1) # Stagger start slightly

#     # Check if any reader connected successfully
#     if not any(r.connected for r in readers):
#         print("\n[FATAL] No serial ports could be opened. Exiting.")
#         for r in readers: r.stop()
#         return

#     # Initialize visualizer
#     viz = GloveVisualizer(hand_configs)
    
#     # This dictionary will hold the most recent frame received for each hand.
#     latest_frames = {} 

#     gc.disable()
#     try:
#         print("\n[INFO] Visualization running. Press Ctrl+C to exit.")
#         # while plt.fignum_exists(viz.fig.number):
#         global temp_time
#         temp_time = time.time()
#         ori_time = time.time()
#         temp_n = 1
#         while True:
#             t0 = time.time()
#             # MODIFICATION 4: New loop logic to get the latest frame from each queue.
#             frames_to_update_this_cycle = {}
            
#             # print(time.time()-temp_time)
#             print((time.time() - ori_time) / temp_n)
#             temp_n += 1

#             for hand, q in data_queues.items():
#                 # print(hand)
#                 latest_frame_this_hand = None
#                 ori_frame = None
#                 first_flag = True
                
#                 while latest_frame_this_hand is None:
#                     try:
#                         # Drain the queue to get the most recent item.
#                         while True:
#                             latest_frame_this_hand = q.get_nowait()
#                     except queue.Empty:
#                         # This is expected, means we've drained the queue.
#                         pass 

#                 if latest_frame_this_hand:
#                     # If we got a new frame, update stats and store it.
#                     stats_monitors[hand].update(latest_frame_this_hand)
#                     # print(time.time()-t0)
#                     latest_frames[hand] = latest_frame_this_hand
                
#                     # After checking all queues, update visualization with the latest data we have.
#                     for hand, frame in latest_frames.items():
#                         # print(f"update:  {hand.name.lower()}")
#                         viz.update(frame, stats_monitors[hand])
                
#                 # If no data is coming at all, still allow GUI events to process
#                 if not latest_frames:
#                     viz.fig.canvas.flush_events()

#                 time.sleep(1e-10) # Sleep briefly to yield CPU

#     except KeyboardInterrupt:
#         print("\n[INFO] Quit by Ctrl-C.")
#     finally:
#         print("[INFO] Shutting down threads...")
#         for reader in readers:
#             reader.stop()
#         viz.close()
#         gc.enable()
#         print("[INFO] Cleanup complete. Exiting.")

# def main():
#     parser = argparse.ArgumentParser(description="Tactile Glove Data Visualizer")
#     parser.add_argument('hands', type=str, choices=['left', 'right', 'both'],
#                         help="Which hand(s) to connect to.")
#     args = parser.parse_args()

#     selected_hands = []
#     if args.hands == 'left':
#         selected_hands.append(HandType.LEFT)
#     elif args.hands == 'right':
#         selected_hands.append(HandType.RIGHT)
#     elif args.hands == 'both':
#         selected_hands.append(HandType.LEFT)
#         selected_hands.append(HandType.RIGHT)

#     if not selected_hands:
#         print("[ERROR] No hands selected. Use 'left', 'right', or 'both'.")
#         return

#     hand_configs = {hand: HandConfig(hand) for hand in selected_hands}
#     data_queues = {hand: queue.Queue(maxsize=2) for hand in selected_hands}
#     stats_monitors = {hand: StatsMonitor(expect_fps=SAMPLE_RATE_HZ) for hand in selected_hands}

#     readers = [SerialReader(config, data_queues[config.hand]) for config in hand_configs.values()]
#     for r in readers:
#         r.start()
#         time.sleep(0.1)

#     if not any(r.connected for r in readers):
#         print("\n[FATAL] No serial ports could be opened. Exiting.")
#         for r in readers: r.stop()
#         return

#     viz = GloveVisualizer(hand_configs, VIS_FPS=2, TXT_FPS=2)
    
#     gc.disable()
#     try:
#         print("\n[INFO] Visualization running. Press Ctrl+C to exit.")
        
#         # 用于性能调试
#         ori_time = time.time()
#         temp_n = 1

#         while True:
#             print(f"Loop time: {(time.time() - ori_time) / temp_n:.5f}s")
#             temp_n += 1

#             # --- MODIFICATION START: 重构主循环逻辑 ---

#             # 阶段 1: 数据收集
#             # 在这个循环周期中，我们找到的所有新的数据帧都存放在这里。
#             new_frames_this_cycle = {}
#             for hand, q in data_queues.items():
#                 latest_frame_this_hand = None
#                 try:
#                     # 依旧是排空队列，只取最新的一个
#                     while True:
#                         latest_frame_this_hand = q.get_nowait()
#                 except queue.Empty:
#                     pass  # 队列已空，正常现象

#                 # 如果为这只手找到了新的数据帧
#                 if latest_frame_this_hand:
#                     new_frames_this_cycle[hand] = latest_frame_this_hand
#                     # 立刻更新统计数据
#                     stats_monitors[hand].update(latest_frame_this_hand)

#             # 阶段 2: 渲染
#             # 只有在收集到新数据帧的情况下，才去调用 viz.update
#             if new_frames_this_cycle:
#                 # 遍历本周期收集到的所有新数据帧，并分发给 visualizer
#                 for hand, frame in new_frames_this_cycle.items():
#                     viz.update(frame, stats_monitors[hand])
#             else:
#                 # 如果完全没有新数据，也要让 GUI 保持响应
#                 viz.fig.canvas.flush_events()

#             # --- MODIFICATION END ---
            
#             # 短暂休眠，避免 CPU 100% 占用，同时让出时间给其他线程
#             time.sleep(1e-7) 

#     except KeyboardInterrupt:
#         print("\n[INFO] Quit by Ctrl-C.")
#     finally:
#         print("[INFO] Shutting down threads...")
#         for reader in readers:
#             reader.stop()
#         viz.close()
#         gc.enable()
#         print("[INFO] Cleanup complete. Exiting.")



def main():
    # --- 参数解析和初始化部分 (与之前相同) ---
    parser = argparse.ArgumentParser(description="Tactile Glove Data Visualizer")
    parser.add_argument('hands', type=str, choices=['left', 'right', 'both'],
                        help="Which hand(s) to connect to.")
    args = parser.parse_args()

    selected_hands = []
    if args.hands == 'left':
        selected_hands.append(HandType.LEFT)
    elif args.hands == 'right':
        selected_hands.append(HandType.RIGHT)
    elif args.hands == 'both':
        selected_hands.append(HandType.LEFT)
        selected_hands.append(HandType.RIGHT)

    if not selected_hands:
        print("[ERROR] No hands selected. Use 'left', 'right', or 'both'.")
        return

    hand_configs = {hand: HandConfig(hand) for hand in selected_hands}
    data_queues = {hand: queue.Queue(maxsize=2) for hand in selected_hands}
    stats_monitors = {hand: StatsMonitor(expect_fps=SAMPLE_RATE_HZ) for hand in selected_hands}

    readers = [SerialReader(config, data_queues[config.hand]) for config in hand_configs.values()]
    for r in readers:
        r.start()
        time.sleep(0.1)

    if not any(r.connected for r in readers):
        print("\n[FATAL] No serial ports could be opened. Exiting.")
        for r in readers: r.stop()
        return

    viz = GloveVisualizer(hand_configs, VIS_FPS=3) # 可以把Visualizer的内部限制放宽
    
    gc.disable()
    try:
        print("\n[INFO] Visualization running. Press Ctrl+C to exit.")
        
        # ==============================================================================
        # --- MODIFICATION: 根据模式选择不同的主循环 ---
        # ==============================================================================
        run_dual_hand_loop_throttled(viz, data_queues, stats_monitors)

    except KeyboardInterrupt:
        print("\n[INFO] Quit by Ctrl-C.")
    finally:
        print("[INFO] Shutting down threads...")
        for reader in readers:
            reader.stop()
        viz.close()
        gc.enable()
        print("[INFO] Cleanup complete. Exiting.")

# --- 新增：为双手模式优化的节流循环 ---
def run_dual_hand_loop_throttled(viz, data_queues, stats_monitors):
    """
    为双手模式设计的循环，包含渲染节流机制。
    """
    # --- 节流参数 ---
    TARGET_RENDER_FPS = 2  # 目标渲染帧率，20-30之间视觉效果就很好
    RENDER_INTERVAL = 1.0 / TARGET_RENDER_FPS
    last_render_time = 0

    ori_time = time.time()
    temp_n = 1

    while True:
        print(f"Loop time: {(time.time() - ori_time) / temp_n:.5f}s")
        temp_n += 1

        # 阶段 1: 数据处理 (总以最快速度运行)
        new_frames_this_cycle = {}
        for hand, q in data_queues.items():
            latest_frame = None
            while latest_frame is None:
                try:
                    while True:
                        latest_frame = q.get_nowait()
                except queue.Empty:
                    pass

            if latest_frame:
                new_frames_this_cycle[hand] = latest_frame
                # 统计数据总是实时更新
                stats_monitors[hand].update(latest_frame)
        
        # 阶段 2: 渲染 (根据节流阀决定是否执行)
        now = time.time()
        if now - last_render_time > RENDER_INTERVAL:
            last_render_time = now
            
            # 如果这个周期有新数据，就用新数据更新
            if new_frames_this_cycle:
                for hand, frame in new_frames_this_cycle.items():
                    viz.update(frame, stats_monitors[hand])
            # # 如果这个周期没有新数据，但时间到了，也要flush_events来保持窗口响应
            # else:
            #      viz.fig.canvas.flush_events()
        
        # 短暂休眠，避免 CPU 100% 占用
        time.sleep(1e-7)

if __name__ == "__main__":
    # 假设 SAMPLE_RATE_HZ 等常量已定义
    SAMPLE_RATE_HZ = 200
    main()
