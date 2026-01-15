import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import argparse
import os
import sys
import time

# Constants
NUM_FINGERS = 5
SENSORS_PER_FINGER = 12
NUM_BENDS = 5
SENSORS_PALM = 72
TOTAL_SENSORS = NUM_FINGERS * SENSORS_PER_FINGER + NUM_BENDS + SENSORS_PALM # 137

FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

# Setup layout settings similar to glove_hand.py default
plt.rcParams['figure.subplot.wspace'] = 0.5
plt.rcParams['figure.subplot.hspace'] = 0.4

class TactileReplayer:
    def __init__(self, log_dir, fps=20):
        self.log_dir = log_dir
        self.fps = fps
        self.interval = 1000.0 / fps # ms
        
        self.hands = {} # 'left', 'right' -> DataFrame
        
        # Load Data
        # Try standard session structure
        l_path = os.path.join(log_dir, 'tactile_left', 'tactile_data.csv')
        r_path = os.path.join(log_dir, 'tactile_right', 'tactile_data.csv')
        
        if os.path.exists(l_path):
            self._load_data('left', l_path)
        if os.path.exists(r_path):
            self._load_data('right', r_path)

        # Fallback: check if log_dir IS the tactile folder
        if not self.hands:
             direct_path = os.path.join(log_dir, 'tactile_data.csv')
             if os.path.exists(direct_path):
                 # Guess side
                 if 'left' in log_dir.lower():
                     self._load_data('left', direct_path)
                 elif 'right' in log_dir.lower():
                     self._load_data('right', direct_path)
                 else:
                     print("Found tactile_data.csv but path doesn't specify left/right. Assuming right.")
                     self._load_data('right', direct_path)
        
        if not self.hands:
            print("No tactile data found. Checked for:")
            print(f" - {l_path}")
            print(f" - {r_path}")
            print(f" - {os.path.join(log_dir, 'tactile_data.csv')}")
            sys.exit(1)
            
        # Synchronize lengths
        max_len = 0
        for side in self.hands:
            max_len = max(max_len, len(self.hands[side]))
            
        self.num_frames = max_len
        print(f"Loaded {len(self.hands)} hands. Max frames: {self.num_frames}")
        
        # Setup Plot
        num_hands = len(self.hands)
        self.fig = plt.figure(figsize=(8 * num_hands, 8))
        self.fig.suptitle("Tactile Replay", fontsize=16)
        
        self.plots = {} # 'left' -> { 'ims': [], 'im_palm': ... }
        
        self._setup_layout()
        self.fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.05)
        
        # Shared Colorbar
        # Use first hand's palm image as reference
        first_hand = sorted(self.hands.keys())[0]
        cax = self.fig.add_axes([0.92, 0.15, 0.015, 0.7])
        self.fig.colorbar(self.plots[first_hand]['im_palm'], cax=cax, label='Sensor Value')
        
        # Playback state
        self.current_frame = 0
        self.is_paused = False
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
    def _load_data(self, side, path):
        if os.path.exists(path):
            try:
                # Read CSV without header
                df = pd.read_csv(path, header=None)
                # Filter valid columns if any extra
                if df.shape[1] >= TOTAL_SENSORS:
                    self.hands[side] = df.iloc[:, :TOTAL_SENSORS].to_numpy() # Convert to numpy for speed
                    print(f"Loaded {side} hand: {df.shape}")
                else:
                    print(f"Skipping {side}: invalid shape {df.shape} (expected >= {TOTAL_SENSORS} cols)")
            except Exception as e:
                print(f"Error loading {side}: {e}")

    def _setup_layout(self):
        # Determine columns roughly
        # Left Hand: 5 cols + 1 palm
        # Right Hand: 5 cols + 1 palm
        # Simple Grid: 3 rows.
        # Top row: Finger tips
        # Mid row: Labels?
        # Bottom row: Palm
        
        num_hands = len(self.hands)
        cols_per_hand = 5 + (1 if num_hands > 1 else 0) # 5 cols + 1 spacing if dual
        total_cols = cols_per_hand * num_hands
        if num_hands > 1:
             total_cols = 5 * num_hands + (num_hands - 1) # Matches 5*num + spacing
        else:
             total_cols = 5

        # From glove_hand.py:
        # num_cols = 5 * self.num_hands + max(0, self.num_hands - 1)
        # gs = self.fig.add_gridspec(3, num_cols, height_ratios=[1, 0.2, 2], wspace=0.5, hspace=0.4)
        
        num_cols = 5 * num_hands + max(0, num_hands - 1)
        gs = self.fig.add_gridspec(3, num_cols, height_ratios=[1, 0.2, 2], wspace=0.5, hspace=0.4)
        
        hand_order = sorted(self.hands.keys()) # ['left', 'right'] or just one
        
        for h_idx, side in enumerate(hand_order):
            col_offset = h_idx * (5 + int(num_hands > 1))
            
            hand_plots = {'ims': [], 'txts': [], 'im_palm': None, 'txt_palm': None}
            
            # Fingers
            for f_idx in range(5):
                ax = self.fig.add_subplot(gs[0, col_offset + f_idx])
                # Initialize empty 5x3
                im = ax.imshow(np.full((5, 3), np.nan), vmin=-10, vmax=255, cmap='viridis')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{side.capitalize()} {FINGER_NAMES[f_idx]}", fontsize=10)
                
                # Text annotations
                txt_grid = [[ax.text(c, r, '', ha='center', va='center', fontsize=7, color='white') 
                             for c in range(3)] for r in range(5)]
                
                hand_plots['ims'].append(im)
                hand_plots['txts'].append(txt_grid)
                
            # Palm
            # Span 5 columns
            ax_palm = self.fig.add_subplot(gs[2, col_offset : col_offset + 5])
            im_palm = ax_palm.imshow(np.full((5, 15), np.nan), vmin=-10, vmax=255, cmap='viridis')
            ax_palm.set_xticks([]); ax_palm.set_yticks([])
            ax_palm.set_title(f"{side.capitalize()} Palm")
            
            txt_palm = [[ax_palm.text(c, r, '', ha='center', va='center', fontsize=7, color='white') 
                         for c in range(15)] for r in range(5)]
            
            hand_plots['im_palm'] = im_palm
            hand_plots['txt_palm'] = txt_palm
            
            self.plots[side] = hand_plots
            
    def _get_frame_data(self, side, frame_idx):
        data_arr = self.hands[side]
        if frame_idx >= len(data_arr):
            return np.full(TOTAL_SENSORS, np.nan) # Return NaNs if OOB
        
        vec = data_arr[frame_idx].copy().astype(float)
        # Handle -1 as invalid (NaN)
        vec[vec == -1] = np.nan
        return vec

    def _process_vector_to_mats(self, vector, side):
        # Vector size 137
        # 0-59: Fingers (5 * 12)
        # 60-64: Bends (5)
        # 65-136: Palm (72)
        
        finger_mats = []
        
        # Fingers
        for f in range(5):
            mat = np.full((5, 3), np.nan)
            
            # 12 pad sensors
            start = f * 12
            # The list in glove_hand.py was flat 12. 
            # In _create_finger_id_mats: 4 rows * 3 cols
            pad_vals = vector[start : start+12]
            
            # Row 0-3
            for r in range(4):
                for c in range(3):
                    mat[r, c] = pad_vals[r*3 + c]
            
            # Bend sensor at 60 + f
            # Placed at 4, 1
            bend_val = vector[60 + f]
            mat[4, 1] = bend_val
            
            finger_mats.append(mat)
            
        # Palm
        palm_mat = np.full((5, 15), np.nan)
        palm_vals = vector[65 : 137] # 72 values
        
        # Row 0: 12 values
        row0_vals = palm_vals[0:12]
        if side == 'right':
            # Right hand palm row 0 is offset by 3 cols (cols 3-14)
            palm_mat[0, 3:15] = row0_vals
        else:
            # Left hand palm row 0 is cols 0-11
            palm_mat[0, 0:12] = row0_vals
            
        # Rows 1-4: 60 values (4 * 15)
        remaining = palm_vals[12:] # 60 vals
        remaining = remaining.reshape(4, 15)
        palm_mat[1:5, :] = remaining
        
        return finger_mats, palm_mat

    def update(self, i):
        if self.is_paused:
            return
            
        if self.current_frame >= self.num_frames:
            self.current_frame = 0
            
        for side, plots in self.plots.items():
            vec = self._get_frame_data(side, self.current_frame)
            
            # If all are NaN, it might mean we ran out of data or it's a gap
            # But process_vector_to_mats handles NaNs fine (propagates them)
            
            f_mats, p_mat = self._process_vector_to_mats(vec, side)
            
            # Update Fingers
            for f_idx in range(5):
                mat = f_mats[f_idx]
                plots['ims'][f_idx].set_data(mat)
                
                # Update text
                t_grid = plots['txts'][f_idx]
                for r in range(5):
                    for c in range(3):
                        val = mat[r, c]
                        txt = f"{int(val)}" if not np.isnan(val) else ""
                        t_grid[r][c].set_text(txt)
                        
            # Update Palm
            plots['im_palm'].set_data(p_mat)
            p_txts = plots['txt_palm']
            for r in range(5):
                for c in range(15):
                    val = p_mat[r, c]
                    txt = f"{int(val)}" if not np.isnan(val) else ""
                    p_txts[r][c].set_text(txt)
        
        self.fig.suptitle(f"Tactile Replay - Frame {self.current_frame}/{self.num_frames}")
        self.current_frame += 1

    def _on_key(self, event):
        if event.key == ' ':
            self.is_paused = not self.is_paused
        elif event.key == 'right':
            self.current_frame += 1
            if self.current_frame >= self.num_frames: self.current_frame = 0
            # Force update viz? 
            # The animate loop will catch up, but might be paused.
            # If paused, we can manually trigger update logic, but matplotlib anim handles it cleaner if running.
        elif event.key == 'left':
            self.current_frame -= 1
            if self.current_frame < 0: self.current_frame = self.num_frames - 1

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update, interval=self.interval, save_count=1)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Replay Tactile Glove data")
    parser.add_argument("log_dir", help="Session directory containing tactile_left/ or tactile_right/")
    parser.add_argument("--fps", type=int, default=300, help="Playback FPS")
    args = parser.parse_args()
    
    replayer = TactileReplayer(args.log_dir, args.fps)
    replayer.run()

if __name__ == "__main__":
    main()
