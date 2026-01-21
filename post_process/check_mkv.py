import os
import sys
import subprocess
import re
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter

# Configuration
REPORT_DIR = "corruption_reports"

def get_video_metadata(file_path):
    """
    Retrieves total frame count and duration using ffprobe.
    """
    try:
        command = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            "-select_streams", "v:0",
            file_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        data = json.loads(result.stdout)

        if not data.get("streams"):
            return None

        stream = data["streams"][0]
        
        # Try to get exact frame count
        frames = stream.get("nb_frames")
        if not frames:
            # Estimate from duration * fps if nb_frames is missing
            duration = float(data["format"].get("duration", 0))
            avg_frame_rate = stream.get("avg_frame_rate", "30/1")
            num, den = map(float, avg_frame_rate.split('/'))
            fps = num / den if den > 0 else 30
            frames = int(duration * fps)
        else:
            frames = int(frames)
            
        return frames
    except Exception as e:
        print(f"[System] Could not retrieve metadata for {file_path}: {e}")
        return 0

def parse_error_type(log_line):
    """
    Categorizes the raw ffmpeg error string into a readable type.
    """
    log_line = log_line.lower()
    if "reference" in log_line and ">=" in log_line:
        return "Invalid Reference Frame"
    if "intra mode" in log_line or "top block unavailable" in log_line:
        return "Intra Prediction Error"
    if "decoding mb" in log_line:
        return "Macroblock Decode Error"
    if "cabac" in log_line:
        return "CABAC Entropy Error"
    if "slice" in log_line:
        return "Slice Header Error"
    return "General Decoding Error"

def analyze_video(file_path, total_frames):
    """
    Runs ffmpeg to scan the file, parsing stderr to map errors to frame numbers.
    """
    command = [
        "ffmpeg",
        "-v", "info",       # Info level needed to see progress "frame=..."
        "-i", file_path,
        "-f", "null",       # Null output
        "-"
    ]

    error_events = [] # List of tuples: (frame_index, error_message, error_type)
    current_frame = 0
    
    # Regex to extract current frame from ffmpeg progress line: "frame=  123"
    frame_pattern = re.compile(r"frame=\s*(\d+)")
    # Regex to detect h264/hevc errors
    error_prefix_pattern = re.compile(r"\[.*?\] (.*)")

    try:
        # We process stderr line by line. 
        # Note: ffmpeg uses \r for progress bars, which requires careful handling.
        process = subprocess.Popen(
            command, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding='utf-8', 
            errors='replace'
        )

        buffer = ""
        while True:
            # Read chunk by chunk to handle \r correctly
            chunk = process.stderr.read(1024)
            if not chunk and process.poll() is not None:
                break
            
            if chunk:
                buffer += chunk
                while '\n' in buffer or '\r' in buffer:
                    # Split by either newline or carriage return
                    if '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                    else:
                        line, buffer = buffer.split('\r', 1)
                    
                    line = line.strip()
                    if not line:
                        continue

                    # 1. Update Current Frame Position
                    match_frame = frame_pattern.search(line)
                    if match_frame:
                        current_frame = int(match_frame.group(1))

                    # 2. Detect Errors
                    # Filter for lines that look like errors (contain explicit error keywords or the [h264] tag)
                    if "error" in line.lower() or ("[" in line and "]" in line and "frame=" not in line):
                        # Filter out purely informational lines
                        if any(x in line for x in ["Input #", "Output #", "Metadata:", "Duration:"]):
                            continue
                        
                        match_err = error_prefix_pattern.search(line)
                        if match_err:
                            raw_msg = match_err.group(1)
                            err_type = parse_error_type(raw_msg)
                            # Record the error at the current known frame
                            error_events.append({
                                'frame': current_frame,
                                'raw': raw_msg,
                                'type': err_type
                            })

    except Exception as e:
        print(f"[Error] Execution failed: {e}")
        return []

    return error_events

def generate_visualization(file_path, total_frames, error_events, output_dir):
    """
    Generates a PNG dashboard visualizing the corruption.
    """
    if not error_events:
        return

    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]
    
    # Data Preparation
    error_frames = [e['frame'] for e in error_events]
    error_types = [e['type'] for e in error_events]
    unique_corrupt_frames = len(set(error_frames))
    corruption_ratio = (unique_corrupt_frames / total_frames) * 100 if total_frames > 0 else 0
    
    # Setup Figure
    fig = plt.figure(figsize=(12, 10))
    plt.suptitle(f"Corruption Analysis: {filename}", fontsize=16, fontweight='bold')
    
    # Grid Layout
    gs = fig.add_gridspec(3, 2)
    
    # 1. Timeline (Top spanning)
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.set_title("Corruption Timeline (Frame Position)")
    ax_timeline.set_xlim(0, total_frames)
    ax_timeline.set_xlabel("Frame Index")
    ax_timeline.get_yaxis().set_visible(False)
    
    # Plot background track
    ax_timeline.hlines(1, 0, total_frames, colors='lightgray', linewidth=20)
    # Plot errors
    ax_timeline.scatter(error_frames, [1]*len(error_frames), color='red', alpha=0.5, s=10, marker='|')
    ax_timeline.text(0, 1.05, "Start", transform=ax_timeline.transAxes)
    ax_timeline.text(1, 1.05, "End", transform=ax_timeline.transAxes, ha='right')

    # 2. Error Type Distribution (Bottom Left)
    ax_bar = fig.add_subplot(gs[1, 0])
    type_counts = Counter(error_types)
    labels = list(type_counts.keys())
    values = list(type_counts.values())
    
    ax_bar.barh(labels, values, color='salmon')
    ax_bar.set_title("Distribution by Error Type")
    ax_bar.set_xlabel("Count")

    # 3. Health Ratio (Bottom Right)
    ax_pie = fig.add_subplot(gs[1, 1])
    ax_pie.set_title("Frame Health Ratio")
    
    # Cap the pie chart visualization if ratio is tiny to ensure visibility
    sizes = [total_frames - unique_corrupt_frames, unique_corrupt_frames]
    ax_pie.pie(sizes, labels=['Healthy', 'Corrupt'], colors=['#66b3ff', '#ff9999'], 
               autopct='%1.1f%%', startangle=90, explode=(0, 0.1))

    # 4. Text Summary (Footer)
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')
    summary_text = (
        f"Total Frames: {total_frames}\n"
        f"Corrupt Frames (Unique): {unique_corrupt_frames}\n"
        f"Total Error Events: {len(error_events)}\n"
        f"Damage Percentage: {corruption_ratio:.4f}%\n"
        f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    ax_text.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12, 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    # Save
    output_path = os.path.join(output_dir, f"VISUAL_{base_name}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    return output_path

def scan_directory(root_dir):
    print(f"Directory: {root_dir}")
    print("-" * 60)
    
    # Ensure output directory exists
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    # Find MKV files
    mkv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mkv'):
                mkv_files.append(os.path.join(dirpath, filename))

    total_files = len(mkv_files)
    print(f"Found {total_files} MKV files. Starting analysis...")
    print("Output directory for visualizations: " + os.path.abspath(REPORT_DIR))
    print("-" * 60)

    for index, file_path in enumerate(mkv_files):
        filename = os.path.basename(file_path)
        print(f"[{index + 1}/{total_files}] Processing: {filename}")
        
        # 1. Get Metadata
        total_frames = get_video_metadata(file_path)
        if total_frames == 0:
            print("  -> Skipped (Could not determine duration)")
            continue

        # 2. Analyze Stream
        print(f"  -> Scanning stream ({total_frames} frames)... ", end='', flush=True)
        errors = analyze_video(file_path, total_frames)
        
        if not errors:
            print("OK")
        else:
            print(f"CORRUPT ({len(errors)} errors)")
            # 3. Generate Visualization
            img_path = generate_visualization(file_path, total_frames, errors, REPORT_DIR)
            print(f"  -> Visualization saved: {img_path}")

    print("-" * 60)
    print("Process complete.")

if __name__ == "__main__":
    target_dir = "."
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    
    if not os.path.isdir(target_dir):
        print("Error: Invalid directory path.")
    else:
        scan_directory(target_dir)