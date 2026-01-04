#!/usr/bin/env python3
"""
Integrate Grounded-SAM-2 tracking with DiffuEraser for AprilTag removal.
This script runs Grounded-SAM-2 tracking to detect AprilTags and then uses
DiffuEraser to remove them from the video.
"""
from __future__ import annotations

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
import torch
import gc
from tqdm import tqdm
import ipdb

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "third_party" / "Grounded-SAM-2"))
sys.path.insert(0, str(repo_root / "third_party" / "DiffuEraser"))

# Import modules that are used in functions
import supervision as sv
from torchvision.ops import box_convert
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device
from pupil_apriltags import Detector

def create_mask_video_from_segments(
    frames_dir: Path, 
    video_segments: Dict[int, Dict[int, np.ndarray]], 
    output_mask_path: Path,
    video_info: Dict = None
) -> None:
    """
    Create a mask video from segmentation results.
    
    Args:
        frames_dir: Directory containing original video frames
        video_segments: Segmentation results from SAM2
        output_mask_path: Path to save the mask video
        video_info: Video information (optional, will be inferred from frames if not provided)
    """
    # Get frame names and sort them
    frame_names = [
        p.name for p in frames_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort(key=lambda p: int(Path(p).stem))
    
    if not frame_names:
        raise RuntimeError(f"No frames found in {frames_dir}")
    
    # Get frame dimensions from the first frame if video_info not provided
    if video_info is None:
        first_frame_path = frames_dir / frame_names[0]
        first_frame = cv2.imread(str(first_frame_path))
        if first_frame is None:
            raise RuntimeError(f"Could not read first frame: {first_frame_path}")
        height, width = first_frame.shape[:2]
        if height <= 0 or width <= 0:
            raise RuntimeError(f"Invalid frame dimensions: height={height}, width={width}")
        fps = 30  # Default FPS, adjust as needed
    else:
        height = video_info['height']
        width = video_info['width']
        if height <= 0 or width <= 0:
            raise RuntimeError(f"Invalid video_info dimensions: height={height}, width={width}")
        fps = video_info.get('fps', 30)
    
    # Initialize video writer (mp4v for .mp4 output consistency)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions: width={width}, height={height}")
    video_writer = cv2.VideoWriter(
        str(output_mask_path), fourcc, fps, (width, height), isColor=True
    )
    if not video_writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {output_mask_path}")
    
    # Process each frame
    for frame_idx, frame_name in enumerate(tqdm(frame_names, desc="Creating mask video")):
        # Create blank mask for this frame
        if height <= 0 or width <= 0:
            raise RuntimeError(f"Invalid frame dimensions: height={height}, width={width}")
        mask_frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Check if this frame has segmentation results
        if frame_idx in video_segments:
            frame_masks = video_segments[frame_idx]

            # Combine all object masks for this frame
            if height <= 0 or width <= 0:
                raise RuntimeError(f"Invalid combined mask dimensions: height={height}, width={width}")
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            for obj_id, mask in frame_masks.items():
                # Ensure mask is valid
                if mask is None or mask.size == 0:
                    # Create an empty mask with correct dimensions
                    mask_resized = np.zeros((height, width), dtype=np.uint8)
                    continue

                # Ensure mask has the right number of dimensions
                # Handle different dimensionalities: (H, W) or (1, H, W) or (H, W, 1)
                if mask.ndim == 3:
                    if mask.shape[0] == 1:
                        # Shape is (1, H, W), squeeze the first dimension
                        mask = mask.squeeze(0)
                    elif mask.shape[2] == 1:
                        # Shape is (H, W, 1), squeeze the last dimension
                        mask = mask.squeeze(2)
                    else:
                        # Unexpected 3D shape, take the first channel
                        mask = mask[0]
                elif mask.ndim < 2:
                    # Create an empty mask with correct dimensions
                    mask_resized = np.zeros((height, width), dtype=np.uint8)
                    continue

                # Ensure mask is the right size and shape
                if mask.shape[0] != height or mask.shape[1] != width:
                    # Check if mask is empty or has invalid dimensions
                    if mask.size == 0 or width <= 0 or height <= 0:
                        # Create an empty mask with correct dimensions
                        if width <= 0 or height <= 0:
                            print(f"Warning: Invalid dimensions for resize - width: {width}, height: {height}")
                        mask_resized = np.zeros((max(1, height), max(1, width)), dtype=np.uint8)
                    else:
                        # Additional check to ensure width and height are valid for cv2.resize
                        target_size = (int(width), int(height))
                        if target_size[0] <= 0 or target_size[1] <= 0:
                            print(f"Warning: Invalid target size for resize - {target_size}")
                            mask_resized = np.zeros((height, width), dtype=np.uint8)
                        else:
                            # Resize mask to match frame size
                            mask_resized = cv2.resize(
                                mask.astype(np.uint8),
                                target_size,
                                interpolation=cv2.INTER_NEAREST
                            )
                else:
                    mask_resized = mask.astype(np.uint8)

                # Convert boolean mask to uint8 if needed
                if mask_resized.dtype == bool:
                    mask_resized = mask_resized.astype(np.uint8) * 255
                elif mask_resized.dtype == np.float32 or mask_resized.dtype == np.float64:
                    mask_resized = (mask_resized * 255).astype(np.uint8)

                # Ensure mask values are in proper range for visualization
                if mask_resized.dtype == np.uint8:
                    # Ensure mask values are either 0 or 255 for clear visibility
                    mask_resized = np.where(mask_resized > 0, 255, 0).astype(np.uint8)
                elif mask_resized.dtype == bool:
                    # Convert boolean mask to 0/255
                    mask_resized = np.where(mask_resized, 255, 0).astype(np.uint8)
                else:
                    # For float types, ensure proper conversion
                    mask_resized = np.where(mask_resized > 0, 255, 0).astype(np.uint8)

                # Ensure mask_resized has the same shape as combined_mask before combining
                if mask_resized.shape != combined_mask.shape:
                    # This should not happen if the resize logic above works correctly
                    # But adding this as an extra safety check
                    target_height, target_width = combined_mask.shape
                    if mask_resized.shape[0] != target_height or mask_resized.shape[1] != target_width:
                        # Check if target dimensions are valid for cv2.resize
                        if target_width <= 0 or target_height <= 0:
                            print(f"Warning: Invalid target dimensions for resize - width: {target_width}, height: {target_height}")
                            # Create a mask with the correct shape filled with zeros
                            mask_resized = np.zeros((target_height, target_width), dtype=mask_resized.dtype)
                        else:
                            # Resize mask_resized to match combined_mask dimensions
                            mask_resized = cv2.resize(
                                mask_resized,
                                (target_width, target_height),
                                interpolation=cv2.INTER_NEAREST
                            )

                # Combine with existing mask
                combined_mask = np.maximum(combined_mask, mask_resized)

            # Apply combined mask to all 3 channels (RGB)
            mask_frame[:, :, 0] = combined_mask  # Red channel
            mask_frame[:, :, 1] = combined_mask  # Green channel
            mask_frame[:, :, 2] = combined_mask  # Blue channel

        # Write mask frame to video
        if mask_frame.size == 0:
            raise RuntimeError(f"Mask frame is empty for frame {frame_idx}")
        video_writer.write(mask_frame)

        # Clean up variables to free memory
        del mask_frame, combined_mask

    video_writer.release()
    print(f"Mask video saved to {output_mask_path}")

    # Explicitly clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def resize_frame_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Resize frame to fit max_width while keeping aspect ratio."""
    if max_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_video_from_frames_dir(
    frames_dir: Path,
    output_video_path: Path,
    fps: float
) -> None:
    """Create a video from a directory of frames with a specific fps."""
    frame_names = [
        p.name for p in frames_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort(key=lambda p: int(Path(p).stem))

    if not frame_names:
        raise RuntimeError(f"No frames found in {frames_dir}")

    first_frame = cv2.imread(str(frames_dir / frame_names[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frames_dir / frame_names[0]}")
    height, width = first_frame.shape[:2]
    if height <= 0 or width <= 0:
        raise RuntimeError(f"Invalid frame dimensions: height={height}, width={width}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not video_writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {output_video_path}")

    for frame_name in tqdm(frame_names, desc="Creating chunk video"):
        frame = cv2.imread(str(frames_dir / frame_name))
        if frame is None:
            raise RuntimeError(f"Could not read frame: {frames_dir / frame_name}")
        video_writer.write(frame)

    video_writer.release()


def concatenate_videos(
    video_paths: List[Path],
    output_path: Path,
    fps: float
) -> None:
    """Concatenate multiple videos (same resolution) into a single output."""
    if not video_paths:
        raise RuntimeError("No chunk videos provided for concatenation")

    first_cap = cv2.VideoCapture(str(video_paths[0]))
    if not first_cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_paths[0]}")
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions: width={width}, height={height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {output_path}")

    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()

    writer.release()


def detect_apriltag_centers(image_bgr: np.ndarray, family: str) -> list[tuple[int, float, float]]:
    """
    Detect AprilTag centers in an image using pupil_apriltags library.

    Args:
        image_bgr: Input image in BGR format
        family: AprilTag family (e.g. 16h5, 25h9, 36h11, 41h12)

    Returns:
        List of tuples containing (tag_id, center_x, center_y)
    """
    try:
        detector = Detector(families=family)
    except Exception as exc:
        raise RuntimeError("pupil_apriltags not available; install pupil-apriltags") from exc

    family_norm = family.strip()
    family_lower = family_norm.lower()
    if family_lower in {"41h12", "tag41h12"}:
        family_norm = "tagStandard41h12"
    elif not family_lower.startswith("tag"):
        family_norm = f"tag{family_norm}"

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector = Detector(families=family_norm)
    detections = detector.detect(gray)
    results: list[tuple[int, float, float]] = []
    if not detections:
        return results
    for det in detections:
        cx, cy = det.center
        results.append((int(det.tag_id), float(cx), float(cy)))
    return results


def run_grounded_sam2_tracking_chunked(
    video_path: Path,
    prompt: str,
    frames_dir: Path,
    results_dir: Path,
    output_video: Path,
    device: str = "auto",
    use_apriltag_points: bool = False,
    apriltag_family: str = "36h11",
    prompt_type: str = "mask",
    ann_frame_idx: int = 0,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    stride: int = 1,
    max_frames: int = 0,
    chunk_size: int = 50,  # Smaller chunk size for better memory management
    chunk_output_dir: Path | None = None,
    max_width: int = 0,
    dynamic_apriltag_prompts: bool = False
) -> List[Dict[str, str]]:
    """
    Run Grounded-SAM-2 tracking with chunked processing for long videos.
    Uses sv.ImageSink to extract all frames first, then processes in chunks.
    Includes retry mechanism and progress saving.
    Returns chunk metadata for downstream processing.
    """
    import tempfile
    import shutil
    import math
    import time
    import json

    # Get video info
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    total_frames = video_info.total_frames
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    print(f"Processing long video: {total_frames} frames in chunks of {chunk_size}")

    # Use sv.ImageSink to extract all frames to the frames_dir
    print(f"Extracting all video frames to: {frames_dir}")
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Get all frames from video
    frame_generator = sv.get_video_frames_generator(
        str(video_path), stride=stride, start=0, end=max_frames if max_frames > 0 else None
    )

    with sv.ImageSink(
        target_dir_path=frames_dir,
        overwrite=True,
        image_name_pattern="{:05d}.jpg",
    ) as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            frame = resize_frame_max_width(frame, max_width)
            sink.save_image(frame)

    # Get all frame names
    frame_names = [
        p.name for p in frames_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort(key=lambda p: int(Path(p).stem))

    if not frame_names:
        raise RuntimeError(f"No frames found in {frames_dir}")

    # Calculate number of chunks
    num_chunks = math.ceil(len(frame_names) / chunk_size)

    if chunk_output_dir is None:
        chunk_output_dir = results_dir.parent / "chunks"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    # Define progress/manifest paths
    progress_file = chunk_output_dir / "progress.json"
    manifest_file = chunk_output_dir / "manifest.json"

    # Load existing progress if available
    start_chunk_idx = 0
    manifest_data: Dict[str, List[Dict[str, str]]] = {"chunks": []}

    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                start_chunk_idx = progress_data.get('last_completed_chunk', 0)
                if manifest_file.exists():
                    with open(manifest_file, 'r') as mf:
                        manifest_data = json.load(mf)
                print(f"Resuming from chunk {start_chunk_idx + 1}, previously completed {start_chunk_idx} chunks")
        except Exception as e:
            print(f"Could not load progress file: {e}, starting from beginning")
            start_chunk_idx = 0
            manifest_data = {"chunks": []}

    for chunk_idx in range(start_chunk_idx, num_chunks):
        print(f"Processing chunk {chunk_idx + 1}/{num_chunks}")

        # Calculate start and end indices for this chunk
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(frame_names))

        # Get frame names for this chunk
        chunk_frame_names = frame_names[start_idx:end_idx]
        print(f"Current chunk contains {len(chunk_frame_names)} frames")

        # Create temporary directory for processing this chunk
        with tempfile.TemporaryDirectory() as temp_chunk_dir:
            temp_chunk_dir = Path(temp_chunk_dir)

            # Copy frames for this chunk to temporary directory with sequential numbering
            for i, frame_name in enumerate(chunk_frame_names):
                src_path = frames_dir / frame_name
                dst_path = temp_chunk_dir / f"{i:05d}.jpg"
                shutil.copy2(src_path, dst_path)

            first_chunk_frame = cv2.imread(str(temp_chunk_dir / "00000.jpg"))
            if first_chunk_frame is None:
                raise RuntimeError(f"Could not read chunk frame: {temp_chunk_dir / '00000.jpg'}")
            chunk_height, chunk_width = first_chunk_frame.shape[:2]

            # Process this chunk using the existing tracking function with retry mechanism
            max_retries = 3
            chunk_segments = None
            for attempt in range(max_retries):
                try:
                    print(f"Processing chunk {chunk_idx + 1}, attempt {attempt + 1}")

                    # Process this chunk using the existing tracking function
                    chunk_output_video = temp_chunk_dir / f"tracking_chunk_{chunk_idx}.mp4"
                    chunk_results_dir = temp_chunk_dir / "results"

                    chunk_segments = run_grounded_sam2_tracking(
                        video_path=None,  # No video path needed since we're using frames from directory
                        prompt=prompt,
                        frames_dir=temp_chunk_dir,
                        results_dir=chunk_results_dir,
                        output_video=chunk_output_video,
                        device=device,
                        use_apriltag_points=use_apriltag_points,
                        apriltag_family=apriltag_family,
                        prompt_type=prompt_type,
                        ann_frame_idx=0,  # Use first frame of chunk
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        stride=1,  # Frames are already extracted
                        max_frames=0,  # Process all frames in chunk
                        chunk_size=chunk_size,  # Pass through for nested calls
                        dynamic_apriltag_prompts=dynamic_apriltag_prompts
                    )
                    print(f"Chunk {chunk_idx + 1} processed successfully on attempt {attempt + 1}, got {len(chunk_segments)} frames")
                    break  # Success, exit retry loop
                except Exception as e:
                    print(f"Attempt {attempt + 1} for chunk {chunk_idx + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying chunk {chunk_idx + 1} in 5 seconds...")
                        time.sleep(5)
                        # Clear memory before retry
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        print(f"All {max_retries} attempts failed for chunk {chunk_idx + 1}. Raising error.")
                        raise e

            # Persist chunk artifacts and manifest
            chunk_video_path = chunk_output_dir / f"video_{chunk_idx:05d}.mp4"
            chunk_mask_path = chunk_output_dir / f"mask_{chunk_idx:05d}.mp4"

            create_video_from_frames_dir(
                frames_dir=temp_chunk_dir,
                output_video_path=chunk_video_path,
                fps=video_info.fps
            )
            create_mask_video_from_segments(
                frames_dir=temp_chunk_dir,
                video_segments=chunk_segments,
                output_mask_path=chunk_mask_path,
                video_info={"height": chunk_height, "width": chunk_width, "fps": video_info.fps}
            )

            manifest_data["chunks"].append({
                "index": chunk_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "video_path": str(chunk_video_path),
                "mask_path": str(chunk_mask_path)
            })
            with open(manifest_file, 'w') as mf:
                json.dump(manifest_data, mf)

            # Save progress after each chunk
            progress_data = {
                'last_completed_chunk': chunk_idx + 1,
                'total_chunks': num_chunks,
                'completed_chunks': chunk_idx + 1
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f)

            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()

            # Force garbage collection for large objects
            if 'temp_chunk_dir' in locals() and locals()['temp_chunk_dir'] is not None:
                # temp_chunk_dir is automatically cleaned up by the with statement
                pass

    # Remove progress file after successful completion
    if progress_file.exists():
        progress_file.unlink()

    return manifest_data.get("chunks", [])


def run_grounded_sam2_tracking(
    video_path: Path,
    prompt: str,
    frames_dir: Path,
    results_dir: Path,
    output_video: Path,
    device: str = "auto",
    use_apriltag_points: bool = False,
    apriltag_family: str = "36h11",
    prompt_type: str = "mask",
    ann_frame_idx: int = 0,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    stride: int = 1,
    max_frames: int = 0,
    chunk_size: int = 100,  # Add chunk size parameter for memory management
    max_width: int = 0,
    dynamic_apriltag_prompts: bool = False
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Run Grounded-SAM-2 tracking and return segmentation results.
    This is a simplified version that returns the segmentation data without
    fully implementing the tracking logic.
    """
    # Set offline mode for transformers to avoid online access
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Define paths to model checkpoints
    GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
    SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Change to Grounded-SAM-2 directory
    base_dir = repo_root / "third_party" / "Grounded-SAM-2"
    os.chdir(base_dir)
    sys.path.insert(0, str(base_dir))

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SAM2 models (always needed) with half precision if using CUDA
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    video_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT,
                                                 device=device,
                                                 dtype=torch_dtype)
    sam2_image_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT,
                                  device=device,
                                  dtype=torch_dtype)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # If video_path is provided, extract frames from video
    if video_path is not None:
        # Extract frames from video
        video_info = sv.VideoInfo.from_video_path(str(video_path))
        print(f"Video info: {video_info}")

        # Calculate total frames to process
        total_frames = video_info.total_frames
        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        # Process video in chunks to manage memory
        frame_generator = sv.get_video_frames_generator(
            str(video_path), stride=stride, start=0, end=None
        )
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Process frames in chunks
        saved = 0
        with sv.ImageSink(
            target_dir_path=frames_dir,
            overwrite=True,
            image_name_pattern="{:05d}.jpg",
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                frame = resize_frame_max_width(frame, max_width)
                sink.save_image(frame)
                saved += 1
                if max_frames > 0 and saved >= max_frames:
                    break

    # Get frame names
    frame_names = [
        p.name
        for p in frames_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(Path(p).stem))

    if not frame_names:
        raise RuntimeError(f"No JPEG frames found in {frames_dir}")

    if ann_frame_idx < 0 or ann_frame_idx >= len(frame_names):
        raise ValueError(f"ann-frame-idx {ann_frame_idx} out of range (0..{len(frame_names)-1})")

    # Initialize video predictor state
    inference_state = video_predictor.init_state(video_path=str(frames_dir))
    ann_frame_idx = 0  # Use first frame for annotation

    # Detect objects in the annotation frame
    img_path = frames_dir / frame_names[ann_frame_idx]
    objects: list[str] = []
    tag_to_obj_id: Dict[int, int] = {}
    next_obj_id = 1

    if use_apriltag_points:
        image_source = cv2.imread(str(img_path))
        if image_source is None:
            print(f"[error] failed to read frame: {img_path}")
            return {}
        tag_centers = detect_apriltag_centers(image_source, apriltag_family)
        if not tag_centers:
            print("[warning] no AprilTags detected; adjust frame index or family")
            return {}
        for object_id, (tag_id, cx, cy) in enumerate(tag_centers, start=1):
            points = np.array([[cx, cy]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
            objects.append(f"apriltag_{tag_id}")
            tag_to_obj_id[int(tag_id)] = object_id
        if tag_to_obj_id:
            next_obj_id = max(tag_to_obj_id.values()) + 1
    else:
        # Only load GroundingDINO model when not using AprilTag points
        grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=device,
        )
        # Set model to half precision if using CUDA
        if device.startswith("cuda"):
            # Check if the model has a half method before calling it
            if hasattr(grounding_model, 'half'):
                grounding_model = grounding_model.half()

        image_source, image = load_image(str(img_path))

        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if boxes is None or len(boxes) == 0:
            print("[warning] no boxes detected; adjust prompt or thresholds")
            return {}

        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        class_names = labels

        image_predictor.set_image(image_source)
        objects = class_names

        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Move tensors to appropriate device and dtype if needed
        if device.startswith("cuda"):
            if isinstance(masks, torch.Tensor):
                masks = masks.to(device=device, dtype=torch.float16)
            if isinstance(logits, torch.Tensor):
                logits = logits.to(device=device, dtype=torch.float16)

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Add masks to video predictor based on prompt type
        if prompt_type == "point":
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
            for object_id, points in enumerate(all_sample_points, start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
        elif prompt_type == "box":
            for object_id, box in enumerate(input_boxes, start=1):
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        elif prompt_type == "mask":
            for object_id, mask in enumerate(masks, start=1):
                labels = np.ones((1), dtype=np.int32)
                video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask,
                )
        else:
            raise NotImplementedError("Unsupported prompt type")

    # For long videos, we need to implement a chunked processing approach
    # However, SAM2 tracking is inherently sequential, so we'll process with memory optimization
    video_segments = {}

    print("Starting video propagation...")

    # Process tracking with memory management by immediately moving results to CPU
    # This is the best we can do without modifying the SAM2 core functionality
    try:
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
            inference_state
        ):
            # Immediately convert to appropriate dtype and move to CPU to save GPU memory
            if device.startswith("cuda"):
                out_mask_logits = out_mask_logits.to(dtype=torch.float16).cpu()

            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            if dynamic_apriltag_prompts:
                img_path = frames_dir / frame_names[out_frame_idx]
                image_source = cv2.imread(str(img_path))
                if image_source is not None:
                    tag_centers = detect_apriltag_centers(image_source, apriltag_family)
                    for tag_id, cx, cy in tag_centers:
                        obj_id = tag_to_obj_id.get(int(tag_id))
                        if obj_id is None:
                            obj_id = next_obj_id
                            next_obj_id += 1
                            tag_to_obj_id[int(tag_id)] = obj_id
                        points = np.array([[cx, cy]], dtype=np.float32)
                        labels = np.array([1], dtype=np.int32)
                        video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=out_frame_idx,
                            obj_id=obj_id,
                            points=points,
                            labels=labels,
                        )

            # Periodic memory cleanup
            if out_frame_idx % chunk_size == 0 and out_frame_idx > 0:
                print(f"Processed frame {out_frame_idx}, performing memory cleanup")
                torch.cuda.empty_cache()
                gc.collect()

                # Force garbage collection for large objects
                # We can safely delete out_mask_logits as it's a local variable
                pass
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Out of memory error occurred at frame {len(video_segments)}.")
            print("Consider reducing the video length or using a smaller model.")
            raise
        else:
            raise

    # Create annotated output video
    results_dir.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    id_to_objects = {i: obj for i, obj in enumerate(objects, start=1)}

    for frame_idx, segments in video_segments.items():
        img = cv2.imread(str(frames_dir / frame_names[frame_idx]))
        object_ids = list(segments.keys())
        masks = list(segments.values())
        if masks:
            masks = np.concatenate(masks, axis=0)

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),
                mask=masks,
                class_id=np.array(object_ids, dtype=np.int32),
            )
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(
                annotated_frame,
                detections=detections,
                labels=[id_to_objects[i] for i in object_ids],
            )
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        else:
            annotated_frame = img.copy()

        cv2.imwrite(
            str(results_dir / f"annotated_frame_{frame_idx:05d}.jpg"),
            annotated_frame,
        )

        # Clean up memory after processing each frame
        del img, segments, object_ids, masks, detections, annotated_frame
        if frame_idx % 50 == 0:  # More frequent cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Create output video from annotated frames
    create_video_from_images(str(results_dir), str(output_video))

    # Explicitly clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return video_segments


def run_diffueraser(
    input_video: Path,
    input_mask: Path,
    output_dir: Path,
    base_model_path: str = "weights/stable-diffusion-v1-5",
    vae_path: str = "weights/sd-vae-ft-mse",
    diffueraser_path: str = "weights/diffuEraser",
    propainter_model_dir: str = "weights/propainter",
    chunk_size: int = 50,  # Add chunk size parameter for memory management
    output_path: Path | None = None
) -> Path:
    """
    Run DiffuEraser with the given video and mask.

    Args:
        input_video: Path to input video
        input_mask: Path to mask video
        output_dir: Directory to save results
        base_model_path: Path to Stable Diffusion base model
        vae_path: Path to VAE model
        diffueraser_path: Path to DiffuEraser model
        propainter_model_dir: Path to Propainter model

    Returns:
        Path to the output video with AprilTags removed
    """
    
    # Initialize device
    device = get_device()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths for intermediate results
    priori_path = output_dir / "priori.mp4"
    if output_path is None:
        output_path = output_dir / "diffueraser_result.mp4"
    
    # Initialize models
    # Set offline mode for Hugging Face to ensure local files are used
    os.environ['HF_HUB_OFFLINE'] = '1'

    # Change to the DiffuEraser directory to ensure relative paths work correctly
    original_cwd = os.getcwd()
    diffueraser_dir = Path(__file__).parent.parent / "third_party" / "DiffuEraser"
    os.chdir(diffueraser_dir)

    try:
        ckpt = "2-Step"
        # Initialize DiffuEraser with half precision if using CUDA
        video_inpainting_sd = DiffuEraser(device, base_model_path, vae_path, diffueraser_path, ckpt=ckpt)
        # Set model to half precision if using CUDA
        if str(device).startswith("cuda"):
            # Check if the pipeline has a half method before calling it
            if hasattr(video_inpainting_sd.pipeline, 'half'):
                video_inpainting_sd.pipeline = video_inpainting_sd.pipeline.half()
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

    # Change to the DiffuEraser directory for Propainter as well
    os.chdir(diffueraser_dir)
    try:
        propainter = Propainter(propainter_model_dir, device=device)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
    
    # Generate priori using Propainter with chunk processing
    os.chdir(diffueraser_dir)
    try:
        video_info = sv.VideoInfo.from_video_path(str(input_video))
        fps = video_info.fps if video_info.fps > 0 else 30.0
        total_frames = video_info.total_frames
        video_length_seconds = max(1.0, total_frames / fps) if total_frames > 0 else 1.0

        propainter.forward(
            str(input_video), str(input_mask), str(priori_path),
            video_length=video_length_seconds,
            ref_stride=min(10, chunk_size // 2),  # Adjust stride based on chunk size
            neighbor_length=min(10, chunk_size // 2),
            subvideo_length=min(50, chunk_size),  # Use chunk size as max subvideo length
            mask_dilation=8,
            save_fps=fps
        )
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

    # Run DiffuEraser with chunk processing
    os.chdir(diffueraser_dir)
    try:
        guidance_scale = None  # Default value is 0

        video_info = sv.VideoInfo.from_video_path(str(input_video))
        fps = video_info.fps if video_info.fps > 0 else 30.0
        total_frames = video_info.total_frames
        video_length_seconds = max(1.0, total_frames / fps) if total_frames > 0 else 1.0

        video_inpainting_sd.forward(
            str(input_video), str(input_mask), str(priori_path), str(output_path),
            max_img_size=960,
            video_length=video_length_seconds,
            mask_dilation_iter=8,
            guidance_scale=guidance_scale
        )
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

    # Explicitly clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return output_path


def run_diffueraser_chunked(
    chunks: List[Dict[str, str]],
    output_dir: Path,
    base_model_path: Path,
    vae_path: Path,
    diffueraser_path: Path,
    propainter_model_dir: Path,
    chunk_size: int,
    fps: float
) -> Path:
    """
    Run DiffuEraser per chunk using saved chunk videos and masks, then concatenate outputs.
    """
    if not chunks:
        raise RuntimeError("No chunk metadata provided for DiffuEraser processing")

    chunk_output_dir = output_dir / "diffueraser_chunks"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    chunks_sorted = sorted(chunks, key=lambda c: int(c["index"]))
    chunk_results: List[Path] = []
    for chunk in chunks_sorted:
        chunk_index = int(chunk["index"])
        chunk_video = Path(chunk["video_path"])
        chunk_mask = Path(chunk["mask_path"])
        if not chunk_video.exists():
            raise RuntimeError(f"Chunk video not found: {chunk_video}")
        if not chunk_mask.exists():
            raise RuntimeError(f"Chunk mask not found: {chunk_mask}")

        chunk_dir = chunk_output_dir / f"chunk_{chunk_index:05d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_output_path = chunk_dir / "diffueraser_result.mp4"

        result_path = run_diffueraser(
            input_video=chunk_video,
            input_mask=chunk_mask,
            output_dir=chunk_dir,
            base_model_path=str(base_model_path),
            vae_path=str(vae_path),
            diffueraser_path=str(diffueraser_path),
            propainter_model_dir=str(propainter_model_dir),
            chunk_size=chunk_size,
            output_path=chunk_output_path
        )
        chunk_results.append(result_path)

    output_path = output_dir / "diffueraser_result.mp4"
    concatenate_videos(chunk_results, output_path, fps)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Integrate Grounded-SAM-2 tracking with DiffuEraser for AprilTag removal."
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--prompt", required=True, help="Text prompt for object detection (e.g. 'AprilTag')")
    parser.add_argument("--output-dir", default="./outputs/apriltag_removal",
                        help="Directory to save all results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Force device selection (default: auto)")
    parser.add_argument("--box-threshold", type=float, default=0.35,
                        help="Grounding-DINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Grounding-DINO text threshold")
    parser.add_argument("--stride", type=int, default=1,
                        help="Frame stride when extracting video frames")
    parser.add_argument("--max-width", type=int, default=0,
                        help="Resize frames to fit this width (0 = no resize)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit number of extracted frames (0 = no limit)")
    parser.add_argument("--use-apriltag-points", action="store_true",
                        help="Use AprilTag centers as point prompts instead of GroundingDINO (pupil_apriltags)")
    parser.add_argument("--apriltag-family", default="36h11",
                        help="AprilTag family (e.g. 16h5, 25h9, 36h11, 41h12)")
    parser.add_argument("--dynamic-apriltag-prompts", action="store_true",
                        help="Detect AprilTags per frame and add centers as prompts during propagation")
    parser.add_argument("--tracking-chunk-size", type=int, default=100,
                        help="Chunk size for tracking memory management (default: 100)")
    parser.add_argument("--diffueraser-chunk-size", type=int, default=50,
                        help="Chunk size for DiffuEraser memory management (default: 50)")

    args = parser.parse_args()
    
    # Define paths
    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    
    if not video_path.exists():
        print(f"[error] video not found: {video_path}")
        return 1
    
    # Create temporary directories
    frames_dir = output_dir / "frames"
    results_dir = output_dir / "annotated_frames"
    tracking_output = output_dir / "tracking_result.mp4"
    mask_video_path = output_dir / "mask.mp4"
    chunks_dir = output_dir / "chunks"
    resized_video_path = output_dir / "video_resized.mp4"
    
    print("Step 1: Running Grounded-SAM-2 tracking...")
    # Run SAM2 tracking to detect AprilTags
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    total_frames = video_info.total_frames
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    # Use chunked processing for long videos
    if total_frames > args.tracking_chunk_size:
        print(f"Video has {total_frames} frames, using chunked processing with chunk size {args.tracking_chunk_size}")
        chunks = run_grounded_sam2_tracking_chunked(
            video_path=video_path,
            prompt=args.prompt,
            frames_dir=frames_dir,
            results_dir=results_dir,
            output_video=tracking_output,
            device=args.device,
            use_apriltag_points=args.use_apriltag_points,
            apriltag_family=args.apriltag_family,
            prompt_type="mask",
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            stride=args.stride,
            max_frames=args.max_frames,
            chunk_size=args.tracking_chunk_size,  # Use chunk size from command line for memory management
            max_width=args.max_width,
            dynamic_apriltag_prompts=args.dynamic_apriltag_prompts,
            chunk_output_dir=chunks_dir
        )
        print(f"Chunked processing completed, got {len(chunks)} chunks")

        # Clean up memory after chunked processing
        torch.cuda.empty_cache()
        gc.collect()
    else:
        video_segments = run_grounded_sam2_tracking(
            video_path=video_path,
            prompt=args.prompt,
            frames_dir=frames_dir,
            results_dir=results_dir,
            output_video=tracking_output,
            device=args.device,
            use_apriltag_points=args.use_apriltag_points,
            apriltag_family=args.apriltag_family,
            prompt_type="mask",
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            stride=args.stride,
            max_frames=args.max_frames,
            chunk_size=args.tracking_chunk_size,  # Use chunk size from command line for memory management
            max_width=args.max_width,
            dynamic_apriltag_prompts=args.dynamic_apriltag_prompts
        )
        print(f"Direct processing completed, got {len(video_segments)} total frames")
        print("Step 2: Creating mask video from tracking results...")
        # Create mask video from segmentation results
        first_frame = cv2.imread(str(frames_dir / "00000.jpg"))
        if first_frame is None:
            raise RuntimeError(f"Could not read first frame: {frames_dir / '00000.jpg'}")
        resized_height, resized_width = first_frame.shape[:2]
        create_mask_video_from_segments(
            frames_dir=frames_dir,
            video_segments=video_segments,
            output_mask_path=mask_video_path,
            video_info={"height": resized_height, "width": resized_width, "fps": video_info.fps}
        )

        # Clean up memory after creating mask video
        torch.cuda.empty_cache()
        gc.collect()

    print("Step 3: Running DiffuEraser for AprilTag removal...")
    # Run DiffuEraser with the generated mask
    # Set correct paths for the weights (using absolute paths to avoid HF repo ID validation)
    project_root = Path(__file__).parent.parent
    base_model_path = project_root / "third_party" / "DiffuEraser" / "weights" / "stable-diffusion-v1-5"
    vae_path = project_root / "third_party" / "DiffuEraser" / "weights" / "sd-vae-ft-mse"
    diffueraser_path = project_root / "third_party" / "DiffuEraser" / "weights" / "diffuEraser"
    propainter_model_dir = project_root / "third_party" / "DiffuEraser" / "weights" / "propainter"

    if total_frames > args.tracking_chunk_size:
        diffueraser_output = run_diffueraser_chunked(
            chunks=chunks,
            output_dir=output_dir,
            base_model_path=base_model_path,
            vae_path=vae_path,
            diffueraser_path=diffueraser_path,
            propainter_model_dir=propainter_model_dir,
            chunk_size=args.diffueraser_chunk_size,
            fps=video_info.fps
        )
    else:
        input_video = video_path
        if args.max_width > 0:
            create_video_from_frames_dir(
                frames_dir=frames_dir,
                output_video_path=resized_video_path,
                fps=video_info.fps
            )
            input_video = resized_video_path
        diffueraser_output = run_diffueraser(
            input_video=input_video,
            input_mask=mask_video_path,
            output_dir=output_dir,
            base_model_path=str(base_model_path),
            vae_path=str(vae_path),
            diffueraser_path=str(diffueraser_path),
            propainter_model_dir=str(propainter_model_dir),
            chunk_size=args.diffueraser_chunk_size  # Use chunk size from command line for memory management
        )
    
    print(f"Process completed! Results saved to {output_dir}")
    print(f"- Tracking result: {tracking_output}")
    if total_frames <= args.tracking_chunk_size:
        print(f"- Mask video: {mask_video_path}")
    else:
        print(f"- Chunk artifacts: {chunks_dir}")
    print(f"- Final result (AprilTags removed): {diffueraser_output}")

    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
