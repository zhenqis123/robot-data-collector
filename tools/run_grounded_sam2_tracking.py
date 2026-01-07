#!/usr/bin/env python3
"""
Run Grounded-SAM-2 tracking with a custom video and prompt.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
third_party_root = repo_root / "third_party" / "Grounded-SAM-2"
if third_party_root.exists():
    sys.path.insert(0, str(third_party_root))

import cv2
import numpy as np
import supervision as sv
import torch
from torchvision.ops import box_convert
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images


GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Grounded-SAM-2 tracking on a custom video."
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--prompt", required=True, help="Text prompt, e.g. 'april tag on desk.'")
    parser.add_argument("--output-video", default="./outputs/tracking_demo.mp4",
                        help="Output video path")
    parser.add_argument("--frames-dir", default="./custom_video_frames",
                        help="Directory to write extracted frames")
    parser.add_argument("--results-dir", default="./tracking_results",
                        help="Directory to write annotated frames")
    parser.add_argument("--prompt-type", default="box", choices=["point", "box", "mask"],
                        help="Prompt type for SAM2 video predictor")
    parser.add_argument("--use-apriltag-points", action="store_true",
                        help="Use AprilTag centers as point prompts instead of GroundingDINO (pupil_apriltags)")
    parser.add_argument("--apriltag-family", default="36h11",
                        help="AprilTag family (e.g. 16h5, 25h9, 36h11, 41h12)")
    parser.add_argument("--ann-frame-idx", type=int, default=0,
                        help="Frame index (in saved frames) used to initialize prompts")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Force device selection (default: auto)")
    parser.add_argument("--box-threshold", type=float, default=0.35,
                        help="Grounding-DINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Grounding-DINO text threshold")
    parser.add_argument("--stride", type=int, default=1,
                        help="Frame stride when extracting video frames")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit number of extracted frames (0 = no limit)")
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return choice


def enable_cuda_fastpath() -> None:
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def extract_frames(video_path: Path, frames_dir: Path, stride: int, max_frames: int) -> None:
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    print(video_info)
    frame_generator = sv.get_video_frames_generator(
        str(video_path), stride=stride, start=0, end=None
    )
    frames_dir.mkdir(parents=True, exist_ok=True)
    with sv.ImageSink(
        target_dir_path=frames_dir,
        overwrite=True,
        image_name_pattern="{:05d}.jpg",
    ) as sink:
        saved = 0
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break


def list_frame_names(frames_dir: Path) -> list[str]:
    frame_names = [
        p.name
        for p in frames_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(Path(p).stem))
    if not frame_names:
        raise RuntimeError(f"No JPEG frames found in {frames_dir}")
    return frame_names


def detect_apriltag_centers(image_bgr: np.ndarray, family: str) -> list[tuple[int, float, float]]:
    try:
        from pupil_apriltags import Detector
    except ModuleNotFoundError as exc:
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


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    repo_root = Path(__file__).resolve().parent.parent
    base_dir = repo_root / "third_party" / "Grounded-SAM-2"
    if not base_dir.exists():
        print(f"[error] missing Grounded-SAM-2 at {base_dir}")
        return 1

    video_path = Path(args.video).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    frames_dir = Path(args.frames_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()

    if not video_path.exists():
        print(f"[error] video not found: {video_path}")
        return 1
    if args.stride < 1:
        print("[error] --stride must be >= 1")
        return 1

    os.chdir(base_dir)
    sys.path.insert(0, str(base_dir))

    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=device,
    )

    video_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)
    sam2_image_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    extract_frames(video_path, frames_dir, args.stride, args.max_frames)
    frame_names = list_frame_names(frames_dir)

    if args.ann_frame_idx < 0 or args.ann_frame_idx >= len(frame_names):
        print(f"[error] ann-frame-idx {args.ann_frame_idx} out of range (0..{len(frame_names)-1})")
        return 1

    inference_state = video_predictor.init_state(video_path=str(frames_dir))
    ann_frame_idx = args.ann_frame_idx

    if device == "cuda":
        enable_cuda_fastpath()

    img_path = frames_dir / frame_names[ann_frame_idx]
    objects: list[str] = []

    if args.use_apriltag_points:
        image_source = cv2.imread(str(img_path))
        if image_source is None:
            print(f"[error] failed to read frame: {img_path}")
            return 1
        tag_centers = detect_apriltag_centers(image_source, args.apriltag_family)
        if not tag_centers:
            print("[error] no AprilTags detected; adjust frame index or family")
            return 1
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
    else:
        image_source, image = load_image(str(img_path))

        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        if boxes is None or len(boxes) == 0:
            print("[error] no boxes detected; adjust prompt or thresholds")
            return 1

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

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        if args.prompt_type == "point":
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
        elif args.prompt_type == "box":
            for object_id, box in enumerate(input_boxes, start=1):
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        elif args.prompt_type == "mask":
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

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    results_dir.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    id_to_objects = {i: obj for i, obj in enumerate(objects, start=1)}

    for frame_idx, segments in video_segments.items():
        img = cv2.imread(str(frames_dir / frame_names[frame_idx]))
        object_ids = list(segments.keys())
        masks = list(segments.values())
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
        cv2.imwrite(
            str(results_dir / f"annotated_frame_{frame_idx:05d}.jpg"),
            annotated_frame,
        )

    create_video_from_images(str(results_dir), str(output_video))
    if not output_video.exists():
        print(f"[error] output video not created: {output_video}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
