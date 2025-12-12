import argparse
import time
import numpy as np
import cv2

from pupil_apriltags import Detector


def draw_tag_overlay(frame_bgr, det, color=(0, 255, 0)):
    """
    在图像上绘制检测框、角点、中心点、ID 等信息
    det.corners: (4,2) float
    det.center:  (2,)  float
    """
    corners = det.corners.astype(np.int32)
    center = tuple(np.round(det.center).astype(int))

    # 外框
    cv2.polylines(frame_bgr, [corners], True, color, 2, cv2.LINE_AA)

    # 角点
    for i, p in enumerate(corners):
        cv2.circle(frame_bgr, tuple(p), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame_bgr, str(i), (p[0] + 4, p[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # 中心点
    cv2.circle(frame_bgr, center, 4, (255, 0, 0), -1, cv2.LINE_AA)

    # ID 标签
    text = f"ID={det.tag_id}"
    # 把文字放在角点0附近
    p0 = corners[0]
    cv2.putText(frame_bgr, text, (p0[0], p0[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def project_points(K, rvec, tvec, pts3):
    """3D 点投影到图像平面"""
    pts2, _ = cv2.projectPoints(pts3, rvec, tvec, K, None)
    return pts2.reshape(-1, 2)


def estimate_pose_and_draw_axes(frame_bgr, det, tag_size_m, K, dist=None):
    """
    用 OpenCV solvePnP 做一个简单位姿估计，并绘制坐标轴
    需要你提供相机内参 K (3x3)；dist 可以为 None 或 0 向量
    """
    if dist is None:
        dist = np.zeros((5, 1), dtype=np.float64)

    # AprilTag 角点在 tag 坐标系下的 3D 坐标（tag 平面 z=0）
    s = tag_size_m / 2.0
    obj_pts = np.array([
        [-s, -s, 0],
        [ s, -s, 0],
        [ s,  s, 0],
        [-s,  s, 0],
    ], dtype=np.float64)

    img_pts = det.corners.astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return

    # 画坐标轴（单位：米）
    axis_len = tag_size_m * 0.5
    axes_3d = np.array([
        [0, 0, 0],                 # 原点
        [axis_len, 0, 0],          # X
        [0, axis_len, 0],          # Y
        [0, 0, -axis_len],         # Z（朝相机方向画负号更直观）
    ], dtype=np.float64)

    axes_2d = project_points(K, rvec, tvec, axes_3d).astype(int)
    o = tuple(axes_2d[0])
    x = tuple(axes_2d[1])
    y = tuple(axes_2d[2])
    z = tuple(axes_2d[3])

    cv2.line(frame_bgr, o, x, (0, 0, 255), 3, cv2.LINE_AA)   # X 红
    cv2.line(frame_bgr, o, y, (0, 255, 0), 3, cv2.LINE_AA)   # Y 绿
    cv2.line(frame_bgr, o, z, (255, 0, 0), 3, cv2.LINE_AA)   # Z 蓝

    # 显示 tvec（米）
    txt = f"t=({tvec[0,0]:.3f},{tvec[1,0]:.3f},{tvec[2,0]:.3f})m"
    cv2.putText(frame_bgr, txt, (o[0] + 5, o[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--family", type=str, default="tagStandard41h12")
    ap.add_argument("--decimate", type=float, default=1.0, help="quad_decimate (>=1.0). bigger=faster but less accurate")
    ap.add_argument("--blur", type=float, default=0.0, help="quad_sigma, gaussian blur sigma")
    ap.add_argument("--sharpen", type=float, default=0.25, help="decode_sharpening")
    ap.add_argument("--refine", action="store_true", help="refine_edges")
    ap.add_argument("--pose", action="store_true", help="enable pose drawing (requires --fx --fy --cx --cy and --tagsize)")
    ap.add_argument("--tagsize", type=float, default=0.05, help="tag size in meters (e.g. 0.05 for 50mm)")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    args = ap.parse_args()

    det = Detector(
        families=args.family,
        nthreads=4,
        quad_decimate=args.decimate,
        quad_sigma=args.blur,
        refine_edges=args.refine,
        decode_sharpening=args.sharpen,
    )

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.cam}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # 如果启用 pose，需要 K
    K = None
    if args.pose:
        if None in (args.fx, args.fy, args.cx, args.cy):
            raise SystemExit("Pose enabled but missing intrinsics. Provide --fx --fy --cx --cy (and optionally adjust --tagsize).")
        K = np.array([[args.fx, 0, args.cx],
                      [0, args.fy, args.cy],
                      [0, 0, 1]], dtype=np.float64)

    last_t = time.time()
    fps = 0.0

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = det.detect(gray)

        # overlays
        for d in detections:
            draw_tag_overlay(frame, d)
            if args.pose and K is not None:
                estimate_pose_and_draw_axes(frame, d, args.tagsize, K)

        # FPS
        now = time.time()
        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_t = now
        cv2.putText(frame, f"{args.family}  det={len(detections)}  FPS={fps:.1f}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("AprilTag realtime", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
