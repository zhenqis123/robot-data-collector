#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def draw_axes(ax, T, axis_len=0.05, lw=2.0, alpha=1.0):
    origin = T[:3, 3]
    R = T[:3, :3]
    colors = ["r", "g", "b"]  # x,y,z

    for i in range(3):
        end = origin + R[:, i] * axis_len
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=colors[i],
            linewidth=lw,
            alpha=alpha,
        )


def set_equal_aspect_3d(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins).max()
    span = max(span, 1e-6)

    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)


def main():
    ap = argparse.ArgumentParser(description="Visualize ArUco map (marker0 as world)")
    ap.add_argument("--map", required=True, type=Path, help="aruco_map_0to4.json")
    ap.add_argument("--axis-len", type=float, default=0.05, help="axis length (m)")
    ap.add_argument("--save", type=Path, default=None, help="optional output PNG path")
    args = ap.parse_args()

    data = json.loads(args.map.read_text())
    markers = data["markers"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    points = []

    for mid_str, info in markers.items():
        mid = int(mid_str)
        T = np.array(info["T_W_M"], dtype=np.float64)
        p = T[:3, 3]
        points.append(p)

        if mid == 0:
            # WORLD / marker0
            draw_axes(ax, T, axis_len=args.axis_len * 1.3, lw=4.0)
            ax.scatter(p[0], p[1], p[2], color="k", s=80)
            ax.text(p[0], p[1], p[2], "WORLD (ID 0)", fontsize=11, fontweight="bold")
        else:
            draw_axes(ax, T, axis_len=args.axis_len, lw=2.0, alpha=0.9)
            ax.scatter(p[0], p[1], p[2], color="k", s=30)
            ax.text(p[0], p[1], p[2], f"ID {mid}", fontsize=10)

    points = np.stack(points, axis=0)

    set_equal_aspect_3d(ax, points)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("ArUco Map (World = Marker 0)")

    plt.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=200)
        print(f"[OK] saved figure to {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
