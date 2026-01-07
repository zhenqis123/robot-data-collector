import os
import numpy as np
import cv2

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# -----------------------------
# 配置区：按你的需求改这里
# -----------------------------
OUTPUT_DIR = "aruco_print_out"
PDF_PATH = os.path.join(OUTPUT_DIR, "aruco_table_markers_A4.pdf")

# 建议：DICT_4X4_50 足够常用；如果你担心误检，可用 5X5
ARUCO_DICT_NAME = "DICT_4X4_50"

# 你桌面要贴的 marker IDs（四角 + 中心）
MARKER_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 物理尺寸（毫米）
MARKER_SIZE_MM = 50          # 黑色编码区的边长（不含白边）
QUIET_ZONE_MM = 12           # 白边宽度（建议 >= marker_size 的 1/4）
LABEL_FONT_SIZE = 10

# 生成 PNG 的像素分辨率（越大越清晰；用于打印建议 >= 800）
MARKER_PIXELS = 1000         # 黑色编码区像素大小
QUIET_ZONE_PIXELS = int(MARKER_PIXELS * (QUIET_ZONE_MM / MARKER_SIZE_MM))

# A4 排版：每行/列放几个
GRID_COLS = 2
GRID_ROWS = 3

# 页面边距（毫米）
MARGIN_MM = 12

# -----------------------------
# 工具函数
# -----------------------------
def get_aruco_dictionary(name: str):
    name = name.upper()
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"cv2.aruco 没有这个字典：{name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))

def make_marker_image(dictionary, marker_id: int, marker_pixels: int):
    """生成单个 ArUco marker（黑白，uint8）"""
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_pixels)
    # OpenCV 有些版本返回 1 通道，有些返回 3 通道；统一成 1 通道
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def add_quiet_zone(marker_img: np.ndarray, quiet_px: int):
    """给 marker 四周加白边（quiet zone）"""
    h, w = marker_img.shape[:2]
    out = np.full((h + 2 * quiet_px, w + 2 * quiet_px), 255, dtype=np.uint8)
    out[quiet_px:quiet_px + h, quiet_px:quiet_px + w] = marker_img
    return out

def save_png(path: str, img_gray: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_gray)

# -----------------------------
# 主流程
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dictionary = get_aruco_dictionary(ARUCO_DICT_NAME)

    # 先生成并保存单个 PNG
    png_paths = []
    for mid in MARKER_IDS:
        marker = make_marker_image(dictionary, mid, MARKER_PIXELS)
        marker_q = add_quiet_zone(marker, QUIET_ZONE_PIXELS)
        png_path = os.path.join(OUTPUT_DIR, f"aruco_id_{mid}.png")
        save_png(png_path, marker_q)
        png_paths.append((mid, png_path))

    # 再生成 A4 PDF（按 mm 精确放置）
    c = canvas.Canvas(PDF_PATH, pagesize=A4)
    page_w, page_h = A4

    # 单个 marker 在 PDF 上的物理尺寸（含白边）
    total_size_mm = MARKER_SIZE_MM + 2 * QUIET_ZONE_MM
    total_size_pt = total_size_mm * mm

    usable_w = page_w - 2 * (MARGIN_MM * mm)
    usable_h = page_h - 2 * (MARGIN_MM * mm)

    cell_w = usable_w / GRID_COLS
    cell_h = usable_h / GRID_ROWS

    # 放置 marker（从上到下、从左到右）
    for idx, (mid, png_path) in enumerate(png_paths):
        r = idx // GRID_COLS
        col = idx % GRID_COLS

        if r >= GRID_ROWS:
            # 超出一页就另起一页
            c.showPage()
            r = 0

        # 计算每个 cell 的左下角
        x0 = (MARGIN_MM * mm) + col * cell_w
        y0 = page_h - (MARGIN_MM * mm) - (r + 1) * cell_h

        # 居中放置图片
        x_img = x0 + (cell_w - total_size_pt) / 2
        y_img = y0 + (cell_h - total_size_pt) / 2 + 8 * mm  # 给标签留点空间

        img_reader = ImageReader(png_path)
        c.drawImage(img_reader, x_img, y_img, width=total_size_pt, height=total_size_pt, mask='auto')

        # 标签
        c.setFont("Helvetica", LABEL_FONT_SIZE)
        c.drawCentredString(x0 + cell_w / 2, y_img - 6 * mm, f"ArUco ID = {mid}")

    # 打印注意事项
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN_MM * mm, MARGIN_MM * mm,
                 "Print settings: 100% / Actual size. DO NOT 'Fit to page'. Measure marker size with a ruler.")

    c.save()

    print("Done.")
    print(f"PDF: {PDF_PATH}")
    print(f"PNGs are in: {OUTPUT_DIR}")
    print("\n打印时务必选择：100% / Actual size（不要适应页面）。")

if __name__ == "__main__":
    main()
