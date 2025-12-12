import os
import re
import glob
from math import ceil

import cairosvg
from PIL import Image

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


# -----------------------------
# 配置
# -----------------------------
INPUT_GLOB = "third_party/apriltag-imgs/tag41_12_*.svg"
OUTPUT_DIR = "print_out"
PDF_PATH = os.path.join(OUTPUT_DIR, "apriltag_A4_layout.pdf")

# 物理尺寸（mm）
TAG_SIZE_MM = 50            # Tag 本体（黑框到黑框）边长
QUIET_ZONE_MM = 15          # 额外白边宽度
LABEL_FONT_SIZE = 9

# 网格布局（含白边后尺寸更大，建议 2x3；想更密可减小 QUIET_ZONE_MM）
GRID_COLS = 2
GRID_ROWS = 3

# 页边距（mm）
MARGIN_MM = 12

# 栅格化分辨率：对应“Tag 本体 50mm”的像素大小
TAG_PX = 1400

# 标签下方留空（mm）
LABEL_GAP_MM = 6

# 裁剪虚线样式
CUTLINE_DASH_MM = (2.0, 2.0)  # 虚线：画2mm，空2mm
CUTLINE_WIDTH_PT = 0.6        # 线宽（pt）
CUTLINE_INSET_MM = 0.0        # 裁剪线相对总尺寸的内缩/外扩（+外扩，-内缩）。默认0=正好在边界


# -----------------------------
# 工具
# -----------------------------
def natural_key(path: str):
    base = os.path.basename(path)
    m = re.search(r"(\d+)\.svg$", base)
    return int(m.group(1)) if m else base

def svg_tag_to_png_with_quiet_zone(svg_path: str, out_png_path: str):
    """
    把“无外部白边”的 Tag SVG 转成带 quiet zone 的 PNG。
    - Tag 本体：TAG_PX x TAG_PX
    - quiet zone：按 QUIET_ZONE_MM / TAG_SIZE_MM 等比例换算成像素
    """
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)

    tag_png_tmp = out_png_path.replace(".png", f"__tag_{TAG_PX}px.tmp.png")
    cairosvg.svg2png(
        url=svg_path,
        write_to=tag_png_tmp,
        output_width=TAG_PX,
        output_height=TAG_PX
    )

    tag_im = Image.open(tag_png_tmp).convert("RGB")

    quiet_px = round(TAG_PX * (QUIET_ZONE_MM / TAG_SIZE_MM))
    total_px = TAG_PX + 2 * quiet_px

    out = Image.new("RGB", (total_px, total_px), (255, 255, 255))
    out.paste(tag_im, (quiet_px, quiet_px))
    out.save(out_png_path)

    try:
        os.remove(tag_png_tmp)
    except OSError:
        pass

def draw_cutline(c: canvas.Canvas, x: float, y: float, w: float, h: float):
    """
    画裁剪虚线框（不影响尺寸，只是提示线）
    x,y 是左下角；w,h 是框的宽高（pt）
    """
    c.saveState()
    c.setLineWidth(CUTLINE_WIDTH_PT)
    c.setDash(CUTLINE_DASH_MM[0] * mm, CUTLINE_DASH_MM[1] * mm)
    # 不设颜色也行；默认黑色，打印机一般会打印出来
    c.rect(x, y, w, h, stroke=1, fill=0)
    c.restoreState()

def main():
    svgs = sorted(glob.glob(INPUT_GLOB), key=natural_key)
    if not svgs:
        raise SystemExit(f"没找到文件：{INPUT_GLOB}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    png_items = []
    for svg in svgs:
        stem = os.path.splitext(os.path.basename(svg))[0]
        png = os.path.join(OUTPUT_DIR, f"{stem}_tag{TAG_SIZE_MM}mm_q{QUIET_ZONE_MM}mm.png")
        svg_tag_to_png_with_quiet_zone(svg, png)
        png_items.append((svg, png))

    TOTAL_SIZE_MM = TAG_SIZE_MM + 2 * QUIET_ZONE_MM
    total_pt = TOTAL_SIZE_MM * mm
    inset_pt = CUTLINE_INSET_MM * mm

    c = canvas.Canvas(PDF_PATH, pagesize=A4)
    page_w, page_h = A4

    usable_w = page_w - 2 * (MARGIN_MM * mm)
    usable_h = page_h - 2 * (MARGIN_MM * mm)

    cell_w = usable_w / GRID_COLS
    cell_h = usable_h / GRID_ROWS

    # 放不下就直接报错，避免任何缩放“偷偷发生”
    if total_pt > cell_w or total_pt + (LABEL_GAP_MM * mm) > cell_h:
        raise SystemExit(
            f"当前 GRID_COLS/GRID_ROWS 放不下单个标签。\n"
            f"TOTAL_SIZE_MM={TOTAL_SIZE_MM}mm（含白边），"
            f"cell_w={cell_w/mm:.1f}mm, cell_h={cell_h/mm:.1f}mm。\n"
            f"请降低 GRID_COLS/GRID_ROWS 或减小 QUIET_ZONE_MM。"
        )

    per_page = GRID_COLS * GRID_ROWS
    pages = ceil(len(png_items) / per_page)

    for p in range(pages):
        start = p * per_page
        end = min((p + 1) * per_page, len(png_items))
        chunk = png_items[start:end]

        for idx, (svg_path, png_path) in enumerate(chunk):
            r = idx // GRID_COLS
            col = idx % GRID_COLS

            x0 = (MARGIN_MM * mm) + col * cell_w
            y0 = page_h - (MARGIN_MM * mm) - (r + 1) * cell_h

            x_img = x0 + (cell_w - total_pt) / 2
            y_img = y0 + (cell_h - total_pt) / 2 + (LABEL_GAP_MM * mm)

            # 1) 画 tag（含白边的总尺寸）
            c.drawImage(ImageReader(png_path), x_img, y_img,
                        width=total_pt, height=total_pt, mask='auto')

            # 2) 画裁剪虚线框（默认正好围住总尺寸）
            draw_cutline(c,
                         x_img - inset_pt,
                         y_img - inset_pt,
                         total_pt + 2 * inset_pt,
                         total_pt + 2 * inset_pt)

            # 3) 标签
            base = os.path.basename(svg_path)
            m = re.search(r"(\d+)\.svg$", base)
            tag_id = m.group(1) if m else base

            c.setFont("Helvetica", LABEL_FONT_SIZE)
            c.drawCentredString(
                x0 + cell_w / 2,
                y_img - (LABEL_GAP_MM * mm),
                f"tagStandard41h12  ID={tag_id}  tag={TAG_SIZE_MM}mm  quiet={QUIET_ZONE_MM}mm"
            )

        c.setFont("Helvetica", 8)
        c.drawString(
            MARGIN_MM * mm, MARGIN_MM * mm,
            "Print: Actual size / 100%. Do NOT Fit/Shrink. Measure after printing: "
            f"tag={TAG_SIZE_MM}mm, total={TOTAL_SIZE_MM}mm."
        )

        if p != pages - 1:
            c.showPage()

    c.save()
    print(f"Done. PDF => {PDF_PATH}")
    print(f"Each tag: body {TAG_SIZE_MM}mm + quiet {QUIET_ZONE_MM}mm each side => total {TOTAL_SIZE_MM}mm.")
    print("Acrobat 打印选 Actual Size / 100%，不要 Fit。打印后用尺量确认。")

if __name__ == "__main__":
    main()
