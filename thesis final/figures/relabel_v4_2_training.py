#!/usr/bin/env python3
"""Overlay 'v4.1' on top of the four 'v4.2' title strings in v4_2_training.png."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent
SRC = OUT_DIR / "v4_2_training.png"
DST = OUT_DIR / "v4_1_training.png"

img = Image.open(SRC).convert("RGB")
W, H = img.size
print(f"Source size: {W} x {H}")
draw = ImageDraw.Draw(img)

font_big = font_panel = None
for fp in [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]:
    try:
        font_big = ImageFont.truetype(fp, size=int(H * 0.060))
        font_panel = ImageFont.truetype(fp, size=int(H * 0.045))
        print(f"Using font: {fp}")
        break
    except Exception:
        continue
if font_big is None:
    font_big = ImageFont.load_default()
    font_panel = ImageFont.load_default()

# Suptitle band (top center)
sup_rect = (int(W * 0.18), 0, int(W * 0.82), int(H * 0.10))
draw.rectangle(sup_rect, fill="white")
sup_text = "v4.1 Hybrid Model Training (65 epochs)"
bbox = draw.textbbox((0, 0), sup_text, font=font_big)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
draw.text(((W - tw) // 2, (sup_rect[1] + sup_rect[3] - th) // 2 + 2), sup_text, fill="black", font=font_big)

# Per-panel titles (3 columns, equal width)
panel_w = W // 3
panel_titles = ["Training Loss (v4.1)", "Validation F1 (v4.1)", "Attention Gate Weights (v4.1)"]
y_top, y_bot = int(H * 0.10), int(H * 0.18)
for i, t in enumerate(panel_titles):
    x0 = i * panel_w + int(panel_w * 0.10)
    x1 = (i + 1) * panel_w - int(panel_w * 0.10)
    draw.rectangle((x0, y_top, x1, y_bot), fill="white")
    bbox = draw.textbbox((0, 0), t, font=font_panel)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx, cy = (x0 + x1) // 2, (y_top + y_bot) // 2
    draw.text((cx - tw // 2, cy - th // 2), t, fill="black", font=font_panel)

img.save(DST, "PNG")
print(f"wrote {DST}")
