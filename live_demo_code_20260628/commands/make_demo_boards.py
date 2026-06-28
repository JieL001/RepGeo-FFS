import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "live_demo_code_20260628" / "outputs"
BOARDS = OUTPUTS / "boards"


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_H2 = font(23, True)
F_BODY = font(19)
F_SMALL = font(15)
F_MONO = font(16)


def open_rgb(path):
    img = Image.open(path).convert("RGB")
    return img


def fit_image(img, box_w, box_h, bg=(250, 252, 255)):
    canvas = Image.new("RGB", (box_w, box_h), bg)
    scale = min(box_w / img.width, box_h / img.height)
    nw = max(1, int(img.width * scale))
    nh = max(1, int(img.height * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((box_w - nw) // 2, (box_h - nh) // 2))
    return canvas


def rounded(draw, xy, radius, fill, outline=None, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_label(draw, xy, text, fill=(32, 42, 55), fnt=F_BODY):
    draw.text(xy, text, fill=fill, font=fnt)


def safe_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_flexible(path):
    path = Path(path)
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "utf-16-le", "gbk"):
        try:
            return raw.decode(enc).replace("\x00", "")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore").replace("\x00", "")


def fmt_metric(value, percent=False):
    if value is None or not math.isfinite(float(value)):
        return "-"
    if percent:
        return f"{float(value) * 100:.3f}%"
    return f"{float(value):.4f}"


def build_inference_board():
    BOARDS.mkdir(parents=True, exist_ok=True)
    models = [
        ("FFS base", OUTPUTS / "infer_base"),
        ("RepGeo target gamma=1", OUTPUTS / "infer_repgeo_target_gamma1"),
        ("RepGeo calibrated gamma=0.01", OUTPUTS / "infer_repgeo_calibrated_gamma0p01"),
    ]
    available = [(name, d) for name, d in models if (d / "disp_vis.png").exists()]
    if not available:
        print("[skip] no inference outputs found")
        return None

    first_dir = available[0][1]
    left_path = first_dir / "left.png"
    right_path = first_dir / "right.png"
    if not left_path.exists():
        left_path = ROOT / "demo_data" / "left.png"
    if not right_path.exists():
        right_path = ROOT / "demo_data" / "right.png"

    W = 2400
    margin = 70
    gap = 28
    title_h = 110
    top_h = 430
    disp_h = 520
    caption_h = 105
    H = title_h + top_h + disp_h + caption_h + margin
    img = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(img)

    draw_label(draw, (margin, 38), "Live inference visualization: stereo input -> disparity", (14, 35, 61), F_TITLE)
    draw_label(draw, (margin, 82), "Same stereo pair, multiple checkpoints. No browser frame, no terminal text.", (86, 100, 118), F_BODY)

    top_y = title_h
    cell_w = (W - 2 * margin - gap) // 2
    for idx, (label, path) in enumerate([("Left image", left_path), ("Right image", right_path)]):
        x = margin + idx * (cell_w + gap)
        rounded(draw, (x, top_y, x + cell_w, top_y + top_h), 18, (255, 255, 255), (205, 214, 226), 2)
        draw_label(draw, (x + 20, top_y + 16), label, (28, 42, 58), F_H2)
        panel = fit_image(open_rgb(path), cell_w - 40, top_h - 70, (255, 255, 255))
        img.paste(panel, (x + 20, top_y + 56))

    disp_y = title_h + top_h + 35
    n = len(available)
    cell_w = (W - 2 * margin - gap * (n - 1)) // n
    for idx, (name, directory) in enumerate(available):
        x = margin + idx * (cell_w + gap)
        rounded(draw, (x, disp_y, x + cell_w, disp_y + disp_h), 18, (255, 255, 255), (205, 214, 226), 2)
        draw_label(draw, (x + 20, disp_y + 16), name, (28, 42, 58), F_H2)
        panel = fit_image(open_rgb(directory / "disp_vis.png"), cell_w - 40, disp_h - 100, (255, 255, 255))
        img.paste(panel, (x + 20, disp_y + 58))
        npy = directory / "disp.npy"
        if npy.exists():
            arr = np.load(npy)
            stats = f"disp min/mean/max: {arr.min():.2f} / {arr.mean():.2f} / {arr.max():.2f}"
            draw_label(draw, (x + 20, disp_y + disp_h - 34), stats, (83, 96, 112), F_SMALL)

    cap_y = disp_y + disp_h + 24
    rounded(draw, (margin, cap_y, W - margin, cap_y + 65), 16, (235, 247, 255), (92, 158, 219), 2)
    draw_label(
        draw,
        (margin + 24, cap_y + 18),
        "Use in live demo: open this PNG after running inference scripts; it shows inputs and all available model outputs together.",
        (25, 75, 116),
        F_BODY,
    )
    out = BOARDS / "live_inference_comparison.png"
    img.save(out)
    print(f"[ok] {out}")
    return out


def build_eval_board():
    BOARDS.mkdir(parents=True, exist_ok=True)
    rows = [
        ("FFS base", safe_json(OUTPUTS / "eval_smoke_base.json")),
        ("RepGeo target", safe_json(OUTPUTS / "eval_smoke_repgeo_target.json")),
    ]
    rows = [(name, data) for name, data in rows if data and data.get("metrics")]
    if not rows:
        data = safe_json(OUTPUTS / "eval_smoke.json")
        if data:
            rows = [("FFS eval smoke", data)]
    if not rows:
        print("[skip] no eval json found")
        return None

    metrics = [
        ("EPE", "epe", False),
        ("D1", "d1", True),
        ("bad3", "bad3", True),
        ("EPE visible", "epe_vis", False),
        ("EPE occluded", "epe_occ", False),
    ]
    W = 2000
    H = 930
    margin = 70
    img = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw_label(draw, (margin, 40), "Evaluation smoke results", (14, 35, 61), F_TITLE)
    draw_label(draw, (margin, 84), "Lower is better. Same evaluation path, same KITTI smoke split.", (86, 100, 118), F_BODY)

    table_x = margin
    table_y = 145
    table_w = W - 2 * margin
    row_h = 84
    col_w = [320, 260, 260, 260, 300, table_w - (320 + 260 + 260 + 260 + 300)]
    headers = ["Method"] + [m[0] for m in metrics]
    rounded(draw, (table_x, table_y, table_x + table_w, table_y + row_h * (len(rows) + 1)), 18, (255, 255, 255), (202, 212, 225), 2)
    y = table_y
    x = table_x
    for i, header in enumerate(headers):
        draw.rectangle((x, y, x + col_w[i], y + row_h), fill=(18, 38, 63))
        draw_label(draw, (x + 18, y + 26), header, (255, 255, 255), F_BODY)
        x += col_w[i]
    for r, (name, data) in enumerate(rows):
        y = table_y + row_h * (r + 1)
        fill = (255, 255, 255) if r % 2 == 0 else (242, 247, 252)
        x = table_x
        draw.rectangle((x, y, table_x + table_w, y + row_h), fill=fill)
        draw_label(draw, (x + 18, y + 27), name, (27, 42, 59), F_BODY)
        x += col_w[0]
        for i, (_, key, pct) in enumerate(metrics):
            val = data["metrics"].get(key)
            draw_label(draw, (x + 18, y + 27), fmt_metric(val, pct), (27, 42, 59), F_BODY)
            x += col_w[i + 1]

    if len(rows) >= 2:
        base = rows[0][1]["metrics"]
        ours = rows[1][1]["metrics"]
        chart_y = table_y + row_h * (len(rows) + 1) + 70
        draw_label(draw, (margin, chart_y - 36), "Relative reduction from FFS base", (14, 35, 61), F_H2)
        bar_x = margin
        max_w = W - 2 * margin - 360
        for idx, (label, key, pct) in enumerate(metrics[:3]):
            y = chart_y + idx * 78
            b = float(base.get(key, 0))
            o = float(ours.get(key, 0))
            reduction = 0.0 if b <= 0 else max(0.0, (b - o) / b)
            draw_label(draw, (bar_x, y + 8), label, (38, 52, 69), F_BODY)
            bx = bar_x + 210
            rounded(draw, (bx, y, bx + max_w, y + 38), 10, (230, 236, 244), None)
            rounded(draw, (bx, y, bx + int(max_w * reduction), y + 38), 10, (31, 151, 121), None)
            draw_label(draw, (bx + max_w + 24, y + 7), f"{reduction * 100:.1f}% lower", (31, 111, 91), F_BODY)

    out = BOARDS / "eval_metrics_board.png"
    img.save(out)
    print(f"[ok] {out}")
    return out


def parse_train_log(path):
    path = Path(path)
    if not path.exists():
        return []
    pat = re.compile(
        r"epoch\s+(?P<epoch>\d+)\s+done:.*?train_loss=(?P<train>[0-9.]+).*?val_EPE=(?P<epe>[0-9.]+).*?val_D1=(?P<d1>[0-9.]+).*?val_bad3=(?P<bad3>[0-9.]+)",
        re.IGNORECASE,
    )
    text = read_text_flexible(path)
    rows = []
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            rows.append(
                {
                    "epoch": int(m.group("epoch")),
                    "train_loss": float(m.group("train")),
                    "val_epe": float(m.group("epe")),
                    "val_d1": float(m.group("d1")),
                    "val_bad3": float(m.group("bad3")),
                }
            )
    return rows


def build_train_board():
    BOARDS.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUTS / "train_smoke_repgeo" / "train_smoke_repgeo.log"
    rows = parse_train_log(log_path)
    W = 1900
    H = 900
    margin = 70
    img = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw_label(draw, (margin, 40), "Training smoke summary", (14, 35, 61), F_TITLE)
    draw_label(draw, (margin, 84), "Purpose: prove the training path, loss computation, backward pass, and checkpoint writing are executable.", (86, 100, 118), F_BODY)

    cards = [
        ("Training samples", "4"),
        ("Validation samples", "2"),
        ("Epochs", "1"),
        ("Trainable path", "RepGeo / LoRA residual"),
        ("Loss terms", "supervised + prior + sparse + delta"),
    ]
    card_y = 145
    card_w = (W - 2 * margin - 4 * 20) // 5
    for i, (k, v) in enumerate(cards):
        x = margin + i * (card_w + 20)
        rounded(draw, (x, card_y, x + card_w, card_y + 145), 16, (255, 255, 255), (202, 212, 225), 2)
        draw_label(draw, (x + 18, card_y + 22), k, (80, 94, 112), F_SMALL)
        draw_label(draw, (x + 18, card_y + 72), v, (20, 46, 78), F_H2 if len(v) < 12 else F_BODY)

    table_y = 345
    table_w = W - 2 * margin
    rounded(draw, (margin, table_y, margin + table_w, table_y + 265), 18, (255, 255, 255), (202, 212, 225), 2)
    headers = ["Epoch", "train loss", "val EPE", "val D1", "val bad3"]
    col_w = [180, 360, 320, 320, table_w - (180 + 360 + 320 + 320)]
    x = margin
    for i, h in enumerate(headers):
        draw.rectangle((x, table_y, x + col_w[i], table_y + 68), fill=(18, 38, 63))
        draw_label(draw, (x + 18, table_y + 21), h, (255, 255, 255), F_BODY)
        x += col_w[i]
    if rows:
        for r, row in enumerate(rows[:2]):
            y = table_y + 68 * (r + 1)
            x = margin
            vals = [
                str(row["epoch"]),
                f"{row['train_loss']:.4f}",
                f"{row['val_epe']:.4f}",
                f"{row['val_d1'] * 100:.3f}%",
                f"{row['val_bad3'] * 100:.3f}%",
            ]
            for i, val in enumerate(vals):
                draw.rectangle((x, y, x + col_w[i], y + 68), fill=(255, 255, 255) if r % 2 == 0 else (242, 247, 252))
                draw_label(draw, (x + 18, y + 21), val, (28, 42, 58), F_BODY)
                x += col_w[i]
    else:
        draw_label(draw, (margin + 22, table_y + 96), f"No parsed epoch summary yet. Run 03_train_smoke_repgeo.ps1 first.\nExpected log: {log_path}", (160, 82, 45), F_BODY)

    log_y = 655
    rounded(draw, (margin, log_y, W - margin, log_y + 155), 16, (240, 248, 255), (92, 158, 219), 2)
    lines = []
    if log_path.exists():
        text = read_text_flexible(log_path).splitlines()
        lines = [ln for ln in text if "device:" in ln or "LoRA:" in ln or "trainable params" in ln or "epoch" in ln][-5:]
    if not lines:
        lines = [
            "Run the smoke script to generate this log.",
            "The board will summarize the parsed epoch metrics automatically.",
        ]
    draw_label(draw, (margin + 22, log_y + 18), "Last training log lines", (20, 72, 116), F_H2)
    for i, line in enumerate(lines[:5]):
        draw_label(draw, (margin + 22, log_y + 56 + i * 19), line[:165], (50, 68, 88), F_SMALL)

    out = BOARDS / "train_smoke_board.png"
    img.save(out)
    print(f"[ok] {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "infer", "eval", "train"], default="all")
    args = parser.parse_args()
    if args.mode in ("all", "infer"):
        build_inference_board()
    if args.mode in ("all", "eval"):
        build_eval_board()
    if args.mode in ("all", "train"):
        build_train_board()


if __name__ == "__main__":
    main()
