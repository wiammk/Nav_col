"""Regenerate simplified manuscript pipeline and relative-closure figures."""

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
RESULTS = ROOT / "runs" / "Office_Building" / "experiments"
SEEDS = range(42, 52)
FLEETS = (1, 3, 5, 10)
METHODS = (
    ("local_independent_qlearning", "Independent local Q-learning", "#667085", "o"),
    ("centralized_qlearning", "Centralized Q-learning", "#2563EB", "s"),
    ("fedavg_qlearning", "FedAvg Q-learning", "#16A34A", "^"),
    ("fedavg_dqn", "FedAvg DQN", "#EA580C", "D"),
    ("fedavg_ppo", "FedAvg PPO", "#DC2626", "X"),
)


def read_summary(seed, robots, dynamic):
    name = "dynamic_comparison_summary.csv" if dynamic else "comparison_summary.csv"
    path = RESULTS / f"seed_{seed}" / f"robots_{robots}" / "evaluation" / name
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["method"]: row for row in csv.DictReader(stream)}


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    half = 2.262157 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean, half


FONT = r"C:\Windows\Fonts\arial.ttf"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def centered(draw, xy, text, typeface, fill):
    draw.multiline_text(xy, text, font=typeface, fill=fill, anchor="mm", align="center", spacing=8)


def arrow(draw, start, end, color="#667085", width=5):
    draw.line((start, end), fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 20
    pts = [(x2, y2),
           (x2 - size * math.cos(angle - .5), y2 - size * math.sin(angle - .5)),
           (x2 - size * math.cos(angle + .5), y2 - size * math.sin(angle + .5))]
    draw.polygon(pts, fill=color)


def draw_box(draw, rect, heading, detail, color):
    draw.rounded_rectangle(rect, radius=24, fill="white", outline=color, width=5)
    x1, y1, x2, y2 = rect
    centered(draw, ((x1 + x2) // 2, y1 + 56), heading, font(32, True), "#1F2937")
    centered(draw, ((x1 + x2) // 2, y1 + 125), detail, font(24), "#475467")


def vertical_label(img, xy, text):
    layer = Image.new("RGBA", (380, 70), (255, 255, 255, 0))
    ImageDraw.Draw(layer).text((190, 35), text, font=font(24), fill="#344054", anchor="mm")
    rotated = layer.rotate(90, expand=True)
    img.paste(rotated, (int(xy[0] - rotated.width / 2), int(xy[1] - rotated.height / 2)), rotated)


def save_image(img, basename):
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / f"{basename}.png", dpi=(300, 300), optimize=True)


def pipeline_figure():
    # A compact conceptual view: preserve the original input → graph →
    # learning → shared simulator → evaluation structure without implementation detail.
    img = Image.new("RGB", (2400, 1010), "white")
    d = ImageDraw.Draw(img)
    centered(d, (1200, 82), "From building model to multi-robot evaluation", font(44, True), "#111827")

    # Input and graph construction form the top row.
    draw_box(d, (150, 205, 850, 390), "IFC building model", "rooms · doors · stairs", "#475467")
    draw_box(d, (1550, 205, 2250, 390), "Navigation graph", "nodes = spaces · edges = routes", "#2563EB")
    arrow(d, (880, 298), (1515, 298), "#667085", 5)

    # The graph supplies the tabular baseline and frozen-embedding neural agents.
    draw_box(d, (230, 535, 930, 720), "Tabular Q-learning", "node-index states", "#16A34A")
    draw_box(d, (1470, 535, 2170, 720), "DQN / PPO", "frozen GCN node embeddings", "#7C3AED")
    d.line((1900, 390, 1900, 465, 1780, 465), fill="#667085", width=5)
    arrow(d, (1780, 465), (1780, 515), "#667085", 5)
    d.line((1900, 465, 580, 465), fill="#667085", width=5)
    arrow(d, (580, 465), (580, 515), "#667085", 5)

    # Both policy families are evaluated in the same simulator and protocol.
    draw_box(d, (760, 770, 1510, 970), "Shared multi-robot simulator", "common scenarios and constraints", "#EA580C")
    arrow(d, (580, 720), (900, 755), "#667085", 5)
    arrow(d, (1820, 720), (1370, 755), "#667085", 5)
    draw_box(d, (1600, 770, 2280, 970), "Evaluation", "success · SPL · closures", "#0891B2")
    arrow(d, (1510, 870), (1575, 870), "#667085", 5)
    save_image(img, "framework")


def closure_figure():
    summaries = {
        (seed, k, dynamic): read_summary(seed, k, dynamic)
        for seed in SEEDS for k in FLEETS for dynamic in (False, True)
    }
    img = Image.new("RGB", (3000, 1250), "white")
    d = ImageDraw.Draw(img)
    centered(d, (1500, 75), "Effect of a controlled dynamic edge closure", font(44, True), "#111827")
    colors = {m[0]: m[2] for m in METHODS}
    markers = {m[0]: m[3] for m in METHODS}
    # Draw side-by-side panels with matched plotting areas.
    panel_specs = [(120, 190, 1380, 1070), (1590, 190, 2860, 1070)]
    for panel_index, (x1, y1, x2, y2) in enumerate(panel_specs):
        d.line((x1, y2, x2, y2), fill="#344054", width=3)
        d.line((x1, y1, x1, y2), fill="#344054", width=3)
        tick_values = range(0, 101, 20) if panel_index == 0 else range(0, 26, 5)
        tick_max = 100 if panel_index == 0 else 25
        for t in tick_values:
            yy = y2 - (y2 - y1) * t / tick_max
            d.line((x1, yy, x2, yy), fill="#EAECF0", width=2)
            d.text((x1 - 25, yy), str(t), font=font(23), fill="#344054", anchor="rm")
    centered(d, (750, 150), "FedAvg Q-learning", font(32, True), "#1F2937")
    centered(d, (2225, 150), "Relative success decrease", font(32, True), "#1F2937")
    vertical_label(img, (37, 630), "Success rate (%)")
    vertical_label(img, (1500, 630), "Relative decrease (%)")
    d = ImageDraw.Draw(img)
    for ix, k in enumerate(FLEETS):
        xx0, _, xx1, yy1 = panel_specs[0]
        x = xx0 + 100 + ix * (xx1 - xx0 - 200) / (len(FLEETS) - 1)
        d.text((x, yy1 + 40), str(k), font=font(25), fill="#344054", anchor="mm")
        d.text((x, yy1 + 75), "robots" if ix == 1 else "", font=font(18), fill="#344054", anchor="mm")
        xx0, _, xx1, yy1 = panel_specs[1]
        x = xx0 + 100 + ix * (xx1 - xx0 - 200) / (len(FLEETS) - 1)
        d.text((x, yy1 + 40), str(k), font=font(25), fill="#344054", anchor="mm")

    # FedAvg static and dynamic success curves with seed-level 95% intervals.
    left = panel_specs[0]
    series = []
    for dynamic, label, marker, dash in ((False, "Static", "o", False), (True, "Dynamic edge closure", "s", True)):
        points = []
        for i, k in enumerate(FLEETS):
            vals = [100 * float(summaries[(seed, k, dynamic)]["fedavg_qlearning"]["success_mean"]) for seed in SEEDS]
            mean, ci = mean_ci(vals)
            px = left[0] + 100 + i * (left[2] - left[0] - 200) / 3
            py = left[3] - mean / 100 * (left[3] - left[1])
            points.append((px, py, ci))
        series.append((points, label, "#16A34A", marker, dash))
    for points, label, color, marker, dash in series:
        coords = [(p[0], p[1]) for p in points]
        for a, b in zip(coords, coords[1:]):
            if dash:
                # dashed line in short segments
                for q in np.linspace(0, 1, 12)[:-1:2]:
                    d.line((a[0] + (b[0]-a[0])*q, a[1] + (b[1]-a[1])*q,
                            a[0] + (b[0]-a[0])*(q+.08), a[1] + (b[1]-a[1])*(q+.08)), fill=color, width=5)
            else:
                d.line((a, b), fill=color, width=5)
        for px, py, ci in points:
            err = ci / 100 * (left[3] - left[1])
            d.line((px, py-err, px, py+err), fill=color, width=3)
            d.ellipse((px-11, py-11, px+11, py+11), fill=color) if marker == "o" else d.rectangle((px-10, py-10, px+10, py+10), fill=color)
    d.line((990, 220, 1050, 220), fill="#16A34A", width=5)
    d.ellipse((1010, 210, 1030, 230), fill="#16A34A")
    d.text((1060, 220), "Static", font=font(22), fill="#344054", anchor="lm")
    d.line((990, 265, 1050, 265), fill="#16A34A", width=5)
    d.rectangle((1010, 255, 1030, 275), fill="#16A34A")
    d.text((1060, 265), "Dynamic", font=font(22), fill="#344054", anchor="lm")

    # Paired, per-seed relative decrease: 100*(static-dynamic)/static.
    right = panel_specs[1]
    y_max = 25.0
    for method, label, color, marker in METHODS:
        points = []
        for i, k in enumerate(FLEETS):
            vals = []
            for seed in SEEDS:
                st = float(summaries[(seed, k, False)][method]["success_mean"])
                dy = float(summaries[(seed, k, True)][method]["success_mean"])
                vals.append(100 * (st - dy) / st if st else 0.0)
            mean, ci = mean_ci(vals)
            px = right[0] + 100 + i * (right[2] - right[0] - 200) / 3
            py = right[3] - mean / y_max * (right[3] - right[1])
            points.append((px, py, ci, mean))
        coords = [(p[0], p[1]) for p in points]
        d.line(coords, fill=color, width=4)
        for px, py, ci, mean in points:
            err = ci / y_max * (right[3] - right[1])
            d.line((px, py-err, px, py+err), fill=color, width=3)
            d.line((px-8, py-err, px+8, py-err), fill=color, width=2)
            d.line((px-8, py+err, px+8, py+err), fill=color, width=2)
            if marker in ("o", "D", "X"):
                d.ellipse((px-10, py-10, px+10, py+10), fill=color)
            else:
                d.rectangle((px-9, py-9, px+9, py+9), fill=color)
    # Right-panel tick labels and compact legend.
    for tick in range(0, 26, 5):
        yy = right[3] - tick / y_max * (right[3] - right[1])
        d.text((right[0]-25, yy), str(tick), font=font(22), fill="#344054", anchor="rm")
    legend_y = 210
    for idx, (_, label, color, marker) in enumerate(METHODS):
        yy = legend_y + idx * 39
        d.line((2360, yy, 2410, yy), fill=color, width=4)
        d.text((2425, yy), label, font=font(18), fill="#344054", anchor="lm")
    save_image(img, "figure_3_static_vs_dynamic")


if __name__ == "__main__":
    pipeline_figure()
    closure_figure()
    print(f"Wrote manuscript figures to {OUT}")
