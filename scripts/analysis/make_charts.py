"""Generate the README's eval chart as hand-rolled SVG. No plotting dependency.

Every number is READ FROM THE COMMITTED REPORTS, never typed in:

  results/eval500-20260808/*.json   teacher / SFT / base
  results/eval500-20260810/*.json   the GRPO checkpoint (paired re-run)

so the chart cannot drift from the reports it illustrates. Re-run after any new eval:

    uv run python scripts/analysis/make_charts.py

WHY HAND-ROLLED SVG: one static bar chart does not justify matplotlib in the
dependency tree. It also lets the chart carry an explicit light background, so it
stays readable under both GitHub themes -- a transparent chart with dark axis text
disappears in dark mode.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
IMG = ROOT / "docs" / "img"

BG = "#ffffff"          # explicit, so the chart survives GitHub's dark theme
FRAME = "#d8dee4"
FG = "#1f2328"
MUTED = "#656d76"
GRID = "#eaeef2"

C_BASE = "#8c959f"      # untrained Gemma-4-E4B
C_TEACH = "#5a54c9"     # Qwen3-235B teacher
C_SFT = "#1a7f5a"       # the shipped model
C_GRPO = "#bf6516"      # the RL attempt

FONT = ("-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,"
        "'Liberation Sans',sans-serif")

METRICS = [
    ("schema-valid", "schema_valid_rate"),
    ("found accuracy", "found_accuracy"),
    ("item F1", "f1_mean"),
    ("precision", "precision_mean"),
    ("recall", "recall_mean"),
]
# The teacher IS the reference the students are scored against, so it has no
# precision/recall/F1 of its own -- pairing it against itself would print 1.000.
SELF_REPORT_ONLY = {"schema_valid_rate", "found_accuracy"}
# ...and its "found" number is a DIFFERENT quantity from the students': the rate at
# which it returned found=true, not the accuracy of that flag against a reference.
# Same axis, so it is drawn, but marked -- silently mixing the two would mislead.
TEACHER_KEYS = {"found_accuracy": "found_rate"}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, fill=FG, anchor="start", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')


def rect(x, y, w, h, fill, rx=0, stroke=None, dash=None):
    st = f' stroke="{stroke}"' if stroke else ""
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" '
            f'height="{max(h, 0):.1f}" fill="{fill}" rx="{rx}"{st}{da}/>')


def line(x1, y1, x2, y2, stroke=GRID, width=1):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}"/>')


def load(run_set, name):
    return json.loads((RESULTS / run_set / f"{name}.json").read_text(encoding="utf-8"))


def chart_eval_models():
    models = [
        ("Gemma-4-E4B base", load("eval500-20260808", "gemma-base"), C_BASE),
        ("Qwen3-235B teacher", load("eval500-20260808", "teacher-qwen3-235b"), C_TEACH),
        ("Gemma + SFT", load("eval500-20260808", "gemma-sft"), C_SFT),
        ("Gemma + SFT + GRPO", load("eval500-20260810", "gemma-grpo"), C_GRPO),
    ]
    W, H = 880, 462
    L, R, T, B = 56, 22, 78, 112
    pw, ph = W - L - R, H - T - B

    out = [text(20, 30, "Every model on the same 500 held-out restaurants", size=15,
                weight="600"),
           text(20, 50, "Eval plan: split=eval, seed 42, 40% dietary-conditioned. "
                        "Students scored against the teacher's traces.",
                size=11.5, fill=MUTED)]

    for i in range(6):
        v = i / 5
        y = T + ph - v * ph
        out.append(line(L, y, L + pw, y, GRID))
        out.append(text(L - 9, y + 4, f"{v:.1f}", size=10.5, fill=MUTED, anchor="end"))

    gw = pw / len(METRICS)
    bw = gw * 0.155
    for gi, (label, key) in enumerate(METRICS):
        gx = L + gi * gw
        if gi:
            out.append(line(gx, T - 6, gx, T + ph, "#f0f3f6"))
        for bi, (_, rep, colour) in enumerate(models):
            x = gx + gw / 2 + (bi - 2) * bw * 1.12 + bw * 0.06
            agg = rep["aggregate"]["all"]
            is_teacher = colour == C_TEACH
            if is_teacher and key not in SELF_REPORT_ONLY:
                # draw the empty slot rather than silently closing the gap, so the
                # reader can see WHY there is no teacher bar here
                out.append(rect(x, T + ph - 16, bw, 16, "none", rx=2,
                                stroke="#c9d1d9", dash="3 3"))
                out.append(text(x + bw / 2, T + ph - 22, "n/a", size=9,
                                fill="#98a2ad", anchor="middle"))
                continue
            v = agg.get(TEACHER_KEYS.get(key, key) if is_teacher else key)
            flagged = is_teacher and key in TEACHER_KEYS
            y = T + ph - v * ph
            out.append(rect(x, y, bw, v * ph, colour, rx=2))
            lab = f"{v:.3f}"
            out.append(text(x + bw / 2, y - 6,
                            (lab[1:] if lab.startswith("0.") else lab)
                            + ("†" if flagged else ""),
                            size=9.5, fill=colour, anchor="middle", weight="600"))
        out.append(text(gx + gw / 2, T + ph + 22, label, size=12, fill=FG, anchor="middle"))

    out.append(line(L, T + ph, L + pw, T + ph, "#b9c0c8", 1))

    lx = L
    for name, _, colour in models:
        out.append(rect(lx, H - 74, 11, 11, colour, rx=2))
        out.append(text(lx + 16, H - 64, name, size=12, fill=FG))
        lx += 16 + len(name) * 6.7 + 22

    for i, note in enumerate((
        "n/a — the teacher IS the reference every student is scored against, so it has "
        "no precision/recall of its own.",
        "† the teacher's found number is its found=true RATE, not the accuracy of that "
        "flag against a reference.",
        "Base / teacher / SFT: the 2026-08-08 run-set. GRPO: the 2026-08-10 re-run on "
        "the identical plan, where SFT re-scored .559 F1.",
    )):
        out.append(text(L, H - 40 + i * 14, note, size=10.5, fill=MUTED))

    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}" role="img" aria-label="Eval metrics for four '
            f'models">\n<title>Eval metrics across four models</title>\n'
            f'{rect(0, 0, W, H, BG, rx=6, stroke=FRAME)}\n' + "\n".join(out) + "\n</svg>\n")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    path = IMG / "eval-metrics.svg"
    path.write_text(chart_eval_models(), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.1f} KB)")
