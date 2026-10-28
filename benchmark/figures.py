"""The diagnostic figure: "Where is the variance?"

Panel A, how much raters agree. ICC(A,1) on novelty for three kinds of pair: person with person
(every pair of human raters as a dot, all raters together as a bar), AI judge with AI judge
(every pair of judges, all judges together) and each AI judge with itself across its repeated
samples at temperature 1. The numbers come from ``agree.agreement_ladder``.

Panel B, novelty ratings and variance. Each idea's reference novelty (x) and the spread of novelty
ratings (y): the SD across the human raters, and the mean SD of the AI judges' own repeated
samples. High points are contested ideas.

    python -m benchmark.figures                         # the example folder
    python -m benchmark.figures --data-dir my_study/    # ratings_long.csv and judge_scores.csv

Palette: the dataviz reference categorical slots (validated, light mode).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import agree  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SLOT = {"people": "#2a78d6", "judge": "#1baf7a"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _pool(ratings: pd.DataFrame, conds: set[str] | None) -> pd.DataFrame:
    return ratings if conds is None or "condition" not in ratings else ratings[ratings.condition.isin(conds)]


def _panel_b(b, ratings: pd.DataFrame, ideas: pd.DataFrame, judge_scores: pd.DataFrame | None,
             conds: set[str] | None) -> None:
    _style(b)
    ref = agree.idea_summary(ratings, conds)
    nov = ref[ref.dimension == "novelty"].merge(ideas[["idea_id", "idea_text"]], on="idea_id")
    n_people = _pool(ratings, conds).rater_id.nunique()
    b.scatter(nov["mean"], nov["sd"], s=46, color=SLOT["people"], edgecolor=SURFACE, linewidth=1.0, zorder=3,
              label=f"Human raters (SD across {n_people} people)")
    if judge_scores is not None and len(judge_scores):
        p6 = judge_scores[(judge_scores.perturbation_id == "P6_temperature_sampled") & (judge_scores.dimension == "novelty")]
        jsd = p6.dropna(subset=["score"]).groupby(["judge", "idea_id"]).score.std().groupby("idea_id").mean().rename("judge_sd")
        jm = nov.merge(jsd, left_on="idea_id", right_index=True)
        if len(jm):
            b.scatter(jm["mean"], jm["judge_sd"], s=60, marker="D", color=SLOT["judge"], edgecolor=SURFACE, linewidth=1.0,
                      zorder=3, label=f"AI judges (SD across {p6.sample_index.nunique()} samples, mean of {p6.judge.nunique()} judges)")
            b.set_ylim(-0.05, None)
    b.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, labelcolor=INK2)
    lab = pd.concat([nov.nlargest(3, "sd"), nov.nsmallest(2, "sd")])
    for r in lab.itertuples():
        b.annotate(r.idea_text, (getattr(r, "mean"), r.sd), xytext=(7, -3), textcoords="offset points", fontsize=8, color=INK)
    b.set_xlabel("Reference novelty (mean of human raters)", color=INK2, fontsize=9.5)
    b.set_ylabel("Variance in novelty ratings (SD)", color=INK2, fontsize=9.5)
    b.set_title("B. Novelty ratings and variance", loc="left", color=INK, fontsize=11, fontweight="bold")
    b.set_xlim(nov["mean"].min() - 0.3, nov["mean"].max() + 1.4)


def headline_figure(ratings: pd.DataFrame, ideas: pd.DataFrame, out: Path,
                    judge_scores: pd.DataFrame | None = None, title_note: str = "",
                    reference_conditions: set[str] | None = None) -> Path:
    """The agreement ladder (A) and novelty ratings and variance (B). The reference is all human
    raters, or those in ``reference_conditions``."""
    import itertools

    import numpy as np
    from matplotlib.lines import Line2D

    conds = reference_conditions
    pool = _pool(ratings, conds)
    base = pool[pool.dimension == "novelty"].dropna(subset=["score"])
    human_all = agree._icc_a1(base, "rater_id")
    human_pairs = np.array([agree._icc_a1(base[base.rater_id.isin(p)], "rater_id")
                            for p in itertools.combinations(sorted(base.rater_id.unique()), 2)])
    rows = [(2, "Person with\nperson", human_pairs, human_all, SLOT["people"], "o", f"{len(human_pairs)} pairs of people")]
    if judge_scores is not None and len(judge_scores):
        js = judge_scores.dropna(subset=["score"])
        js = js[js.dimension == "novelty"]
        p0 = js[(js.perturbation_id == "P0_baseline") & (js.sample_index == 0)].groupby(["judge", "idea_id"]).score.mean().reset_index()
        ai_all = agree._icc_a1(p0, "judge")
        ai_pairs = np.array([agree._icc_a1(p0[p0.judge.isin(p)], "judge") for p in itertools.combinations(sorted(p0.judge.unique()), 2)])
        lad = agree.agreement_ladder(ratings, judge_scores, conds)
        self_icc = lad[(lad.dimension == "novelty") & (lad.pair == "AI judge with itself")].icc_a1.to_numpy()
        rows += [(1, "AI judge with\nAI judge", ai_pairs, ai_all, SLOT["judge"], "D", f"{len(ai_pairs)} pairs of judges"),
                 (0, "AI judge with\nitself", self_icc, None, SLOT["judge"], "D", f"{len(self_icc)} judges, {js.sample_index.nunique()} samples each")]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.3), dpi=200, facecolor=SURFACE)
    _style(a)
    a.grid(False, axis="y")
    xmin = np.floor(human_pairs.min() * 10) / 10 - 0.05
    rng = np.random.default_rng(7)
    for y, _, vals, pooled, color, marker, note in rows:
        many = len(vals) > 20
        jitter = rng.uniform(-0.18, 0.18, len(vals)) if many else np.zeros(len(vals))
        a.scatter(vals, y + jitter, s=10 if many else 55, marker=marker, color=color, alpha=0.35 if many else 0.9,
                  edgecolor="none" if many else SURFACE, linewidth=1.0, zorder=3)
        if pooled is not None:
            a.plot([pooled, pooled], [y - 0.3, y + 0.3], color=INK, linewidth=2, zorder=4)
            a.annotate(f"{pooled:.2f}", (pooled, y + 0.3), xytext=(0, 3), textcoords="offset points", ha="center",
                       va="bottom", fontsize=9, color=INK, fontweight="bold")
        else:
            a.annotate(f"{vals.min():.2f} to {vals.max():.2f}", (vals.max(), y + 0.12), xytext=(0, 6),
                       textcoords="offset points", ha="right", va="bottom", fontsize=9, color=INK, fontweight="bold")
        a.text(xmin + 0.02, y - 0.42, note, fontsize=7.5, color=MUTED, ha="left", va="center")
    a.set_yticks([r[0] for r in rows])
    a.set_yticklabels([r[1] for r in rows], fontsize=9, color=INK2)
    a.set_ylim(-0.6, 2.6)
    a.set_xlim(xmin, 1.12)
    a.set_xlabel("Agreement on novelty (ICC(A,1); 1 = identical scores)", color=INK2, fontsize=9.5)
    a.set_title("A. How much raters agree", loc="left", color=INK, fontsize=11, fontweight="bold")
    a.legend(handles=[Line2D([], [], color=INK, lw=2, label="All raters of that kind together"),
                      Line2D([], [], marker="o", ls="", color=SLOT["people"], alpha=0.6, ms=5, label="One pair")],
             frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, labelcolor=INK2)
    _panel_b(b, ratings, ideas, judge_scores, conds)
    if title_note:
        fig.text(0.01, 0.005, title_note, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from . import RESULTS_DIR
    from .load import EXAMPLE_DIR

    p = argparse.ArgumentParser(prog="python -m benchmark.figures")
    p.add_argument("--data-dir", default=str(EXAMPLE_DIR), help="folder with ideas.csv, ratings_long.csv, judge_scores.csv")
    p.add_argument("--scores", default=None, help="judge_scores.csv (default: the one in --data-dir)")
    p.add_argument("--reference-condition", default=None, help="raters in this condition form the reference")
    p.add_argument("--out", default=str(RESULTS_DIR / "figure.png"))
    a = p.parse_args(argv)
    d = Path(a.data_dir)
    js_path = Path(a.scores) if a.scores else d / "judge_scores.csv"
    js = pd.read_csv(js_path) if js_path.exists() else None
    conds = {a.reference_condition} if a.reference_condition else None
    print(headline_figure(pd.read_csv(d / "ratings_long.csv"), pd.read_csv(d / "ideas.csv"), Path(a.out), js,
                          reference_conditions=conds))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
