"""
Tables and figures from ultrabot_sim/runs/:

    python -m ultrabot_sim.analyze                 # all summaries
    python -m ultrabot_sim.analyze --tag check     # one run

Writes runs/figures/:
    comparison.md / comparison.csv   shortest vs safest vs recommended
                                     across all runs (one row per procedure)
    clearance_<tag>_<kind>.png       planned clearance along the path
    tip_error_<tag>.png              IK error, skin RCM deviation and
                                     out-of-plane over time
    depth_<tag>.png                  insertion depth vs target, safety state
    lstm_error_hist.png              one-step LSTM error (mm) of all runs
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                          # noqa: E402

from ultrabot_sim import config as cfg                      # noqa: E402

COLS = ["tag", "kind", "outcome", "success", "insertion_mm",
        "angle_chestwall_deg", "planned_min_clear_mm", "clear_struct",
        "measured_min_clear_mm", "ligaments", "notch_in_lesion_pct",
        "throw_axis_to_centre_mm", "max_ik_err_mm", "max_rcm_mm",
        "max_out_of_plane_mm", "time_s", "abort_reason"]


def rows_from(summary):
    out = []
    for p in summary.get("procedures", []):
        pl = p.get("plan") or {}
        f = p.get("fire") or {}
        out.append({
            "tag": summary.get("tag"), "kind": p["kind"],
            "outcome": p["outcome"], "success": p["success"],
            "insertion_mm": pl.get("insertion_mm"),
            "angle_chestwall_deg": pl.get("angle_chestwall_deg"),
            "planned_min_clear_mm": pl.get("min_clearance_mm"),
            "clear_struct": pl.get("min_clearance_struct"),
            "measured_min_clear_mm": p.get("min_needle_clearance_mm"),
            "ligaments": pl.get("ligaments_crossed"),
            "notch_in_lesion_pct": (100 * f["notch_in_lesion_frac"]
                                    if "notch_in_lesion_frac" in f else None),
            "throw_axis_to_centre_mm": f.get("throw_miss_mm"),
            "max_ik_err_mm": p.get("max_ik_err_mm"),
            "max_rcm_mm": p.get("max_rcm_dev_mm"),
            "max_out_of_plane_mm": p.get("max_out_of_plane_mm"),
            "time_s": (p.get("t_end", 0) - p.get("t_start", 0)),
            "abort_reason": p.get("abort_reason", ""),
        })
    return out


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}"
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = cfg.RUNS_DIR
    figs = runs / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    pat = f"summary_{a.tag}.json" if a.tag else "summary_*.json"
    files = sorted(runs.glob(pat))
    if not files:
        print(f"no {pat} in {runs}; run the simulation first")
        return 1
    rows, lstm_err = [], []
    for fpath in files:
        s = json.loads(fpath.read_text())
        rows += rows_from(s)
        lstm_err += s.get("lstm_errors_mm", [])
        tag = s.get("tag")
        for p in s.get("procedures", []):
            pl = p.get("plan") or {}
            prof = None
            for ps in s.get("plan_sets_last", []):
                q = ps.get(p["kind"])
                if q and q.get("insertion_mm") == pl.get("insertion_mm"):
                    prof = q
            if pl:
                # recompute the profile from the stored plan
                from ultrabot_sim.planner import clearance_profile
                from ultrabot_sim.anatomy import Anatomy, LesionGeom
                from ultrabot_sim.us_geometry import USCalib
                an_d = (prof or {}).get("anatomy") or \
                    next((x.get("anatomy") for x in
                          s.get("plan_sets_last", [])), None)
                if an_d and an_d.get("lesion"):
                    cal = USCalib.from_config()
                    an = Anatomy(cal, LesionGeom(*an_d["lesion"]))
                    L = pl["insertion_mm"] + cfg.THROW_MM
                    ss, cc, fam = clearance_profile(an, np.array(pl["entry_mm"]),
                                                    np.array(pl["direction"]),
                                                    L)
                    plt.figure(figsize=(6, 3.2))
                    plt.plot(ss, cc, lw=2)
                    plt.axhline(cfg.VESSEL_MARGIN_MM, color="r", ls="--",
                                lw=1, label="vessel margin")
                    plt.axhline(cfg.CHESTWALL_CLEARANCE_MM, color="m",
                                ls=":", lw=1, label="chest-wall margin")
                    plt.axvspan(pl["insertion_mm"], L, alpha=0.15,
                                label="throw")
                    plt.xlabel("distance along the needle from the skin (mm)")
                    plt.ylabel("clearance to hard structures (mm)")
                    plt.title(f"{tag} {p['kind']} path")
                    plt.legend(fontsize=7)
                    plt.tight_layout()
                    plt.savefig(figs / f"clearance_{tag}_{p['kind']}.png",
                                dpi=130)
                    plt.close()
        mpath = runs / f"metrics_{tag}.csv"
        if mpath.is_file():
            with open(mpath) as fh:
                rd = list(csv.DictReader(fh))
            t = np.array([float(r["t"]) for r in rd])

            def col(k):
                return np.array([float(r[k]) if r[k] not in ("", None)
                                 else np.nan for r in rd])
            fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
            ax[0].plot(t, col("ik_err_mm"))
            ax[0].set_ylabel("IK error (mm)")
            ax[1].plot(t, col("rcm_dev_mm"))
            ax[1].set_ylabel("skin RCM (mm)")
            ax[2].plot(t, col("out_of_plane_mm"))
            ax[2].set_ylabel("out of plane (mm)")
            ax[2].set_xlabel("simulation time (s)")
            ax[0].set_title(f"robot accuracy - {tag}")
            fig.tight_layout()
            fig.savefig(figs / f"tip_error_{tag}.png", dpi=130)
            plt.close(fig)
            fig = plt.figure(figsize=(7, 3))
            plt.plot(t, col("depth_mm"), label="tip depth")
            plt.plot(t, col("target_depth_mm"), "--", label="target depth")
            plt.ylabel("mm")
            plt.xlabel("simulation time (s)")
            plt.legend()
            plt.tight_layout()
            fig.savefig(figs / f"depth_{tag}.png", dpi=130)
            plt.close(fig)
    if lstm_err:
        plt.figure(figsize=(6, 3))
        plt.hist(lstm_err, bins=30)
        plt.xlabel("LSTM one-step error (mm)")
        plt.ylabel("count")
        plt.title(f"median {np.median(lstm_err):.2f} mm, n={len(lstm_err)}")
        plt.tight_layout()
        plt.savefig(figs / "lstm_error_hist.png", dpi=130)
        plt.close()
    with open(figs / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    md += ["| " + " | ".join(fmt(r[c]) for c in COLS) + " |" for r in rows]
    (figs / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    for kind in ("shortest", "safest", "recommended"):
        k = [r for r in rows if r["kind"] == kind]
        if k:
            ok = sum(1 for r in k if r["success"])
            print(f"\n{kind}: {ok}/{len(k)} successful, median insertion "
                  f"{np.nanmedian([r['insertion_mm'] or np.nan for r in k]):.1f}"
                  f" mm, median planned clearance "
                  f"{np.nanmedian([r['planned_min_clear_mm'] or np.nan for r in k]):.1f} mm")
    print(f"\nfigures in {figs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
