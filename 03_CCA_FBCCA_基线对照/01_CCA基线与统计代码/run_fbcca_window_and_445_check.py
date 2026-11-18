# -*- coding: utf-8 -*-
"""补充检验 4.2.2 连续 4-45 Hz 数据口径的 CCA/FBCCA 表现及 FBCCA 窗长扫描。"""

import csv
import json
from pathlib import Path

import numpy as np

from run_cca_baseline import (
    bandpass,
    cca_predict,
    compute_metrics,
    fbcca_predict,
    load_ads_day1,
    load_ads_day2,
    load_gnautilus,
    wolpaw_itr,
)

OUT_DIR = Path(r"E:\Master_Dissertation_Program\4.2_SSVEP算法补充与跨天实验\02_实验结果汇总")
NPZ_445_DIR = Path(r"E:\Master_Dissertation_Program\采集的脑电数据\4.2.2_重新采集数据处理\03_Dataset\NumPy")


def main() -> None:
    xs, ys = [], []
    for r in ("R01", "R02", "R03", "R04"):
        with np.load(NPZ_445_DIR / f"{r}_analysis_0p5to3p5_uv.npz", allow_pickle=False) as f:
            xs.append(f["X"].copy())
            ys.append(f["y"].copy())
    x_445 = np.concatenate(xs, axis=0)
    y_445 = np.concatenate(ys, axis=0)

    cont_results = {}
    for label, sub_x, sub_y in [
        ("ADS_day1_445cont_R03_val128", x_445[288:416], y_445[288:416]),
        ("ADS_day1_445cont_R04_test128", x_445[416:544], y_445[416:544]),
        ("ADS_day1_445cont_all544", x_445, y_445),
    ]:
        for nh in (1, 2, 3):
            yp, _ = cca_predict(sub_x, 500.0, nh)
            m = compute_metrics(sub_y, yp)
            cont_results[f"{label}__CCA_Nh{nh}"] = m
            print(f"{label:32s} | CCA Nh={nh} | {m['correct']:3d}/{m['count']:3d} ({m['accuracy']:6.2%}) | F1={m['macro_f1']:.4f}")
        yp_fb, _ = fbcca_predict(sub_x, 500.0, 3, [(5.0, 45.0), (11.0, 45.0), (17.0, 45.0)])
        m_fb = compute_metrics(sub_y, yp_fb)
        cont_results[f"{label}__FBCCA_Nh3"] = m_fb
        print(f"{label:32s} | FBCCA Nh=3 | {m_fb['correct']:3d}/{m_fb['count']:3d} ({m_fb['accuracy']:6.2%}) | F1={m_fb['macro_f1']:.4f}")

    # FBCCA window sweep
    gx, gy, gm = load_gnautilus()
    gx_3s = gx[:, :, 125:875]
    ax1, ay1, am1 = load_ads_day1()
    ax2, ay2 = load_ads_day2()
    ax1_bp = bandpass(ax1, 4.0, 45.0, 250.0)
    ax2_bp = bandpass(ax2, 4.0, 45.0, 250.0)

    win_rows = []
    targets = [
        ("gNautilus_test95", gx_3s[gm["test"]], gy[gm["test"]], [(5.0, 28.0), (11.0, 28.0), (17.0, 28.0)], 2),
        ("ADS_day1_test128_4to45Hz", ax1_bp[am1["test"]], ay1[am1["test"]], [(5.0, 45.0), (11.0, 45.0), (17.0, 45.0)], 3),
        ("ADS_day2_all128_4to45Hz", ax2_bp, ay2, [(5.0, 45.0), (11.0, 45.0), (17.0, 45.0)], 3),
    ]
    print("--- FBCCA Window Sweep ---")
    for ds_name, X, y, subbands, nh in targets:
        for w_s in (1.0, 1.5, 2.0, 2.5, 3.0):
            n_pts = int(round(w_s * 250))
            yp, _ = fbcca_predict(X[:, :, :n_pts], 250.0, nh, subbands)
            m = compute_metrics(y, yp)
            itr = wolpaw_itr(m["accuracy"], w_s)
            win_rows.append(
                {
                    "dataset": ds_name,
                    "method": f"FBCCA_3bands_Nh{nh}",
                    "window_s": w_s,
                    "correct": m["correct"],
                    "count": m["count"],
                    "accuracy": round(m["accuracy"], 6),
                    "macro_f1": round(m["macro_f1"], 6),
                    "itr_window_bpm": round(itr, 2),
                }
            )
            print(f"{ds_name:26s} | w={w_s:.1f}s | {m['correct']:3d}/{m['count']:3d} ({m['accuracy']:6.2%}) | F1={m['macro_f1']:.4f} | ITR={itr:.2f}")

    with (OUT_DIR / "FBCCA时间窗长度与ITR对比.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(win_rows[0]))
        writer.writeheader()
        writer.writerows(win_rows)
    (OUT_DIR / "4.2.2连续4到45Hz数据CCA核对.json").write_text(
        json.dumps(cont_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
