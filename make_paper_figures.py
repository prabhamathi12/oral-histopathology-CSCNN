"""
Redraws the summary figures from the outputs written by cscnn_pipeline.py.

    python make_paper_figures.py --work /content/drive/MyDrive/CSCNN_work --out figures
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
LABELS = {"tongue": "a  Tongue biopsies", "ndbufes": "b  NDB-UFES", "rahman": "c  Rahman et al."}
PROPOSED = "cscnn_v3_5blk_do0.3"


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def enhancement_ranks(work, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, d in zip(axes, LABELS):
        rk = pd.read_csv(f"{work}/{d}/enhancement/Table_enh_mean_ranks.csv", index_col=0)
        im = ax.imshow(rk.values, cmap="RdYlGn_r", vmin=1, vmax=len(rk))
        ax.set_xticks(range(rk.shape[1]), rk.columns, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(rk.shape[0]), rk.index)
        for i in range(rk.shape[0]):
            for j in range(rk.shape[1]):
                ax.text(j, i, f"{rk.values[i, j]:.1f}", ha="center", va="center", fontsize=7.5)
        ax.set_title(LABELS[d], loc="left", fontweight="bold")
    fig.colorbar(im, ax=axes, fraction=0.015, label="Mean rank (1 = best)")
    save(fig, out, "enhancement_mean_ranks")


def leakage(work, out):
    lk = pd.read_csv(f"{work}/tongue/leakage_demo_v3/leakage_demo_all_runs.csv")
    names = ["Legacy protocol\n(augment, then split)", "Corrected protocol\n(split, then augment)"]
    lk["prot"] = np.where(lk.protocol.str.startswith("Legacy"), names[0], names[1])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (m, lab) in zip(axes, [("accuracy", "Test accuracy"), ("balanced_accuracy", "Test balanced accuracy")]):
        for k, (p, c) in enumerate(zip(names, ["#7F7F7F", "#C0392B"])):
            g = lk[lk.prot == p]
            ok = g.balanced_accuracy > 0.3
            ax.scatter(np.full(ok.sum(), k) + np.linspace(-0.08, 0.08, ok.sum()), g[ok][m], s=45, color=c, zorder=3)
            ax.scatter(np.full((~ok).sum(), k), g[~ok][m], s=45, facecolor="white", edgecolor=c, zorder=3,
                       label="Run failed to converge" if (~ok).any() else None)
            ax.hlines(g[ok][m].median(), k - 0.2, k + 0.2, color=c, lw=2)
        ax.set_xticks([0, 1], names); ax.set_ylim(0, 1.02); ax.set_ylabel(lab)
    axes[0].legend(fontsize=8, loc="lower left", frameon=False)
    save(fig, out, "augmentation_order")


def cscnn_tongue(work, out):
    P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{work}/tongue/results_v2/{PROPOSED}/*/test_predictions.csv"))])
    cls = sorted(P.y_true.unique())
    y = P.y_true.map({c: i for i, c in enumerate(cls)}).values
    pr = P[[f"p_{c}" for c in cls]].values
    cm = confusion_matrix(y, pr.argmax(1)); cmn = cm / cm.sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.35})
    axes[0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)):
            axes[0].text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.2f})", ha="center", va="center", fontsize=8,
                         color="white" if cmn[i, j] > 0.5 else "black")
    axes[0].set_xticks(range(len(cls)), cls, rotation=20, fontsize=7); axes[0].set_yticks(range(len(cls)), cls, fontsize=7)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Reference")
    yb = label_binarize(y, classes=range(len(cls)))
    for i, c in enumerate(cls):
        f, t, _ = roc_curve(yb[:, i], pr[:, i]); axes[1].plot(f, t, label=f"{c} (AUC = {auc(f, t):.3f})")
    axes[1].plot([0, 1], [0, 1], color="grey", lw=0.8); axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
    save(fig, out, "cscnn_tongue_cm_roc")


def params_vs_ba(work, out):
    show = {"MobileNetV2", "MobileNetV3-Small", "MobileNetV3-Large", "EfficientNetB0", "VGG16", "ResNet50"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, d in zip(axes, LABELS):
        t = pd.read_csv(f"{work}/{d}/figures_v2/Table_model_comparison_fold1_all_models.csv")
        t["ba"] = t.balanced_accuracy.str.split(" ").str[0].astype(float)
        best = t.loc[t.ba.idxmax(), "model"]
        for _, r in t.iterrows():
            prop = r.model.startswith("CSCNN")
            ax.scatter(r.params, r.ba, s=110 if prop else 28, color="#C0392B" if prop else "#2E86C1",
                       marker="*" if prop else "o", zorder=3)
            if prop or r.model in show or r.model == best:
                ax.annotate("CSCNN" if prop else r.model, (r.params, r.ba), fontsize=7, xytext=(4, -3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("Number of parameters (log scale)"); ax.set_ylabel("Balanced accuracy, fold 1")
        ax.set_title(LABELS[d], loc="left", fontweight="bold")
    save(fig, out, "params_vs_balanced_accuracy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/content/drive/MyDrive/CSCNN_work")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    enhancement_ranks(a.work, a.out); leakage(a.work, a.out); cscnn_tongue(a.work, a.out); params_vs_ba(a.work, a.out)
    print("figures written to", a.out)
