"""Generate diagnostic figures for Phase 7 Baseline Model Comparison."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports/modelling/baseline_cv_summary.csv"
FOLD_PATH = ROOT / "reports/modelling/baseline_cv_fold_results.csv"
FIG_DIR = ROOT / "reports/modelling/figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    summary_df = pd.read_csv(SUMMARY_PATH)
    fold_df = pd.read_csv(FOLD_PATH)

    # 1. Balanced Accuracy Comparison Bar Plot
    plt.figure(figsize=(10, 6))
    order = summary_df.sort_values(by="balanced_accuracy_mean", ascending=False)["model"]
    palette = sns.color_palette("mako", n_colors=len(order))
    
    ax = sns.barplot(
        data=summary_df,
        x="balanced_accuracy_mean",
        y="model",
        order=order,
        palette=palette,
        hue="model",
        legend=False,
    )
    
    # Add error bars for std
    for i, model in enumerate(order):
        row = summary_df[summary_df["model"] == model].iloc[0]
        mean_val = row["balanced_accuracy_mean"]
        std_val = row["balanced_accuracy_std"]
        ax.errorbar(
            x=mean_val,
            y=i,
            xerr=std_val,
            fmt="none",
            c="black",
            capsize=4,
            capthick=1.5,
        )
        ax.text(
            mean_val + std_val + 0.01,
            i,
            f"{mean_val:.3f} ± {std_val:.3f}",
            va="center",
            fontsize=10,
            fontweight="semibold",
        )

    plt.axvline(0.5, color="firebrick", linestyle="--", linewidth=1.5, label="Chance Floor (0.50)")
    plt.xlim(0.4, 0.72)
    plt.title("Phase 7: Baseline Model Cross-Validation Balanced Accuracy (5-Fold CV, N=477)", pad=15)
    plt.xlabel("Mean Balanced Accuracy")
    plt.ylabel("Candidate Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig1_path = FIG_DIR / "fig1_balanced_accuracy_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # 2. Sensitivity vs Specificity Tradeoff
    plt.figure(figsize=(9, 6))
    tradeoff_df = summary_df.melt(
        id_vars=["model"],
        value_vars=["sensitivity_mean", "specificity_mean"],
        var_name="Metric",
        value_name="Score",
    )
    tradeoff_df["Metric"] = tradeoff_df["Metric"].map({
        "sensitivity_mean": "Sensitivity (PD Recall)",
        "specificity_mean": "Specificity (Control Recall)",
    })

    ax = sns.barplot(
        data=tradeoff_df,
        x="model",
        y="Score",
        hue="Metric",
        palette=["#2b5c8f", "#d95f02"],
    )
    plt.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    plt.title("Phase 7: Sensitivity vs. Specificity Tradeoff Across Baselines", pad=15)
    plt.xlabel("Candidate Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.08)
    plt.xticks(rotation=20, ha="right")
    plt.legend(loc="upper right")
    plt.tight_layout()
    fig2_path = FIG_DIR / "fig2_sensitivity_vs_specificity.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # 3. ROC-AUC vs Balanced Accuracy Scatter
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=summary_df,
        x="roc_auc_mean",
        y="balanced_accuracy_mean",
        hue="model",
        s=160,
        palette="tab10",
        style="model",
    )
    for _, row in summary_df.iterrows():
        plt.annotate(
            row["model"],
            (row["roc_auc_mean"] + 0.005, row["balanced_accuracy_mean"] + 0.002),
            fontsize=9.5,
        )
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.title("Phase 7: ROC-AUC vs. Balanced Accuracy (5-Fold CV)", pad=15)
    plt.xlabel("Mean ROC-AUC")
    plt.ylabel("Mean Balanced Accuracy")
    plt.xlim(0.48, 0.70)
    plt.ylim(0.48, 0.64)
    plt.tight_layout()
    fig3_path = FIG_DIR / "fig3_roc_auc_vs_balanced_acc.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # 4. Aggregate Out-of-Fold Confusion Matrix for Random Forest
    rf_folds = fold_df[fold_df["model"] == "Random Forest"]
    total_tn = int(rf_folds["tn"].sum())
    total_fp = int(rf_folds["fp"].sum())
    total_fn = int(rf_folds["fn"].sum())
    total_tp = int(rf_folds["tp"].sum())

    cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(7, 6))
    labels = np.array([
        [f"{total_tn}\n({cm_norm[0,0]:.1%})", f"{total_fp}\n({cm_norm[0,1]:.1%})"],
        [f"{total_fn}\n({cm_norm[1,0]:.1%})", f"{total_tp}\n({cm_norm[1,1]:.1%})"],
    ])
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted Control (0)", "Predicted PD (1)"],
        yticklabels=["Actual Control (0)", "Actual PD (1)"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title("Random Forest (Top Model): Aggregate Out-of-Fold Confusion Matrix\n(5-Fold CV, N=477)", pad=15)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    fig4_path = FIG_DIR / "fig4_rf_aggregate_confusion_matrix.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")


if __name__ == "__main__":
    main()
