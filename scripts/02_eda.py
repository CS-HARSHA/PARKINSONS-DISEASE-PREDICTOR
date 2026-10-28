from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from src import config
from src.data_loading import load_processed
from src.eda import summarize_sessions, subject_summary_rows, write_subject_summary

def _plot_group(subject_rows, field, ylabel, filename, title):
    labels = ['Control', 'PD']
    values = [[r[field] for r in subject_rows if r['label_name'] == label] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_ylabel(ylabel); ax.set_title(title); fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=160); plt.close(fig)

def _plot_hold_vs_volatility(subject_rows):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'Control': '#2c7fb8', 'PD': '#d95f0e'}
    markers = {'MIT-CS1PD': 'o', 'MIT-CS2PD': 's'}
    for row in subject_rows:
        ax.scatter(row['mean_hold'], row['mean_abs_consecutive_change'], color=colors[row['label_name']], marker=markers[row['cohort']], alpha=0.8, edgecolor='white', linewidth=0.5)
    for label, color in colors.items():
        ax.scatter([], [], color=color, marker='o', label=label)
    for cohort, marker in markers.items():
        ax.scatter([], [], color='gray', marker=marker, label=cohort)
    ax.set_xlabel('Subject mean hold time (seconds)')
    ax.set_ylabel('Mean absolute consecutive change (seconds)')
    ax.set_title('Hold time versus timing volatility')
    ax.legend(title='Color = label; marker = cohort')
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / 'hold_vs_volatility.png', dpi=180); plt.close(fig)

def main():
    sessions = load_processed()
    summary = summarize_sessions(sessions)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True); config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRICS_DIR / 'eda_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_subject_summary(sessions, config.PROCESSED_DIR / 'subject_eda_summary.csv')
    subject_rows = subject_summary_rows(sessions)
    _plot_hold_vs_volatility(subject_rows)
    _plot_group(subject_rows, 'mean_hold', 'Subject mean hold time (seconds)', 'hold_time_by_label.png', 'Hold time by label')
    _plot_group(subject_rows, 'mean_abs_consecutive_change', 'Mean absolute consecutive change (seconds)', 'consecutive_change_by_label.png', 'Consecutive hold-time change by label')
    _plot_group(subject_rows, 'macd', 'Subject MACD (seconds)', 'macd_by_label.png', 'MACD by label')
    lines = ['# Exploratory data analysis', '', f"The processed dataset contains **{summary['total_subjects']} subjects** and **{summary['total_sessions']} valid sessions**.", '', '## Why MACD is a reasonable feature', '', 'The scientific motivation is dynamic rather than purely distributional: Parkinsonian motor impairment may appear as less consistent changes from one key hold time to the next. Therefore, this EDA compares the absolute consecutive change in hold times between controls and PD subjects, then compares the corridor-restricted MACD feature used by the paper.', '', 'All group summaries below are calculated at the subject level, so a participant with many keystrokes does not count as thousands of independent observations.', '', '## Cohort summary', '', '| Cohort | Subjects | PD | Controls | Sessions | Keystrokes min / median / max |', '|---|---:|---:|---:|---:|---:|']
    for cohort, row in summary['by_cohort'].items():
        lines.append(f"| {cohort} | {row['subjects']} | {row['pd_subjects']} | {row['control_subjects']} | {row['sessions']} | {row['keystrokes_min']} / {row['keystrokes_median']:.0f} / {row['keystrokes_max']} |")
    lines += ['', '## Subject-level group comparison', '', '| Group | Subjects | Mean hold | Hold-time SD | Mean absolute consecutive change | Mean MACD | Median MACD |', '|---|---:|---:|---:|---:|---:|---:|']
    for key, row in summary['subject_level_group_summary'].items():
        lines.append(f"| {key} | {row['n_subjects']} | {row['mean_hold_subject_mean']:.4f} | {row['mean_stdev_hold']:.4f} | {row['mean_abs_consecutive_change']:.4f} | {row['mean_macd']:.4f} | {row['median_macd']:.4f} |")
    lines += ['', '## Figures', '', '- `hold_time_by_label.png`: subject-level average hold time.', '- `consecutive_change_by_label.png`: subject-level average absolute change between consecutive hold times.', '- `macd_by_label.png`: subject-level corridor-restricted MACD.', '- `hold_vs_volatility.png`: each subject plotted by mean hold time and consecutive timing volatility; color indicates label and marker indicates cohort.', '', 'The scatter plot makes the feature motivation visible: PD subjects may occupy a higher-volatility region. These plots support the feature motivation if the PD and control distributions differ, but they are descriptive only. They do not prove causality and they are not used to choose a test-set threshold or claim clinical validity.', '', f"Session-level MACD ranges from **{summary['macd_min']:.6f}** to **{summary['macd_max']:.6f}**, with median **{summary['macd_median']:.6f}**."]
    (config.ROOT / 'reports' / 'eda.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
if __name__ == '__main__': main()
