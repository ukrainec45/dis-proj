"""Standalone figures and CSV for scientific inspection; no inference from empty bins."""
import csv
from pathlib import Path
import numpy as np


def localizer_report(rows):
    """Outcomes of the unchanged localizer, not claims about predictor improvement."""
    result = {}
    for condition in sorted({r['condition'] for r in rows}):
        selected = [r for r in rows if r['condition'] == condition]
        labels = [r['label'] for r in selected]
        errors = np.asarray([v['error_after_m'] for v in labels])
        times = [r['localization_profile']['elapsed_s'] for r in selected if 'localization_profile' in r]
        result[condition] = dict(count=len(selected), accepted_count=sum(v['accepted'] for v in labels),
            improved_count=sum(v['class_index'] >= 5 for v in labels),
            worsened_count=sum(1 <= v['class_index'] <= 3 for v in labels),
            mean_error_after_m=float(errors.mean()), median_error_after_m=float(np.median(errors)),
            p90_error_after_m=float(np.quantile(errors, .9)),
            within_radius={str(radius): float((errors <= radius).mean()) for radius in (1, 5, 10, 20)},
            mean_localization_s=float(np.mean(times)) if times else None,
            p95_localization_s=float(np.quantile(times, .95)) if times else None)
    paired = {}
    by_case = {}
    for row in rows:
        key = (row['scene_id'], row['sample_id'], row['sigma_xy_m'])
        by_case.setdefault(key, {})[row['condition']] = row
    for condition in ('fog_extreme', 'smoke_extreme'):
        differences = [group[condition]['label']['error_after_m']-group['clear']['label']['error_after_m']
                       for group in by_case.values() if condition in group and 'clear' in group]
        paired[condition] = dict(pairs=len(differences),
            mean_error_increase_vs_clear_m=float(np.mean(differences)) if differences else None)
    return dict(by_condition=result, paired_vs_clear=paired,
                localizer_adapted=False, visibility_is_synthetic=True)


def write_figures(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    selected = summary['selected']
    score = summary['classification'][selected]
    with (output/'scores.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['candidate', 'log_loss', 'brier_score', 'sample_count', 'outside_training_ranges'])
        for key, value in summary['classification'].items():
            writer.writerow([key, value['log_loss'], value['brier_score'], value['sample_count'], value['outside_training_ranges']])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, event in zip(axes, ('accept', 'improve', 'worsen')):
        bins = [b for b in score['events'][event]['reliability'] if b['count']]
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.scatter([b['predicted'] for b in bins], [b['observed'] for b in bins])
        for b in bins:
            ax.annotate(f"n={b['count']}", (b['predicted'], b['observed']), xytext=(3, 4), textcoords='offset points', fontsize=8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Predicted probability', ylabel='Observed frequency', title=event)
    figure.suptitle(selected+'; dependent attempts, no flight-risk guarantee', fontsize=10)
    figure.savefig(output/'calibration.png', dpi=180)
    plt.close(figure)
    groups = summary['grouped'][selected]['scene_id']
    figure, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.bar(list(groups), [v['log_loss'] for v in groups.values()])
    ax.set(ylabel='Log loss (lower is better)', title='Performance by held-out territory: '+selected)
    figure.savefig(output/'territories.png', dpi=180)
    plt.close(figure)
    groups = summary['localizer_outcomes']['by_condition']
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].bar(list(groups), [g['median_error_after_m'] for g in groups.values()])
    axes[0].set(ylabel='Median error after attempt (m)', title='Unchanged localizer')
    axes[1].bar(list(groups), [g['accepted_count']/g['count'] for g in groups.values()])
    axes[1].set(ylabel='Accepted fraction', ylim=(0, 1), title='Synthetic visibility conditions')
    figure.savefig(output/'localizer_visibility.png', dpi=180)
    plt.close(figure)
