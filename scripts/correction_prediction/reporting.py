"""Reports from measured inputs or explicitly synthetic tests; no invented scores."""
from pathlib import Path

import numpy as np

from .artifacts import write_once
from .contracts import probabilities


def bin_sensitivity(p, rows):
    """Preregistered coarsenings; no interpolation within the original bins.

    Compare candidate rankings within each target. Absolute log losses across
    different targets are not comparable. Continuous deltas supply the labels.
    """
    p = probabilities(p)
    if len(p) != len(rows):
        raise ValueError('Sensitivity row mismatch')
    representative = [-21, -10, -2, 0, 2, 10, 21]
    results = {}
    for edges in [(-5, -1, 1, 5), (-1, 1)]:
        q = np.zeros((len(p), len(edges)+2))
        q[:, 0] = p[:, 0]
        for i, value in enumerate(representative):
            q[:, np.searchsorted(edges, value, side='right')+1] += p[:, i+1]
        y = np.array([0 if not r['label']['accepted'] else
                      np.searchsorted(edges, r['label']['delta_m'], side='right')+1 for r in rows])
        results[str(edges)] = dict(edges_m=list(edges),
            log_loss=float(-np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean()),
            brier_score=float(np.square(q-np.eye(q.shape[1])[y]).sum(axis=1).mean()))
    return results


def write_score_report(output, evaluations, *, title, selected, final_test=False,
                       selection_note=None, scope_note=None):
    output = Path(output)
    lines = [f'# {title}', '', selection_note or f'Selected on working validation: `{selected}`.', '',
             '| Candidate | Log loss | Multiclass Brier | Accepted conditional RPS |',
             '|---|---:|---:|---:|']
    for name, scores in evaluations.items():
        s = scores.get('overall', scores)
        rps = s['conditional_accepted_rps']
        lines.append(f"| {name} | {s['log_loss']:.6f} | {s['brier_score']:.6f} | {rps if rps is not None else 'unavailable'} |")
    lines += ['', f'Final test evaluated: {final_test}. Validation passed: false.', '',
              'Research only. Technical failures and unsupported queries are separately reported.',
              'Frame repetitions are dependent. These scores do not certify route arrival or safety.',
              'Conditional RPS evaluates accepted outcomes only; log loss covers all eight classes.',
              'Rare/missing classes and few independent territories limit claims.']
    if scope_note:
        lines += ['', scope_note]
    write_once(output/'report.md', '\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, event in zip(axes, ('accept', 'improve', 'worsen')):
        for name, scores in evaluations.items():
            s = scores.get('overall', scores)
            bins = [r for r in s['events'][event]['reliability'] if r['count']]
            axis.plot([r['predicted'] for r in bins], [r['observed'] for r in bins], '.-', label=name)
        axis.plot([0, 1], [0, 1], '--', color='gray')
        axis.set(xlabel='Predicted probability', ylabel='Observed frequency', title=event, xlim=(0, 1), ylim=(0, 1))
    axes[-1].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(output/'calibration.png', dpi=160)
    plt.close(fig)
