"""python -m scripts.correction_prediction --help"""
import argparse
import json
from pathlib import Path

from .campaign import pilot_config
from .dataset import load_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description='Pre-flight correction prediction research')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('pilot-config', help='Print frozen 288-attempt design; no GPU/API call')
    audit = sub.add_parser('audit', help='Validate dataset integrity and summarize counts')
    audit.add_argument('dataset')
    diagnose = sub.add_parser('diagnose', help='Read saved exception messages and tracebacks; no GPU calls')
    diagnose.add_argument('output', help='Campaign directory containing records/')
    pilot = sub.add_parser('pilot', help='Run/resume GPU pilot on the frozen prior quick selection')
    pilot.add_argument('selection_csv')
    pilot.add_argument('output')
    pilot.add_argument('--checkpoint', help='Existing CAMP weights; otherwise download official weights to the experiment cache')
    train = sub.add_parser('train', help='Working splits only; requires agreed main dataset')
    train.add_argument('dataset')
    train.add_argument('output')
    pilot_baselines = sub.add_parser('pilot-baselines', help='Exploratory CPU baselines on pilot data; whole-group checks, no Jev')
    pilot_baselines.add_argument('dataset')
    pilot_baselines.add_argument('output')
    curves = sub.add_parser('learning-curve', help='Whole-territory curves on working validation only')
    curves.add_argument('dataset')
    curves.add_argument('output')
    test = sub.add_parser('evaluate', help='Explicit final-test evaluation of a frozen run')
    test.add_argument('run')
    test.add_argument('dataset')
    test.add_argument('output')
    for cmd in (train, test):
        cmd.add_argument('--jev-numeric')
        cmd.add_argument('--jev-map')
    jev = sub.add_parser('jev', help='Cache-only by default; explicit request budget enables API')
    jev.add_argument('dataset')
    jev.add_argument('output')
    jev.add_argument('--with-image', action='store_true')
    jev.add_argument('--max-requests', type=int, default=0)
    jev.add_argument('--cache-dir')
    jev.add_argument('--model', default='jev-1.14.0')
    jev.add_argument('--frozen-run', help='Only then request final-test predictions')
    opportunities = sub.add_parser('opportunities', help='One-attempt choice on observed final-test groups')
    opportunities.add_argument('run')
    opportunities.add_argument('dataset')
    opportunities.add_argument('evaluated')
    opportunities.add_argument('output')
    opportunities.add_argument('--navigation-scores', help='JSON mapping attempt_id to measured map-only score')
    args = parser.parse_args(argv)
    if args.command == 'pilot-config':
        result = pilot_config()
    elif args.command == 'audit':
        from collections import Counter
        data = load_dataset(args.dataset)
        result = dict(attempts=len(data['rows']), splits=dict(Counter(r['split'] for r in data['rows'])),
                      unique_frames=len({(r['scene_id'], r['sample_id']) for r in data['rows']}),
                      technical_errors=sum(r['label']['technical_error'] for r in data['rows']),
                      unsupported=sum(r['features'] is None for r in data['rows']))
    elif args.command == 'diagnose':
        from .diagnostics import diagnose_campaign
        result = diagnose_campaign(args.output)
    elif args.command == 'pilot':
        from .colab import initialize_backend
        from .campaign import run_pilot
        backend = initialize_backend(Path(args.output)/'runtime', args.checkpoint)
        result = run_pilot(args.selection_csv, backend, args.output)
    elif args.command == 'jev':
        from .jev_campaign import run_jev_predictions
        result = run_jev_predictions(args.dataset, args.output, with_image=args.with_image,
                                    max_requests=args.max_requests, model=args.model,
                                    cache_dir=args.cache_dir, frozen_run=args.frozen_run)
    elif args.command == 'pilot-baselines':
        from .pilot_baselines import run_pilot_baselines
        result = run_pilot_baselines(args.dataset, args.output)
    elif args.command == 'learning-curve':
        from .learning import learning_curve
        result = learning_curve(args.dataset, args.output)
    elif args.command == 'opportunities':
        from .opportunities import evaluate_opportunities
        from .artifacts import read
        nav = read(args.navigation_scores) if args.navigation_scores else None
        result = evaluate_opportunities(args.run, args.dataset, args.evaluated, args.output, nav)
    else:
        from .training import run_training, evaluate_frozen
        external = {k: v for k, v in [('jev_numeric', args.jev_numeric), ('jev_map', args.jev_map)] if v}
        if args.command == 'train':
            result = run_training(args.dataset, args.output, jev_predictions=external)
        else:
            result = evaluate_frozen(args.run, args.dataset, args.output, jev_predictions=external)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
