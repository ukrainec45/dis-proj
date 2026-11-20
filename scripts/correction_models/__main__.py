"""python -m scripts.correction_models --help"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Correction models: independent main experiment')
    commands = parser.add_subparsers(dest='command', required=True)
    unpack = commands.add_parser('unpack', help='Find and extract existing AnyVisLoc ZIPs; no downloads')
    unpack.add_argument('archive_root')
    unpack.add_argument('dataset_root')
    inventory = commands.add_parser('inventory', help='Read all scene metadata; no GPU')
    inventory.add_argument('root')
    inventory.add_argument('output')
    audit = commands.add_parser('audit-template', help='Create scene groups; geographic non-overlap remains unverified')
    audit.add_argument('inventory')
    audit.add_argument('output')
    prepare = commands.add_parser('prepare', help='Freeze 160 frames split by official scene groups')
    prepare.add_argument('inventory')
    prepare.add_argument('audit')
    prepare.add_argument('output')
    collect = commands.add_parser('collect', help='Run/resume a bounded part of the 1920 GPU attempts')
    collect.add_argument('selection')
    collect.add_argument('output')
    collect.add_argument('--runtime', required=True)
    collect.add_argument('--checkpoint')
    collect.add_argument('--max-new-attempts', type=int, default=120)
    collect.add_argument('--max-seconds', type=float, default=3300)
    for name in ('train', 'evaluate'):
        command = commands.add_parser(name, help='CPU model development' if name == 'train' else 'One-shot frozen final test')
        if name == 'evaluate':
            command.add_argument('run')
        command.add_argument('dataset')
        command.add_argument('output')
    inspect = commands.add_parser('inspect', help='Read integrity/counts without exposing final-test outcomes')
    inspect.add_argument('dataset')
    args = parser.parse_args()
    if args.command == 'unpack':
        from .archives import unpack_archives
        result = unpack_archives(args.archive_root, args.dataset_root)
    elif args.command == 'inventory':
        from .selection import inventory
        result = inventory(args.root, args.output)
    elif args.command == 'audit-template':
        from .selection import audit_template
        audit_template(args.inventory, args.output)
        result = dict(template=args.output, reviewed=False)
    elif args.command == 'prepare':
        from .selection import prepare_selection
        result = prepare_selection(args.inventory, args.audit, args.output)
    elif args.command == 'collect':
        from .environment import configure_cuda_determinism, verify_scipy_imports
        configure_cuda_determinism()
        verify_scipy_imports()
        # Reject missing/invalid selection before downloading model weights.
        from .selection import validate_selection
        from .artifacts import load_verified
        validate_selection(load_verified(args.selection))
        from .colab import initialize_backend
        from .campaign import run_collection
        backend = initialize_backend(Path(args.runtime), args.checkpoint)
        result = run_collection(args.selection, backend, args.output,
            max_new_attempts=args.max_new_attempts, max_seconds=args.max_seconds)
    elif args.command == 'train':
        from .training import run_training
        result = run_training(args.dataset, args.output)
        result = {k: result[k] for k in ('selected', 'split_counts', 'final_test_evaluated')}
    elif args.command == 'evaluate':
        from .training import evaluate_frozen
        result = evaluate_frozen(args.run, args.dataset, args.output)
        result = {k: result[k] for k in ('selected', 'supported_count', 'held_out_test_scenes', 'final_test_evaluated')}
    else:
        from .dataset import load_dataset, usable
        data = load_dataset(args.dataset)
        result = dict(completed_attempts=len(data['rows']),
            supported_by_split={s: len(usable(data['rows'], s)) for s in ('train', 'validation', 'calibration', 'test')},
            final_test_outcomes_hidden=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
