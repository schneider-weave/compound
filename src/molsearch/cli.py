from __future__ import annotations

import argparse

from .pipeline import list_candidates, run_iteration, show_best


def _cmd_run(args: argparse.Namespace) -> None:
    run_iteration(args.config, dry_run=False)


def _cmd_dry_run(args: argparse.Namespace) -> None:
    run_iteration(args.config, dry_run=True)


def _cmd_candidates(args: argparse.Namespace) -> None:
    candidates = list_candidates(args.config)
    print(f"Candidates: {len(candidates)}")
    for molecule_id in candidates["molecule_id"].head(args.top).tolist():
        print(molecule_id)


def _cmd_best(args: argparse.Namespace) -> None:
    best_df = show_best(args.config, args.top)
    if best_df.empty:
        print("No scored molecules found.")
        return
    print(best_df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Molecule Active Search CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run iterative active search until convergence")
    run_p.add_argument("--config", required=True)
    run_p.set_defaults(func=_cmd_run)

    dry_p = sub.add_parser("dry-run", help="Select molecules without scoring")
    dry_p.add_argument("--config", required=True)
    dry_p.set_defaults(func=_cmd_dry_run)

    cand_p = sub.add_parser("candidates", help="List filtered candidate IDs")
    cand_p.add_argument("--config", required=True)
    cand_p.add_argument("--top", type=int, default=100)
    cand_p.set_defaults(func=_cmd_candidates)

    best_p = sub.add_parser("best", help="Show top scored molecules")
    best_p.add_argument("--config", required=True)
    best_p.add_argument("--top", type=int, default=20)
    best_p.set_defaults(func=_cmd_best)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
