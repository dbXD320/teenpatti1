"""Run the headline solve (the RULES.md default config) and checkpoint it.

Kept separate from experiments.py so the long run can proceed in the
background and be reloaded rather than repeated.

Usage: python run_main_solve.py [iterations] [log_every]
"""

from __future__ import annotations

import json
import sys
import time

from game import DEFAULT_CONFIG
import cfr
import exploitability as EX

CHECKPOINT = "data/solution_default.pkl"
CURVE = "data/convergence_default.json"


def main() -> None:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    log_every = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    print(f"Solving RULES.md default config: {DEFAULT_CONFIG.label()}", flush=True)
    t0 = time.time()
    tables = cfr.build_private_tables()
    solver = cfr.CFRPlusSolver(DEFAULT_CONFIG, tables=tables)
    print(
        f"  tables+tree built in {time.time() - t0:.1f}s: "
        f"{solver.tree.n_nodes:,} nodes, {tables.n_classes:,} private classes",
        flush=True,
    )

    nbytes = sum(
        a.nbytes for a in solver.regret if a is not None
    ) + sum(a.nbytes for a in solver.strat_sum if a is not None)
    print(f"  regret+strategy tables: {nbytes / 1e6:.0f} MB", flush=True)
    print(f"  running {iterations} iterations, logging every {log_every}", flush=True)

    run = EX.solve_with_logging(
        solver, iterations, log_every=log_every, also_current=True, verbose=True
    )

    solver.save(CHECKPOINT)
    with open(CURVE, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": DEFAULT_CONFIG.label(),
                "seconds": run.seconds,
                "average": [
                    {
                        "iterations": r.iterations,
                        "exploitability": r.exploitability,
                        "milliboots_per_hand": r.milliboots_per_hand,
                        "pct_of_initial_pot": r.pct_of_initial_pot,
                        "game_value_p0": r.game_value_p0,
                        "br_p0": r.br_value_p0,
                        "br_p1": r.br_value_p1,
                    }
                    for r in run.records
                ],
                "current_iterate": [
                    {"iterations": r.iterations, "exploitability": r.exploitability}
                    for r in run.current_iterate
                ],
            },
            fh,
            indent=1,
        )

    final = run.final
    print(flush=True)
    print(f"DONE in {run.seconds / 60:.1f} min", flush=True)
    print(f"  final exploitability : {final.exploitability:.4e} boots/hand", flush=True)
    print(f"                        {final.milliboots_per_hand:.4f} mb/hand", flush=True)
    print(f"                        {final.pct_of_initial_pot:.5f}% of the pot", flush=True)
    print(f"  game value to seat 1 : {final.game_value_p0:+.6f}", flush=True)
    print(f"  monotone decreasing  : {run.is_monotone_decreasing()}", flush=True)
    print(f"  plateau ratio (last 3 logs): {run.plateau_ratio():.4f}", flush=True)
    for thr in (1e-2, 1e-3, 1e-4, 1e-5):
        print(f"  iters to < {thr:.0e}: {run.iterations_to_reach(thr)}", flush=True)
    print(f"  checkpoint -> {CHECKPOINT}", flush=True)


if __name__ == "__main__":
    main()
