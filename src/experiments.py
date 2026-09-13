"""Phase 2 experiments: equilibrium statistics and the parameter sweep.

The headline results are in `blind_statistics`: how often equilibrium play
stays blind, and when it converts. `dominant_action_stats` reports what Phase 3
needs to know about whether a single-label benchmark is viable.

A note on the sweep
-------------------
The requested sweep is over stack depth and boot size. In this specification
both are DEGENERATE, and that is a finding rather than an omission:

  * RULES.md 3.2 fixes the stack at 50 and proves maximum exposure is 33, so the
    stack provably never binds. Every stack >= 34 therefore yields the identical
    game, identical tree and identical equilibrium. This is verified rather than
    asserted (`stack_sweep`).
  * The boot is the unit of account (RULES.md 3.1) and the opening stake equals
    it (7.1), so scaling the boot and the stack together rescales every payoff
    and leaves the strategy unchanged. Raising the boot alone eventually makes
    exposure exceed the stack, which requires an all-in rule that RULES.md
    deliberately does not specify (E1). The engine raises `StackWouldBind`
    there rather than inventing one (`boot_sweep`).

So the only parameters that actually change the equilibrium are the two caps of
RULES.md 9.2, which 9.1 states are modelling devices rather than rules of Teen
Patti. `cap_sweep` varies those, and it is where the real variation lives.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from game import (
    ACTION_CAP,
    Action,
    DEFAULT_CONFIG,
    GameConfig,
    Phase,
    StackWouldBind,
)
import cfr
import exploitability as EX


# ==========================================================================
# Node bookkeeping
# ==========================================================================

def turn_index(state) -> int:
    """Which of the acting player's own turns this is, 1-based.

    Seat 1 acts at k = 0, 2, 4, 6 and seat 2 at k = 1, 3, 5, 7, so k // 2 + 1
    gives the player's own turn number in both cases.
    """
    return state.k // 2 + 1


def node_reach(solver, reach, node: int) -> float:
    """Unconditional probability that `node` is reached under the profile.

    Sums over the joint deal distribution with exact card removal:
        pi(node) = sum_c0 w[c0] * r_0[c0] * sum_c1 COND[c0,c1] * r_1[c1]
    """
    tb = solver.tables
    r0, r1 = reach[node]
    if not isinstance(r1, np.ndarray):
        # A blind opponent's reach is uniform, and COND rows sum to 1.
        mass = float(r1) * np.ones(tb.n_classes)
    else:
        mass = tb.cond @ r1
    if not isinstance(r0, np.ndarray):
        return float(r0) * float(tb.chance_w @ mass)
    return float(tb.chance_w @ (r0 * mass))


def seen_infoset_reach(solver, reach, node: int, player: int) -> np.ndarray:
    """Per-class reach probability of a seen player's information sets."""
    tb = solver.tables
    r_self = reach[node][player]
    r_opp = reach[node][1 - player]
    if not isinstance(r_opp, np.ndarray):
        mass = float(r_opp) * np.ones(tb.n_classes)
    else:
        mass = tb.cond @ r_opp
    if not isinstance(r_self, np.ndarray):
        r_self = float(r_self) * np.ones(tb.n_classes)
    return tb.chance_w * r_self * mass


# ==========================================================================
# The headline result: blind play and conversion timing
# ==========================================================================

@dataclass
class BlindStats:
    config_label: str
    iterations: int
    exploitability: float
    game_value_p0: float

    #: P(seat 1 stays blind at their very first decision). Unconditional: the
    #: root is that decision, so this needs no conditioning.
    p1_stay_blind_first: float
    #: P(seat 2 stays blind at their first decision | that decision is reached).
    p2_stay_blind_first: float

    #: convert_marginal[p][j] = P(hand reaches p's turn j AND p looks there)
    convert_marginal: dict = field(default_factory=dict)
    #: convert_conditional[p][j] = P(p looks | p's turn j reached while blind)
    convert_conditional: dict = field(default_factory=dict)
    #: reach_blind[p][j] = P(p's turn j is reached with p still blind)
    reach_blind: dict = field(default_factory=dict)
    #: P(p never looks: either converts nowhere or the hand ends first)
    never_convert: dict = field(default_factory=dict)

    #: expected number of betting actions taken at the blind (half) price
    expected_blind_bets: dict = field(default_factory=dict)
    #: P(p exercises the blind-only right to demand a show)
    blind_show_prob: dict = field(default_factory=dict)
    #: P(p demands a show while seen), for contrast
    seen_show_prob: dict = field(default_factory=dict)
    #: P(p packs while still blind)
    blind_pack_prob: dict = field(default_factory=dict)


def blind_statistics(solver, report: EX.ExploitabilityReport | None = None) -> BlindStats:
    """Extract the equilibrium blind/seen statistics from the average strategy."""
    tree = solver.tree
    strategies = solver.strategy_profile(average=True)
    reach = solver._down_pass(strategies)
    rep = report or EX.compute_exploitability(solver, average=True)

    max_turns = (solver.config.action_cap + 1) // 2
    convert_marg = {p: {j: 0.0 for j in range(1, max_turns + 1)} for p in (0, 1)}
    reach_blind = {p: {j: 0.0 for j in range(1, max_turns + 1)} for p in (0, 1)}
    blind_bets = {p: 0.0 for p in (0, 1)}
    blind_show = {p: 0.0 for p in (0, 1)}
    seen_show = {p: 0.0 for p in (0, 1)}
    blind_pack = {p: 0.0 for p in (0, 1)}
    p2_first_stay_num = p2_first_stay_den = 0.0

    for raw in tree.decision_ids:
        n = int(raw)
        st = tree.states[n]
        p = st.to_act
        sigma = strategies[n]
        pi = node_reach(solver, reach, n)
        acts = tree.actions[n]
        j = turn_index(st)

        if st.phase is Phase.LOOK:
            i_see = acts.index(Action.SEE)
            i_stay = acts.index(Action.STAY_BLIND)
            reach_blind[p][j] += pi
            convert_marg[p][j] += pi * float(sigma[i_see])
            if p == 1 and j == 1:
                p2_first_stay_num += pi * float(sigma[i_stay])
                p2_first_stay_den += pi
            continue

        # Bet node. A blind actor here pays the half-price stake (RULES.md 6.4).
        if tree.actor_blind[n]:
            # chaal, raise and show all pay the blind (half) price; pack pays
            # nothing, so it is counted separately.
            for a in (Action.CHAAL, Action.RAISE, Action.SHOW):
                if a in acts:
                    prob = float(sigma[acts.index(a)])
                    blind_bets[p] += pi * prob
                    if a is Action.SHOW:
                        blind_show[p] += pi * prob
            if Action.PACK in acts:
                blind_pack[p] += pi * float(sigma[acts.index(Action.PACK)])
        else:
            if Action.SHOW in acts:
                # A seen actor's strategy is per class; weight by class reach.
                w = seen_infoset_reach(solver, reach, n, p)
                seen_show[p] += float(w @ sigma[:, acts.index(Action.SHOW)])

    root_sigma = strategies[0]
    root_acts = tree.actions[0]
    p1_stay = float(root_sigma[root_acts.index(Action.STAY_BLIND)])

    convert_cond = {
        p: {
            j: (convert_marg[p][j] / reach_blind[p][j] if reach_blind[p][j] > 1e-15
                else float("nan"))
            for j in convert_marg[p]
        }
        for p in (0, 1)
    }
    never = {p: 1.0 - sum(convert_marg[p].values()) for p in (0, 1)}

    return BlindStats(
        config_label=solver.config.label(),
        iterations=solver.iterations,
        exploitability=rep.exploitability,
        game_value_p0=rep.game_value_p0,
        p1_stay_blind_first=p1_stay,
        p2_stay_blind_first=(
            p2_first_stay_num / p2_first_stay_den
            if p2_first_stay_den > 1e-15 else float("nan")
        ),
        convert_marginal=convert_marg,
        convert_conditional=convert_cond,
        reach_blind=reach_blind,
        never_convert=never,
        expected_blind_bets=blind_bets,
        blind_show_prob=blind_show,
        seen_show_prob=seen_show,
        blind_pack_prob=blind_pack,
    )


def format_blind_stats(bs: BlindStats) -> str:
    lines = [f"=== Blind/seen equilibrium statistics: {bs.config_label} ===",
             f"  iterations={bs.iterations}  exploitability={bs.exploitability:.3e}"
             f"  v_0={bs.game_value_p0:+.6f}",
             "",
             f"  P(seat 1 stays blind at turn 1, unconditional) = "
             f"{bs.p1_stay_blind_first:.6f}",
             f"  P(seat 2 stays blind at turn 1 | reached)      = "
             f"{bs.p2_stay_blind_first:.6f}",
             ""]
    for p in (0, 1):
        lines.append(f"  --- seat {p + 1} conversion timing ---")
        lines.append("   turn | P(reach blind) | P(convert there) | P(look | reached)")
        for j in sorted(bs.convert_marginal[p]):
            lines.append(
                f"   {j:4d} | {bs.reach_blind[p][j]:14.6f} |"
                f" {bs.convert_marginal[p][j]:16.6f} |"
                f" {bs.convert_conditional[p][j]:17.6f}"
            )
        lines.append(f"   never looks (incl. hand ending first): "
                     f"{bs.never_convert[p]:.6f}")
        lines.append(f"   expected blind (half-price) bets/hand: "
                     f"{bs.expected_blind_bets[p]:.6f}")
        lines.append(f"   P(demands a show while blind)        : "
                     f"{bs.blind_show_prob[p]:.6f}")
        lines.append(f"   P(demands a show while seen)         : "
                     f"{bs.seen_show_prob[p]:.6f}")
        lines.append(f"   P(packs while still blind)           : "
                     f"{bs.blind_pack_prob[p]:.6f}")
        lines.append("")
    return "\n".join(lines)


# ==========================================================================
# Dominant-action statistics (for Phase 3's benchmark design)
# ==========================================================================

@dataclass
class DominanceStats:
    config_label: str
    threshold: float
    #: group -> (n_infosets_reachable, unweighted_dominant_frac, reach_weighted_frac)
    groups: dict = field(default_factory=dict)
    overall_unweighted: float = 0.0
    overall_weighted: float = 0.0


def dominant_action_stats(solver, threshold: float = 0.5,
                          reach_floor: float = 1e-12) -> DominanceStats:
    """Fraction of information sets with one action above `threshold`.

    PokerBench filters its benchmark to decisions with a dominant action. If
    the blind/seen decisions turn out to be heavily mixed, a single-label
    benchmark cannot represent them and a distribution-distance metric is
    needed instead, so the LOOK nodes are reported as their own group.
    """
    tree = solver.tree
    strategies = solver.strategy_profile(average=True)
    reach = solver._down_pass(strategies)

    groups = {
        "look (see / stay-blind)": [0, 0.0, 0.0, 0.0],
        "bet, actor blind": [0, 0.0, 0.0, 0.0],
        "bet, actor seen": [0, 0.0, 0.0, 0.0],
    }

    for raw in tree.decision_ids:
        n = int(raw)
        st = tree.states[n]
        sigma = strategies[n]
        if st.phase is Phase.LOOK:
            key = "look (see / stay-blind)"
        elif tree.actor_blind[n]:
            key = "bet, actor blind"
        else:
            key = "bet, actor seen"
        g = groups[key]

        if sigma.ndim == 1:
            pi = node_reach(solver, reach, n)
            if pi <= reach_floor:
                continue
            dominant = float(sigma.max()) > threshold
            g[0] += 1
            g[1] += 1.0 if dominant else 0.0
            g[2] += pi * (1.0 if dominant else 0.0)
            g[3] += pi
        else:
            w = seen_infoset_reach(solver, reach, n, st.to_act)
            live = w > reach_floor
            if not live.any():
                continue
            dominant = sigma.max(axis=1) > threshold
            g[0] += int(live.sum())
            g[1] += float((dominant & live).sum())
            g[2] += float(w[live & dominant].sum())
            g[3] += float(w[live].sum())

    out = DominanceStats(config_label=solver.config.label(), threshold=threshold)
    tot_n = tot_dom = tot_w = tot_wd = 0.0
    for key, (n_sets, n_dom, w_dom, w_tot) in groups.items():
        out.groups[key] = {
            "n_infosets": int(n_sets),
            "unweighted_dominant": (n_dom / n_sets) if n_sets else float("nan"),
            "reach_weighted_dominant": (w_dom / w_tot) if w_tot > 0 else float("nan"),
            "reach_mass": w_tot,
        }
        tot_n += n_sets
        tot_dom += n_dom
        tot_w += w_tot
        tot_wd += w_dom
    out.overall_unweighted = tot_dom / tot_n if tot_n else float("nan")
    out.overall_weighted = tot_wd / tot_w if tot_w > 0 else float("nan")
    return out


def format_dominance(ds: DominanceStats) -> str:
    lines = [
        f"=== Dominant-action share (p > {ds.threshold}): {ds.config_label} ===",
        "  group                     | infosets  | unweighted | reach-weighted",
        "  --------------------------|-----------|------------|---------------",
    ]
    for key, d in ds.groups.items():
        lines.append(
            f"  {key:<25} | {d['n_infosets']:>9,} |"
            f" {d['unweighted_dominant']:>10.4f} |"
            f" {d['reach_weighted_dominant']:>14.4f}"
        )
    lines.append(
        f"  {'OVERALL':<25} | {'':>9} | {ds.overall_unweighted:>10.4f} |"
        f" {ds.overall_weighted:>14.4f}"
    )
    return "\n".join(lines)


# ==========================================================================
# Sweeps
# ==========================================================================

def solve_config(config: GameConfig, iterations: int, tables, log_every: int = 0,
                 verbose: bool = False):
    """Solve one config, returning (solver, report). Raises if the stack binds."""
    solver = cfr.CFRPlusSolver(config, tables=tables)
    if log_every:
        run = EX.solve_with_logging(solver, iterations, log_every=log_every,
                                    verbose=verbose)
        return solver, run.final, run
    solver.iterate(iterations)
    return solver, EX.compute_exploitability(solver, average=True), None


def trees_are_structurally_identical(a, b) -> bool:
    """True if two trees differ in nothing but their config object.

    Stronger than comparing two solves: if the public trees and every public
    state field agree, the games are the same game and any solver must return
    the same answer.
    """
    if a.n_nodes != b.n_nodes or a.children != b.children or a.actions != b.actions:
        return False
    for sa, sb in zip(a.states, b.states):
        if (sa.stake, sa.contrib, sa.seen, sa.k, sa.r, sa.to_act, sa.phase,
                sa.history, sa.terminal) != (
                sb.stake, sb.contrib, sb.seen, sb.k, sb.r, sb.to_act, sb.phase,
                sb.history, sb.terminal):
            return False
    return True


def stack_sweep(tables, iterations: int = 40, stacks=(34, 40, 50, 75, 100)) -> list:
    """Verify that every non-binding stack gives the identical equilibrium.

    RULES.md 3.2 chose the stack so it can never bind (max exposure 33), which
    makes stack depth a non-parameter. Checked two ways: the public trees are
    compared structurally, and short solves are compared numerically.
    """
    rows = []
    reference = None
    ref_tree = None
    for stack in stacks:
        cfg = GameConfig(boot=1, starting_stack=stack)
        try:
            solver, rep, _ = solve_config(cfg, iterations, tables)
        except StackWouldBind as exc:
            rows.append({"stack": stack, "status": "binds", "detail": str(exc)})
            continue
        stats = blind_statistics(solver, rep)
        row = {
            "stack": stack,
            "status": "ok",
            "n_nodes": solver.tree.n_nodes,
            "game_value_p0": rep.game_value_p0,
            "exploitability": rep.exploitability,
            "p1_stay_blind_first": stats.p1_stay_blind_first,
        }
        if reference is None:
            reference, ref_tree = row, solver.tree
            row["tree_identical"] = True
            row["values_identical"] = True
        else:
            row["tree_identical"] = trees_are_structurally_identical(
                ref_tree, solver.tree
            )
            row["values_identical"] = bool(
                abs(row["game_value_p0"] - reference["game_value_p0"]) < 1e-12
                and abs(row["p1_stay_blind_first"]
                        - reference["p1_stay_blind_first"]) < 1e-12
            )
        rows.append(row)
    return rows


def boot_sweep(tables, iterations: int = 40, boots=(1, 2, 3, 5)) -> list:
    """Boot size at the RULES.md stack, and boot scaled with the stack.

    Two columns per boot: at the fixed 50-unit stack (where a boot above 1
    eventually exceeds the stack and leaves the specification), and with the
    stack scaled by the same factor (where the game is a pure rescaling).
    """
    rows = []
    base = None
    for boot in boots:
        row = {"boot": boot}
        # (a) fixed stack, as RULES.md specifies it
        try:
            cfg = GameConfig(boot=boot, starting_stack=50)
            solver, rep, _ = solve_config(cfg, iterations, tables)
            stats = blind_statistics(solver, rep)
            row["fixed_stack"] = {
                "status": "ok",
                "game_value_p0": rep.game_value_p0,
                "exploitability": rep.exploitability,
                "p1_stay_blind_first": stats.p1_stay_blind_first,
            }
        except StackWouldBind:
            row["fixed_stack"] = {
                "status": "outside RULES.md: exposure exceeds the stack and E1 "
                          "specifies no all-in rule"
            }
        # (b) stack scaled with the boot: a pure change of units
        cfg = GameConfig(boot=boot, starting_stack=50 * boot)
        solver, rep, _ = solve_config(cfg, iterations, tables)
        stats = blind_statistics(solver, rep)
        scaled = {
            "status": "ok",
            "game_value_p0": rep.game_value_p0,
            "game_value_per_boot": rep.game_value_p0 / boot,
            "p1_stay_blind_first": stats.p1_stay_blind_first,
        }
        if base is None:
            base = scaled
            scaled["strategy_matches_boot_1"] = True
        else:
            scaled["strategy_matches_boot_1"] = bool(
                abs(scaled["p1_stay_blind_first"]
                    - base["p1_stay_blind_first"]) < 1e-9
                and abs(scaled["game_value_per_boot"]
                        - base["game_value_per_boot"]) < 1e-9
            )
        row["scaled_stack"] = scaled
        rows.append(row)
    return rows


def cap_sweep(tables, grid=None, verbose: bool = True) -> list:
    """Sweep the two caps of RULES.md 9.2, the only parameters that bite.

    RULES.md 9.1 states the caps are modelling devices, not rules of Teen
    Patti, so varying them measures how the blind/seen equilibrium depends on
    how much betting room the game is given.
    """
    if grid is None:
        grid = [
            (2, 2, 3000), (4, 2, 1500), (6, 2, 600),
            (8, 0, 1500), (8, 1, 600),
        ]
    rows = []
    for action_cap, raise_cap, iters in grid:
        cfg = GameConfig(action_cap=action_cap, raise_cap=raise_cap)
        t0 = time.time()
        solver, rep, _ = solve_config(cfg, iters, tables)
        stats = blind_statistics(solver, rep)
        dom = dominant_action_stats(solver)
        rows.append({
            "action_cap": action_cap,
            "raise_cap": raise_cap,
            "iterations": iters,
            "seconds": time.time() - t0,
            "n_nodes": solver.tree.n_nodes,
            "exploitability": rep.exploitability,
            "game_value_p0": rep.game_value_p0,
            "p1_stay_blind_first": stats.p1_stay_blind_first,
            "p2_stay_blind_first": stats.p2_stay_blind_first,
            "convert_conditional_p0": stats.convert_conditional[0],
            "convert_conditional_p1": stats.convert_conditional[1],
            "never_convert": stats.never_convert,
            "expected_blind_bets": stats.expected_blind_bets,
            "blind_show_prob": stats.blind_show_prob,
            "look_dominant_weighted": dom.groups["look (see / stay-blind)"][
                "reach_weighted_dominant"],
            "overall_dominant_weighted": dom.overall_weighted,
        })
        if verbose:
            r = rows[-1]
            print(
                f"  k<={action_cap} r<={raise_cap}: nodes={r['n_nodes']:5d} "
                f"expl={r['exploitability']:.2e} v0={r['game_value_p0']:+.6f} "
                f"P1 stay-blind={r['p1_stay_blind_first']:.4f} "
                f"({r['seconds']:.0f}s)",
                flush=True,
            )
    return rows


# ==========================================================================
# Entry point
# ==========================================================================

def main() -> None:
    import sys

    reuse = "--fresh" not in sys.argv
    print("Building private tables...", flush=True)
    tables = cfr.build_private_tables()

    results: dict = {}

    # ---- the default config: reuse the long background solve if present ----
    solver = cfr.CFRPlusSolver(DEFAULT_CONFIG, tables=tables)
    loaded = False
    if reuse:
        try:
            solver.load("data/solution_default.pkl")
            loaded = True
            print(f"Loaded checkpoint at {solver.iterations} iterations", flush=True)
        except FileNotFoundError:
            print("No checkpoint; solving the default config fresh (slow)", flush=True)
    if not loaded:
        solver.iterate(200)

    rep = EX.compute_exploitability(solver, average=True)
    print(rep.line(), flush=True)
    stats = blind_statistics(solver, rep)
    dom = dominant_action_stats(solver)
    print()
    print(format_blind_stats(stats))
    print(format_dominance(dom))
    results["default"] = {
        "blind": _jsonable(stats),
        "dominance": _jsonable(dom),
        "exploitability": rep.exploitability,
        "game_value_p0": rep.game_value_p0,
        "iterations": solver.iterations,
    }

    print("\n=== Stack sweep (expect all identical) ===", flush=True)
    results["stack_sweep"] = stack_sweep(tables)
    for row in results["stack_sweep"]:
        print(f"  {row}", flush=True)

    print("\n=== Boot sweep ===", flush=True)
    results["boot_sweep"] = boot_sweep(tables)
    for row in results["boot_sweep"]:
        print(f"  {row}", flush=True)

    print("\n=== Cap sweep (the parameters that actually bite) ===", flush=True)
    results["cap_sweep"] = cap_sweep(tables)

    with open("data/experiments_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, default=str)
    print("\nWrote experiments_results.json", flush=True)


def _jsonable(obj):
    d = asdict(obj)
    return json.loads(json.dumps(d, default=str))


if __name__ == "__main__":
    main()
