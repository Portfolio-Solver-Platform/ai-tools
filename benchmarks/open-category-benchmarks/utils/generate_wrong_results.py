"""
Generates wrong_results.csv — rows where a solver is provably wrong.

Uses gecode, chuffed, and cp-sat as trusted solvers whose results are
assumed correct. Other solvers that contradict them are marked wrong:
  - Claims UNSAT when a trusted solver found a solution.
  - Claims Optimal with a different objective than a trusted solver's Optimal.

These rows should receive 0 points in Borda scoring.

Usage:
    python generate_wrong_results.py
"""
import csv
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
COMBINED_CSV = ROOT / 'combined.csv'
TYPES_CSV = ROOT / 'problem_types.csv'
OUTPUT_CSV = ROOT / 'wrong_results.csv'

TRUSTED_SOLVERS = {'org.gecode.gecode', 'org.chuffed.chuffed', 'cp-sat'}
UNTRUSTED_SOLVERS = {'izplus', 'org.minizinc.mip.highs', 'org.choco.choco', 'org.minizinc.mip.coin-bc'}
SOLVED_STATUSES = {'Satisfied', 'Optimal', 'AllSolutions'}


def load_problem_types(path: Path) -> dict[tuple[str, str], str]:
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[(row['problem'], row['model'])] = row['type']
    return types


def main():
    problem_types = load_problem_types(TYPES_CSV)

    instances: dict[tuple, list[dict]] = defaultdict(list)
    with open(COMBINED_CSV) as f:
        for row in csv.DictReader(f):
            key = (row['year'], row['problem'], row['model'], row['name'])
            instances[key].append(row)

    wrong_rows: list[dict] = []

    for (year, problem, model, name), group in instances.items():
        kind = problem_types.get((problem, model))

        trusted = [r for r in group if r['solver'] in TRUSTED_SOLVERS]
        others = [r for r in group if r['solver'] not in TRUSTED_SOLVERS]

        # ── Check trusted solver consistency ─────────────────────────────
        trusted_optimals = [r for r in trusted if r['status'] == 'Optimal' and r['objective']]
        if len(trusted_optimals) >= 2:
            objs = set(float(r['objective']) for r in trusted_optimals)
            if len(objs) > 1:
                print(f'WARNING: trusted solvers disagree on {year} {problem}/{name}: '
                      + ', '.join(f"{r['solver']} c={r['cores']} obj={r['objective']}" for r in trusted_optimals))
                continue  # skip — can't determine truth

        # Get the trusted optimal objective (if any)
        trusted_opt_obj = None
        if trusted_optimals:
            trusted_opt_obj = float(trusted_optimals[0]['objective'])

        # Best feasible objective from trusted solvers (any status with an objective)
        trusted_best_feasible = None
        if kind in ('MIN', 'MAX'):
            for r in trusted:
                if r['objective']:
                    val = float(r['objective'])
                    if trusted_best_feasible is None:
                        trusted_best_feasible = val
                    elif kind == 'MIN':
                        trusted_best_feasible = min(trusted_best_feasible, val)
                    else:
                        trusted_best_feasible = max(trusted_best_feasible, val)

        # Count unique solver names that found solutions
        solved_rows = [r for r in group if r['status'] in SOLVED_STATUSES]
        solved_solver_names = {r['solver'] for r in solved_rows}

        # ── Mark wrong UNSAT claims ──────────────────────────────────────
        # Only if multiple unique solver names found solutions (one could be wrong)
        if len(solved_solver_names) >= 2:
            for r in group:
                if r['status'] in ('Unsatisfiable', 'Unsat'):
                    wrong_rows.append({
                        'solver': r['solver'],
                        'cores': r['cores'],
                        'year': year,
                        'problem': problem,
                        'model': model,
                        'name': name,
                        'reason': 'false_unsat',
                    })

        # ── Mark Optimal claims worse than trusted feasible ──────────────
        # If a trusted solver found obj=X, any Optimal claim worse than X
        # is provably wrong (X is feasible, so worse can't be optimal)
        if trusted_best_feasible is not None and kind in ('MIN', 'MAX'):
            for r in others:
                if r['status'] == 'Optimal' and r['objective']:
                    val = float(r['objective'])
                    is_worse = (kind == 'MIN' and val > trusted_best_feasible) or \
                               (kind == 'MAX' and val < trusted_best_feasible)
                    if is_worse:
                        wrong_rows.append({
                            'solver': r['solver'],
                            'cores': r['cores'],
                            'year': year,
                            'problem': problem,
                            'model': model,
                            'name': name,
                            'reason': 'wrong_optimal',
                        })

        # ── Mark wrong Optimal claims (disagreements) ────────────────────
        if kind in ('MIN', 'MAX'):
            all_optimals = [r for r in group if r['status'] == 'Optimal' and r['objective']]
            if len(all_optimals) >= 2:
                # Best objective per unique solver name
                obj_by_solver: dict[str, float] = {}
                for r in all_optimals:
                    val = float(r['objective'])
                    name_s = r['solver']
                    existing = obj_by_solver.get(name_s)
                    if existing is None:
                        obj_by_solver[name_s] = val
                    elif kind == 'MIN':
                        obj_by_solver[name_s] = min(existing, val)
                    else:
                        obj_by_solver[name_s] = max(existing, val)

                if len(set(obj_by_solver.values())) > 1:
                    # Use trusted solver objective if available
                    if trusted_opt_obj is not None:
                        correct_obj = trusted_opt_obj
                    elif trusted_best_feasible is not None:
                        # No trusted Optimal, but trusted feasible bound
                        # eliminates any Optimal claim worse than it
                        correct_obj = trusted_best_feasible
                    else:
                        # Majority vote by unique solver name
                        obj_counts = Counter(obj_by_solver.values())
                        top = obj_counts.most_common(2)
                        if len(top) >= 2 and top[0][1] == top[1][1]:
                            # Tie — print for manual review, skip
                            solvers_per_obj = defaultdict(list)
                            for s, o in obj_by_solver.items():
                                solvers_per_obj[o].append(s)
                            print(f'TIE: {year} {problem}/{name} (type={kind})')
                            for o, solvers in sorted(solvers_per_obj.items()):
                                print(f'  obj={o}: {", ".join(solvers)}')
                            continue
                        correct_obj = top[0][0]

                    for r in all_optimals:
                        if float(r['objective']) != correct_obj:
                            wrong_rows.append({
                                'solver': r['solver'],
                                'cores': r['cores'],
                                'year': year,
                                'problem': problem,
                                'model': model,
                                'name': name,
                                'reason': 'wrong_optimal',
                            })

    # ── Print untrusted solvers that are the only one with a result ────
    print('\nUNTRUSTED SOLE SOLVER (manual review needed):')
    for (year, problem, model, name), group in sorted(instances.items()):
        kind = problem_types.get((problem, model))
        if kind not in ('MIN', 'MAX'):
            continue
        optimals = [r for r in group if r['status'] == 'Optimal' and r['objective']]
        if not optimals:
            continue
        opt_solver_names = {r['solver'] for r in optimals}
        if len(opt_solver_names) == 1:
            solver_name = next(iter(opt_solver_names))
            if solver_name in UNTRUSTED_SOLVERS:
                obj = optimals[0]['objective']
                print(f'  {year} {problem}/{name} (type={kind})  solver={solver_name}  obj={obj}')

    # Deduplicate
    seen = set()
    unique_rows = []
    for r in wrong_rows:
        key = (r['solver'], r['cores'], r['year'], r['problem'], r['model'], r['name'])
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    unique_rows.sort(key=lambda r: (r['year'], r['problem'], r['name'], r['solver'], r['cores']))

    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['solver', 'cores', 'year', 'problem', 'model', 'name', 'reason'])
        w.writeheader()
        w.writerows(unique_rows)

    reason_counts = Counter(r['reason'] for r in unique_rows)
    print(f'\nWrote {len(unique_rows)} wrong results to {OUTPUT_CSV}')
    for reason, count in reason_counts.most_common():
        print(f'  {reason}: {count}')


if __name__ == '__main__':
    main()
