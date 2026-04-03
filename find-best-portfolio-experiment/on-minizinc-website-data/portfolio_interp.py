import json
import numpy as np
from multiprocessing import Pool
from systems import solver_systems, CPSAT_SYSTEMS

MAX_TIME_MS = 1200000


def open_competitors(data):
    r = data["results"]
    return [s for s, o in zip(r["solvers"], r["open_solvers"]) if o]


def interpolate(v1, v8, cores):
    if v1 is None or v8 is None:
        return v1 if cores < 4.5 else v8
    if cores == 1 or v1 == v8:
        return v1
    if cores == 8:
        return v8
    t = (cores - 1) / 7
    if v1 > 0 and v8 > 0:
        return v1 * (v8 / v1) ** t
    return v1 + (v8 - v1) * t


def interpolated_performance(data, base, cores, system_info):
    r = data["results"]
    nb_i = len(r["benchmarks"])
    benchmarks = system_info["benchmarks"]

    if 1 in benchmarks and 8 in benchmarks:
        s1 = r["solvers"].index(benchmarks[1])
        s8 = r["solvers"].index(benchmarks[8])
        perf = []
        for i in range(nb_i):
            time_ms = interpolate(parse_time(r["times"][s1][i]), parse_time(r["times"][s8][i]), cores)
            obj1 = parse_obj(r["objectives"][s1][i])
            obj8 = parse_obj(r["objectives"][s8][i])
            obj = interpolate(obj1, obj8, cores)
            result = r["results"][s8][i] if cores >= 4.5 else r["results"][s1][i]
            perf.append((result, time_ms, obj))
        return perf

    available_cores = next(iter(benchmarks))
    s = r["solvers"].index(benchmarks[available_cores])
    return [(r["results"][s][i], r["times"][s][i], r["objectives"][s][i]) for i in range(nb_i)]


def parse_obj(obj):
    try:
        return float(obj)
    except (ValueError, TypeError):
        return None


def parse_time(time):
    try:
        return float(time)
    except:
        return 1200000.0


def pairwise(result_s, time_s, obj_s, result_s2, time_s2, obj_s2, kind):
    solved_s = result_s in ("S", "SC")
    solved_s2 = result_s2 in ("S", "SC")
    if not solved_s:
        return 0.0
    if not solved_s2:
        return 1.0
    o_s, o_s2 = parse_obj(obj_s), parse_obj(obj_s2)
    if kind in ("MIN", "MAX") and o_s is not None and o_s2 is not None:
        better_s = (kind == "MIN" and o_s < o_s2) or (kind == "MAX" and o_s > o_s2)
        worse_s = (kind == "MIN" and o_s > o_s2) or (kind == "MAX" and o_s < o_s2)
        if better_s:
            return 1.0
        if worse_s:
            return 0.5
        if result_s == "SC" and result_s2 != "SC":
            return 1.0
        if result_s != "SC" and result_s2 == "SC":
            return 0.0
    t1 = int(time_s) // 1000
    t2 = int(time_s2) // 1000
    if t1 + t2 == 0:
        return 0.5
    return t2 / (t1 + t2)


def precompute_scores(data, perf_cache, competitors):
    r = data["results"]
    nb_i = len(r["benchmarks"])
    comp_idxs = [r["solvers"].index(n) for n in competitors]
    kind_for_instance = []
    for p, insts in enumerate(r["instances"]):
        for _ in insts:
            kind_for_instance.append(r["kind"][p])

    score_matrix = {}
    for key, perf in perf_cache.items():
        mat = np.zeros((len(comp_idxs), nb_i))
        for ci, s2 in enumerate(comp_idxs):
            for i in range(nb_i):
                result_s, time_s, obj_s = perf[i]
                mat[ci, i] = pairwise(
                    result_s, time_s, obj_s,
                    r["results"][s2][i], r["times"][s2][i], r["objectives"][s2][i],
                    kind_for_instance[i],
                )
        score_matrix[key] = mat
    return score_matrix


def enumerate_portfolios(systems, remaining=8, index=0):
    bases = list(systems.keys())
    if index == len(bases):
        yield []
        return
    base = bases[index]
    info = systems[base]
    yield from enumerate_portfolios(systems, remaining, index + 1)
    for c in sorted(info["supported_cores"]):
        if c <= remaining:
            for rest in enumerate_portfolios(systems, remaining - c, index + 1):
                yield [(base, c)] + rest


_score_matrix = None


def _init_worker(sm):
    global _score_matrix
    _score_matrix = sm


def _eval_portfolio(portfolio):
    combined = _score_matrix[portfolio[0]]
    for key in portfolio[1:]:
        combined = np.maximum(combined, _score_matrix[key])
    return combined.sum()


def best_portfolios(data, top_n=20):
    systems = solver_systems(data)
    competitors = open_competitors(data)

    perf_cache = {}
    for base, info in systems.items():
        for c in info["supported_cores"]:
            perf_cache[(base, c)] = interpolated_performance(data, base, c, info)

    print("Precomputing pairwise scores...")
    score_matrix = precompute_scores(data, perf_cache, competitors)

    portfolios = []
    for portfolio in enumerate_portfolios(systems):
        if portfolio:
            portfolios.append(tuple((b, c) for b, c in portfolio))
    print(f"Evaluating {len(portfolios)} portfolios...")

    with Pool(initializer=_init_worker, initargs=(score_matrix,)) as pool:
        scores = pool.map(_eval_portfolio, portfolios, chunksize=1000)

    candidates = sorted(zip(scores, portfolios), reverse=True)
    return candidates[:top_n]


if __name__ == "__main__":
    with open("results.json") as f:
        data = json.load(f)

    for rank, (s, portfolio) in enumerate(best_portfolios(data), 1):
        cores = sum(c for _, c in portfolio)
        names = ", ".join(f"{b}({c}c)" for b, c in portfolio)
        print(f"#{rank:2d}  score={s:7.2f}  cores={cores}  {names}")
