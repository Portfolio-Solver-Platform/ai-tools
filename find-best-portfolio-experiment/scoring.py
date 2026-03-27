import json


def score(data, solver_name, competitors):
    r = data["results"]
    s = r["solvers"].index(solver_name)
    rivals = [r["solvers"].index(name) for name in competitors]
    return sum(r["scores"][s][s2][i] for s2 in rivals for i in range(len(r["benchmarks"])))


if __name__ == "__main__":
    with open("results.json") as f:
        data = json.load(f)

    open_solvers = [s for s, o in zip(data["results"]["solvers"], data["results"]["open_solvers"]) if o]
    for name in sorted(open_solvers, key=lambda n: -score(data, n, open_solvers)):
        print(f"{name:40s} {score(data, name, open_solvers):.2f}")
