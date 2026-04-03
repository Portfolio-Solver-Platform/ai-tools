import json

CPSAT_SYSTEMS = {"or-tools_cp-sat", "or-tools_cp-sat_ls"}


def solver_systems(data):
    r = data["results"]
    systems = {}
    for name in r["solvers"]:
        for suffix, cores in [("-fd", 1), ("-free", 1), ("-open", 8), ("-par", 8)]:
            # if "jacop" in name or "choco" in name or "scip" in name or "optim" in name or "izplus" in name or "yuck" in name or "prolog" in name or "pumpkin" in name or "_ls" in name or "atlantis" in name: 
            #     continue
            if "jacop" in name or "izplus" in name: 
                continue
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                entry = systems.setdefault(base, {"benchmarks": {}, "supported_cores": {1}})
                entry["benchmarks"][cores] = name
                if cores == 8:
                    if base in CPSAT_SYSTEMS:
                        entry["supported_cores"] = {1, 8}
                    else:
                        entry["supported_cores"] = set(range(1, 9))
                break
    return systems


if __name__ == "__main__":
    with open("results.json") as f:
        data = json.load(f)

    print(solver_systems(data))