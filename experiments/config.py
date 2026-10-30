"""Expand the shared integrator-table format used by the learning benchmarks."""

from itertools import product

ADJOINTS = {
    "direct": "direct",
    "full": "checkpoint_full",
    "recursive": "checkpoint_recursive",
    "reversible": "reversible",
}


def method_label(solver, adjoint, adaptive=False):
    names = {value: key for key, value in ADJOINTS.items()}
    names["recursive_checkpoint"] = "recursive"
    return f"{solver}_{'adaptive_' if adaptive else ''}{names.get(adjoint, adjoint)}"


def expand_integrators(data):
    """Expand seeds, integrators, stepping modes and adjoints in TOML order."""
    if "integrators" not in data:
        if "sweep" in data or "methods" in data:
            raise ValueError("Define runs with [integrators.<name>] tables.")
        return [data.get("base", data)]
    axes = data.get("sweep", {})
    if set(axes) & {"method", "solver", "adjoint", "adaptive", "n_steps"}:
        raise ValueError("Set integrator choices and n_steps in the integrator tables.")
    runs = []
    for values in product(*axes.values()):
        shared = {**data.get("base", {}), **dict(zip(axes, values, strict=True))}
        for solver, table in data["integrators"].items():
            settings = {
                key: value
                for key, value in table.items()
                if key not in ("adjoints", "stepping", "adaptive")
            }
            adjoints = table["adjoints"]
            modes = table.get("stepping", ["fixed"])
            if (
                not isinstance(adjoints, list)
                or not adjoints
                or any(a not in ADJOINTS for a in adjoints)
            ):
                raise ValueError(
                    f"{solver}: adjoints must select from {list(ADJOINTS)}."
                )
            if (
                not isinstance(modes, list)
                or not modes
                or any(m not in ("fixed", "adaptive") for m in modes)
            ):
                raise ValueError(f"{solver}: stepping must select fixed or adaptive.")
            for mode, adjoint in product(modes, adjoints):
                overrides = {}
                if mode == "adaptive":
                    for key, value in table.get("adaptive", {}).items():
                        if key not in ("n_steps", "atol", "dtmin", "max_steps"):
                            raise ValueError(
                                f"{solver}: unknown adaptive setting {key!r}."
                            )
                        overrides[key if key == "n_steps" else f"adaptive_{key}"] = (
                            value
                        )
                runs.append(
                    {
                        **shared,
                        **settings,
                        **overrides,
                        "solver": solver,
                        "adjoint": ADJOINTS[adjoint],
                        "adaptive": mode == "adaptive",
                    }
                )
    if not runs:
        raise ValueError("Select at least one integrator and seed.")
    return runs
