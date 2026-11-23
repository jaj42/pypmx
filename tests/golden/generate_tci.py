"""Golden TCI runs from the pre-refactor ``tci.py`` (plan.md step 6).

Run from the repository root with the step-5 code (commit e4eab23) checked out:

    uv run python tests/golden/generate_tci.py

It writes ``tests/golden/tci/runs.npz`` with, for each model in ``MODELS``, the
per-second lambdas and coefficients, the UDFs, the effect-site peak time, and
the rates/Cp/Ce of a plasma- and an effect-targeted run (``TARGETS``, one
10-s step each). Like ``generate.py``, it uses the old API (the ``Parameters``
dataclass) and can no longer run once step 6 is in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pypmx import tci
from pypmx.drugs import EleveldPropofol, MarshPropofol, MintoRemifentanil

old_tci: Any = tci  # pre-step-6 API (Parameters, calculate_lambdas)

OUT = Path(__file__).parent / "tci" / "runs.npz"
DELTA_SECONDS = 10
# 10 min at 3, 10 min at 4, 10 min at 2 (µg/ml or ng/ml), one value per step.
TARGETS = np.repeat([3.0, 4.0, 2.0], 60)
MODELS = {
    "EleveldPropofol": EleveldPropofol,
    "MarshPropofol": MarshPropofol,
    "MintoRemifentanil": MintoRemifentanil,
}


def run(
    lambdas: Any,
    udfs: Any,
    coefs: Any,
    peak_time: int,
    target_effect: bool,
    model: Any = None,
    update: Any = None,
) -> dict[str, np.ndarray]:
    """Closed-loop TCI: each step, ask ``model`` for a rate, then advance."""
    model = model or old_tci.model
    update = update or old_tci.update
    n = len(lambdas)
    Ap, Ae = np.zeros(n), np.zeros(n)
    rates, cp, ce = [], [], []
    for target in TARGETS:
        rate = model(
            Ap,
            Ae,
            lambdas,
            udfs,
            coefs,
            target,
            peak_time,
            DELTA_SECONDS,
            target_effect=target_effect,
        )
        Ap = update(Ap, lambdas, coefs.p_coef, rate, DELTA_SECONDS)
        Ae = update(Ae, lambdas, coefs.e_coef, rate, DELTA_SECONDS)
        rates.append(rate)
        cp.append(Ap.sum())
        ce.append(Ae.sum())
    return {"rate": np.array(rates), "cp": np.array(cp), "ce": np.array(ce)}


def main() -> None:
    out: dict[str, Any] = {"targets": TARGETS}
    for key, cls in MODELS.items():
        p = cls().pk
        params: Any = old_tci.Parameters(
            vc=p.V1, k10=p.k10, k12=p.k12, k21=p.k21, k13=p.k13, k31=p.k31, ke0=p.ke0
        ).per_seconds()
        lambdas = old_tci.calculate_lambdas(params)
        udfs, coefs, peak_time = old_tci.calculate_udfs(params, lambdas, DELTA_SECONDS)
        out[f"{key}/lambdas"] = lambdas
        out[f"{key}/p_coef"] = coefs.p_coef
        out[f"{key}/e_coef"] = coefs.e_coef
        out[f"{key}/p_udf"] = udfs.p_udf
        out[f"{key}/e_udf"] = udfs.e_udf
        out[f"{key}/peak_time"] = peak_time
        for mode, target_effect in (("plasma", False), ("effect", True)):
            res = run(lambdas, udfs, coefs, peak_time, target_effect)
            for name, values in res.items():
                out[f"{key}/{mode}/{name}"] = values
    np.savez(OUT, **out)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
