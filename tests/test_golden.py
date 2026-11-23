"""Regression test against the pre-refactor golden snapshots (plan.md step 0).

Each ``tests/golden/*.npz`` holds the parameters, eigenvalues and simulated
outputs of one drug model case, generated from the code as it was before the
PK API unification (old API, ``solve_ivp(rtol=1e-10, atol=1e-12)`` on each
model's own ODE). The generator was removed once every model was migrated; the
snapshots are now fixed references. Each file's ``meta`` (JSON) holds the model
key, case name, covariates, constructor options and the infusion regimen.

``ADAPTERS`` maps a model key to a function that recomputes the same outputs
through the **current** API. Every model key in the snapshots needs one.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pypmx.drugs import (
    AllegaertParacetamol,
    Cefazolin,
    DyckDexmedetomidine,
    EleveldPropofol,
    EleveldRemifentanil,
    GeptsSufentanil,
    GrassinDelyleTranexamicAcid,
    HannivoortDexmedetomidine,
    KatariaPropofol,
    KuipersLidocaine,
    MarshPropofol,
    MintoRemifentanil,
    Noradrenaline,
    PaedfusorPropofol,
    SchniderPropofol,
    SuPropofolHemodynamics,
    SuPropofolRemifentanilHemodynamics,
    SuPropofolRemifentanilHemodynamicsVariant,
    ZomorodiMidazolam,
)
from pypmx.pk import LinearPKModel
from pypmx.utils import continuize_order_one

GOLDEN_DIR = Path(__file__).parent / "golden"
FILES = sorted(GOLDEN_DIR.glob("*.npz"))
N_SNAPSHOTS = 60

# Time grid of every snapshot (min)
TIME = np.linspace(0.0, 240.0, 481)

PARAM_RTOL = 1e-10
# The snapshots were integrated with solve_ivp (DOP853, rtol=1e-10), which is
# only accurate to ~1e-5 relative (up to 8e-6 vs an expm reference, e.g.
# KuipersLidocaine_B_high), while eig_advan matches expm to ~1e-14.
SIM_RTOL, SIM_ATOL = 2e-5, 1e-9
SIM_KEYS = ("cp", "ce", "effect", "hr", "sv", "map")


def linear_pk_adapter(
    cls: Callable[..., LinearPKModel],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Outputs of a migrated ``LinearPKModel`` for one golden case."""

    def compute(meta: dict[str, Any]) -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # cases are inside the studied ranges
            m = cls(meta["covars"], **meta["options"])
        p = m.pk
        names = ["V1", "V2", "CL", "Q2", "ke0", "k10", "k12", "k21"]
        if p.n_cmt == 3:
            names += ["V3", "Q3", "k13", "k31"]
        out: dict[str, Any] = {
            "params": {n: float(getattr(p, n)) for n in names},
            "eigenvalues": np.sort(np.linalg.eigvals(m.system_matrix).real),
        }
        pred = m.simulate(meta["infu_time"], meta["infu_rate"], TIME)
        out["cp"] = pred.cp
        if pred.ce is not None:
            out["ce"] = pred.ce
        return out

    return compute


# snapshot name -> new-API name (pk field or Noradrenaline.Parameters field)
NORADRENALINE_NAMES = {"Q": "Q2", "KE0": "ke0"}


def noradrenaline_adapter(meta: dict[str, Any]) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = Noradrenaline(meta["covars"] or None)
    values = m.pk._asdict() | m.parameters._asdict()
    names = (
        "V1", "V2", "CL", "Q", "EPROD", "KE0",
        "E0", "EC50", "EMAX", "GAMMA", "PKLAG", "PDLAG",
    )  # fmt: skip
    pred = m.simulate(meta["infu_time"], meta["infu_rate"], TIME)
    return {
        "params": {n: float(values[NORADRENALINE_NAMES.get(n, n)]) for n in names},
        "eigenvalues": np.sort(np.linalg.eigvals(m.system_matrix).real),
        "cp": pred.cp,
        "ce": pred.ce,
        "effect": pred.effect,
    }


def cefazolin(covars: dict[str, Any], CrCL: float) -> Cefazolin:
    """Old ``Cefazolin(CrCL)`` option -> ``CrCL`` covariate (0 meant unknown)."""
    return Cefazolin(covars | {"CrCL": CrCL or None})


# Fixed plasma concentration profiles driving the Su PD models
SU_TIMES = np.array([0.0, 5.0, 30.0, 90.0, 150.0, 240.0])
SU_CP_PROP = np.array([0.0, 4.0, 3.0, 3.0, 0.5, 0.2])  # ug/ml
SU_CP_REMI = np.array([0.0, 3.0, 2.0, 2.0, 0.2, 0.0])  # ng/ml


def su_adapter(meta: dict[str, Any]) -> dict[str, Any]:
    cp_prop = continuize_order_one(SU_TIMES, SU_CP_PROP)
    cp_remi = continuize_order_one(SU_TIMES, SU_CP_REMI)
    covars = meta["covars"] or None
    if meta["kind"] == "su_propofol":
        m = SuPropofolHemodynamics(cp_prop, covars, random_effects=False)
        t, hr, sv, map_ = m.calculate_at(TIME)
    else:
        cls = (
            SuPropofolRemifentanilHemodynamics
            if meta["model"] == "SuPropofolRemifentanilHemodynamics"
            else SuPropofolRemifentanilHemodynamicsVariant
        )
        t, hr, sv, map_ = cls(cp_prop, cp_remi, covars, random_effects=False)(TIME)
    assert np.array_equal(t, TIME)
    return {"hr": hr, "sv": sv, "map": map_}


ADAPTERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "Noradrenaline": noradrenaline_adapter,
    "SuPropofolHemodynamics": su_adapter,
    "SuPropofolRemifentanilHemodynamics": su_adapter,
    "SuPropofolRemifentanilHemodynamicsVariant": su_adapter,
} | {
    name: linear_pk_adapter(cls)
    for name, cls in {
        "EleveldPropofol": EleveldPropofol,
        "Schnider": SchniderPropofol,
        "Marsh": MarshPropofol,
        "Kataria": KatariaPropofol,
        "Paedfusor": PaedfusorPropofol,
        "EleveldRemifentanil": EleveldRemifentanil,
        "Minto": MintoRemifentanil,
        "GeptsSufentanil": GeptsSufentanil,
        "Cefazolin": cefazolin,
        "HannivoortDexmedetomidine": HannivoortDexmedetomidine,
        "DyckDexmedetomidine": DyckDexmedetomidine,
        "GrassinDelyleTranexamicAcid": GrassinDelyleTranexamicAcid,
        "AllegaertParacetamol": AllegaertParacetamol,
        "ZomorodiMidazolam": ZomorodiMidazolam,
        "KuipersLidocaine": KuipersLidocaine,
    }.items()
}


def load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(path) as d:
        meta = json.loads(str(d["meta"]))
        arrays = {k: d[k] for k in d.files if k != "meta"}
    return meta, arrays


def to_arrays(out: dict[str, Any]) -> dict[str, Any]:
    """Adapter outputs in the snapshot's array layout."""
    arrays: dict[str, Any] = {"t": TIME}
    if "params" in out:
        arrays["param_names"] = np.array(list(out["params"]))
        arrays["param_values"] = np.array(list(out["params"].values()), dtype=float)
    for key in ("eigenvalues", *SIM_KEYS):
        if key in out:
            arrays[key] = np.asarray(out[key], dtype=float)
    return arrays


def test_snapshots_present():
    assert len(FILES) == N_SNAPSHOTS
    for path in FILES:
        meta, _ = load(path)
        assert path.stem == f"{meta['model']}_{meta['case']}"
        assert meta["model"] in ADAPTERS


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_golden(path: Path):
    meta, gold = load(path)
    new = to_arrays(ADAPTERS[meta["model"]](dict(meta)))

    assert set(new) == set(gold)
    np.testing.assert_array_equal(new["t"], gold["t"])
    if "param_names" in gold:
        got = dict(zip(new["param_names"].tolist(), new["param_values"]))
        want = dict(zip(gold["param_names"].tolist(), gold["param_values"]))
        assert set(got) == set(want)
        for name, value in want.items():
            np.testing.assert_allclose(got[name], value, rtol=PARAM_RTOL, err_msg=name)
    if "eigenvalues" in gold:
        np.testing.assert_allclose(
            new["eigenvalues"], gold["eigenvalues"], rtol=PARAM_RTOL
        )
    for key in SIM_KEYS:
        if key in gold:
            np.testing.assert_allclose(
                new[key], gold[key], rtol=SIM_RTOL, atol=SIM_ATOL, err_msg=key
            )
