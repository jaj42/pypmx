"""Golden snapshots of the pre-refactor drug models (plan.md step 0).

Run from the repository root to (re)generate the snapshots:

    uv run python tests/golden/generate.py

Each case is saved as ``tests/golden/<model>_<case>.npz`` with:

- ``meta``: JSON with the model key, covariates, constructor options, the kind of
  simulation and the infusion regimen;
- ``t``: time grid (min);
- ``param_names`` / ``param_values``: PK (and PD) parameters;
- ``eigenvalues``: sorted eigenvalues of the PK system (plus -ke0 when there is an
  effect site), in 1/min;
- ``cp``, and ``ce`` when the model has an effect site (ke0 > 0);
- ``effect`` (Noradrenaline MAP) or ``hr``/``sv``/``map`` (Su PD models).

PK simulations integrate each model's **own** ODE with
``solve_ivp(rtol=1e-10, atol=1e-12)``, piecewise between infusion breakpoints.
The Su PD models are run through their public method with their own solver
settings, because the refactor does not touch them.

Only regenerate on purpose, and say why in the commit message.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from pypmx.drugs import (
    AllegaertParacetamol,
    Cefazolin,
    DyckDexmedetomidine,
    EleveldPropofol,
    GeptsSufentanil,
    GrassinDelyleTranexamicAcid,
    HannivoortDexmedetomidine,
    KuipersLidocaine,
    Noradrenaline,
    SuPropofolHemodynamics,
    SuPropofolRemifentanilHemodynamics,
    SuPropofolRemifentanilHemodynamicsVariant,
    ZomorodiMidazolam,
)
from pypmx.drugs import propofol as propofol_models
from pypmx.drugs.propofol import Kataria, Marsh, Paedfusor, Schnider
from pypmx.drugs.remifentanil import Eleveld as EleveldRemifentanil
from pypmx.drugs.remifentanil import Minto
from pypmx.utils import continuize_order_one

FloatArray = NDArray[np.float64]

GOLDEN_DIR = Path(__file__).parent
RTOL = 1e-10
ATOL = 1e-12

# Standard regimen: 2-min loading infusion, maintenance until 60 min, washout.
T_END = 240.0
TIME = np.linspace(0.0, T_END, 481)
INFU_TIME = (0.0, 2.0, 60.0)

# (loading rate, maintenance rate) in the model's amount unit per minute
RATES: dict[str, tuple[float, float]] = {
    "EleveldPropofol": (50.0, 8.0),
    "Schnider": (50.0, 8.0),
    "Marsh": (50.0, 8.0),
    "Kataria": (15.0, 2.5),
    "Paedfusor": (15.0, 2.5),
    "EleveldRemifentanil": (25.0, 15.0),
    "Minto": (25.0, 15.0),
    "HannivoortDexmedetomidine": (10.0, 1.0),
    "DyckDexmedetomidine": (10.0, 1.0),
    "GrassinDelyleTranexamicAcid": (500.0, 1.0),
    "AllegaertParacetamol": (500.0, 5.0),
    "ZomorodiMidazolam": (1000.0, 50.0),
    "KuipersLidocaine": (50.0, 2.0),
    "GeptsSufentanil": (10.0, 0.5),
    "Cefazolin": (1000.0, 10.0),
    "Noradrenaline": (40.0, 20.0),
}

PARAM_NAMES = ("V1", "V2", "V3", "CL", "Q2", "Q3", "ke0")
MICRO_NAMES = ("k10", "k12", "k21", "k13", "k31")

# Fixed plasma concentration profiles for the Su PD models
SU_TIMES = np.array([0.0, 5.0, 30.0, 90.0, 150.0, 240.0])
SU_CP_PROP = np.array([0.0, 4.0, 3.0, 3.0, 0.5, 0.2])  # ug/ml
SU_CP_REMI = np.array([0.0, 3.0, 2.0, 2.0, 0.2, 0.0])  # ng/ml


def case(
    model: str,
    name: str,
    /,
    kind: str = "pk",
    covars: dict[str, Any] | None = None,
    **options: Any,
) -> dict[str, Any]:
    return {
        "model": model,
        "case": name,
        "kind": kind,
        "covars": covars or {},
        "options": options,
    }


CASES: list[dict[str, Any]] = [
    # Propofol
    case("EleveldPropofol", "default"),
    case(
        "EleveldPropofol",
        "elderly_f_noopiates",
        covars={"age": 75, "weight": 60, "height": 160, "sex": "f", "opiates": False},
    ),
    case("EleveldPropofol", "child", covars={"age": 6, "weight": 20, "height": 115}),
    case("Schnider", "default"),
    case(
        "Schnider",
        "elderly_f",
        covars={"age": 75, "weight": 60, "height": 160, "sex": "f"},
    ),
    case("Marsh", "default"),
    case("Marsh", "w50", covars={"weight": 50}),
    case("Kataria", "a5_w20", covars={"age": 5, "weight": 20}),
    case("Kataria", "a11_w40", covars={"age": 11, "weight": 40}),
    case("Paedfusor", "a2_w12", covars={"age": 2, "weight": 12}),
    case("Paedfusor", "a10_w35", covars={"age": 10, "weight": 35}),
    # Remifentanil
    case("EleveldRemifentanil", "default"),
    case(
        "EleveldRemifentanil",
        "elderly_f",
        covars={"age": 75, "weight": 60, "height": 160, "sex": "f"},
    ),
    case(
        "EleveldRemifentanil", "child", covars={"age": 6, "weight": 20, "height": 115}
    ),
    case("Minto", "default"),
    case(
        "Minto",
        "elderly_f",
        covars={"age": 75, "weight": 60, "height": 160, "sex": "f"},
    ),
    # Dexmedetomidine
    case("HannivoortDexmedetomidine", "moaas"),
    case("HannivoortDexmedetomidine", "bis", effect="bis"),
    case("HannivoortDexmedetomidine", "moaas_w100", covars={"weight": 100}),
    case("DyckDexmedetomidine", "default"),
    case("DyckDexmedetomidine", "h155", covars={"height": 155}),
    case("DyckDexmedetomidine", "h195", covars={"height": 195}),
    # Tranexamic acid
    case("GrassinDelyleTranexamicAcid", "default"),
    case("GrassinDelyleTranexamicAcid", "w50_a45", covars={"weight": 50, "age": 45}),
    case("GrassinDelyleTranexamicAcid", "w100", covars={"weight": 100}),
    # Paracetamol
    *(
        case("AllegaertParacetamol", s, setting=s)
        for s in ("none", "abdominal", "orthopaedic", "icu")
    ),
    case("AllegaertParacetamol", "pregnancy", covars={"sex": "f"}, setting="pregnancy"),
    case(
        "AllegaertParacetamol",
        "elderly_obese",
        covars={"age": 80, "weight": 115, "height": 175},
    ),
    case(
        "AllegaertParacetamol",
        "light_f",
        covars={"age": 30, "weight": 52, "height": 160, "sex": "f"},
    ),
    # Midazolam
    case("ZomorodiMidazolam", "simple"),
    case("ZomorodiMidazolam", "bsa", bsa_adjusted=True),
    case(
        "ZomorodiMidazolam",
        "bsa_small",
        covars={"age": 50, "weight": 58, "height": 158},
        bsa_adjusted=True,
    ),
    case(
        "ZomorodiMidazolam",
        "bsa_large",
        covars={"age": 78, "weight": 112, "height": 190},
        bsa_adjusted=True,
    ),
    # Lidocaine
    *(case("KuipersLidocaine", m, model=m) for m in ("A", "B", "C", "D")),
    *(
        case(
            "KuipersLidocaine",
            f"{m}_low",
            covars={"age": 20, "weight": 52, "height": 158, "cardiac_output": 3.8},
            model=m,
        )
        for m in ("B", "C", "D")
    ),
    *(
        case(
            "KuipersLidocaine",
            f"{m}_high",
            covars={"age": 75, "weight": 100, "height": 180, "cardiac_output": 9.5},
            model=m,
        )
        for m in ("B", "C", "D")
    ),
    # Sufentanil, cefazolin (seconds / concentration-state models)
    case("GeptsSufentanil", "default"),
    case("Cefazolin", "no_crcl", CrCL=0),
    case("Cefazolin", "crcl80", CrCL=80),
    case("Cefazolin", "crcl40", CrCL=40),
    case("Cefazolin", "crcl130", CrCL=130),
    # Noradrenaline
    case("Noradrenaline", "baseline", kind="noradrenaline", infusion=False),
    case("Noradrenaline", "infusion", kind="noradrenaline", infusion=True),
    case(
        "Noradrenaline",
        "infusion_w100",
        kind="noradrenaline",
        covars={"weight": 100},
        infusion=True,
    ),
    # Su PD models
    case("SuPropofolHemodynamics", "default", kind="su_propofol"),
    case("SuPropofolHemodynamics", "a70", kind="su_propofol", covars={"age": 70}),
    case(
        "SuPropofolRemifentanilHemodynamics", "default", kind="su_propofol_remifentanil"
    ),
    case(
        "SuPropofolRemifentanilHemodynamics",
        "a70",
        kind="su_propofol_remifentanil",
        covars={"age": 70},
    ),
    case(
        "SuPropofolRemifentanilHemodynamicsVariant",
        "default",
        kind="su_propofol_remifentanil",
    ),
    case(
        "SuPropofolRemifentanilHemodynamicsVariant",
        "a70",
        kind="su_propofol_remifentanil",
        covars={"age": 70},
    ),
]


# ---------------------------------------------------------------------------
# Helpers


def regimen(model: str) -> tuple[tuple[float, ...], tuple[float, float, float]]:
    load, maint = RATES[model]
    return INFU_TIME, (load, maint, 0.0)


def step_function(
    infu_time: Sequence[float], infu_rate: Sequence[float]
) -> Callable[[float], float]:
    """Right-continuous piecewise-constant rate, 0 before the first breakpoint."""

    def rate(t: float) -> float:
        idx = int(np.searchsorted(infu_time, t, side="right")) - 1
        return float(infu_rate[idx]) if idx >= 0 else 0.0

    return rate


def solve_piecewise(
    rhs: Callable[[float, FloatArray, float], Any],
    y0: Sequence[float] | FloatArray,
    t_eval: FloatArray,
    breakpoints: Sequence[float],
    rate: Callable[[float], float],
) -> FloatArray:
    """Integrate ``rhs(t, y, r)`` with ``r = rate(t)`` constant on each segment.

    Returns the states at ``t_eval``, shape (n_states, n_times).
    """
    edges = sorted({float(t_eval[0]), float(t_eval[-1])} | set(breakpoints))
    edges = [e for e in edges if t_eval[0] <= e <= t_eval[-1]]
    y = np.asarray(y0, dtype=float)
    out = np.full((len(y), len(t_eval)), np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        r = rate(a)
        mask = (t_eval >= a) & (t_eval <= b)
        seg_t = np.union1d(t_eval[mask], [b])
        sol = solve_ivp(
            lambda t, y: rhs(t, y, r),
            (a, b),
            y,
            t_eval=seg_t,
            rtol=RTOL,
            atol=ATOL,
            method="DOP853",
        )
        assert sol.success, sol.message
        out[:, mask] = sol.y[:, np.isin(seg_t, t_eval[mask])]
        y = sol.y[:, -1]
    assert np.all(np.isfinite(out))
    return out


def micro_matrix(p: dict[str, float]) -> FloatArray:
    """Amount-space PK matrix (+ Ce row if ke0 > 0) from micro constants."""
    if "k13" in p:
        k123 = p["k10"] + p["k12"] + p["k13"]
        S = np.array(
            [
                [-k123, p["k21"], p["k31"]],
                [p["k12"], -p["k21"], 0.0],
                [p["k13"], 0.0, -p["k31"]],
            ]
        )
    else:
        S = np.array(
            [
                [-(p["k10"] + p["k12"]), p["k21"]],
                [p["k12"], -p["k21"]],
            ]
        )
    n = S.shape[0]
    if p["ke0"] > 0:
        S = np.pad(S, ((0, 1), (0, 1)))
        S[n, 0] = p["ke0"] / p["V1"]
        S[n, n] = -p["ke0"]
    return S


def sorted_eigenvalues(S: FloatArray) -> FloatArray:
    ev = np.linalg.eigvals(S)
    assert np.allclose(ev.imag, 0)
    return np.sort(ev.real)


def pk_output(
    params: dict[str, float],
    S_old: FloatArray | None,
    rhs: Callable[[float, FloatArray, float], Any],
    n_states: int,
    meta: dict[str, Any],
    observe: Callable[[FloatArray], tuple[FloatArray, FloatArray]],
    time_scale: float = 1.0,
) -> dict[str, Any]:
    """Common PK output: parameters, eigenvalues, Cp and (if ke0 > 0) Ce.

    ``time_scale`` converts minutes to the model's time unit (60 for seconds).
    """
    infu_time, infu_rate = regimen(meta["model"])
    meta["infu_time"], meta["infu_rate"] = list(infu_time), list(infu_rate)
    rate = step_function(infu_time, [r / time_scale for r in infu_rate])
    tt = TIME * time_scale
    y = solve_piecewise(
        rhs, np.zeros(n_states), tt, [t * time_scale for t in infu_time], rate
    )
    cp, ce = observe(y)

    if S_old is not None:
        # Old matrices are in concentration space with an always-present Ce row;
        # the eigenvalues are the same as in amount space. Drop Ce when ke0 = 0.
        S = S_old if params["ke0"] > 0 else S_old[:-1, :-1]
    else:
        S = micro_matrix(params)
    out: dict[str, Any] = {
        "params": params,
        "eigenvalues": sorted_eigenvalues(S),
        "cp": cp,
    }
    if params["ke0"] > 0:
        out["ce"] = ce
    return out


def params_from(obj: Any, names: Sequence[str]) -> dict[str, float]:
    out = {}
    for n in names:
        v = getattr(obj, n, np.nan)
        if not np.isnan(v):
            out[n] = float(v)
    return out


def macro_from_micro(V1: float, k: dict[str, float], ke0: float) -> dict[str, float]:
    """Macro parameters for models defined by V1 and micro constants.

    V2 and V3 are derived as Q/k so that they match the dynamics (the published
    V2/V3 of e.g. Marsh are rounded and slightly inconsistent with k12/k21).
    """
    p = {"V1": V1, "CL": k["k10"] * V1, "Q2": k["k12"] * V1, "ke0": ke0}
    p["V2"] = p["Q2"] / k["k21"]
    if "k13" in k:
        p["Q3"] = k["k13"] * V1
        p["V3"] = p["Q3"] / k["k31"]
    return p | k


# ---------------------------------------------------------------------------
# Model kinds


NAMEDTUPLE_MODELS: dict[str, Callable[..., Any]] = {
    "EleveldPropofol": EleveldPropofol,
    "Schnider": Schnider,
    "EleveldRemifentanil": EleveldRemifentanil,
    "Minto": Minto,
    "HannivoortDexmedetomidine": HannivoortDexmedetomidine,
    "DyckDexmedetomidine": DyckDexmedetomidine,
    "GrassinDelyleTranexamicAcid": GrassinDelyleTranexamicAcid,
    "AllegaertParacetamol": AllegaertParacetamol,
    "ZomorodiMidazolam": ZomorodiMidazolam,
    "KuipersLidocaine": KuipersLidocaine,
}

# NamedTuple models defined by V1 and micro constants whose published V2/V3 are
# inconsistent with k21/k31: report V2/V3 derived as Q/k (see macro_from_micro).
MICRO_DEFINED = {"Schnider"}

ATTRIBUTE_MODELS: dict[str, Callable[..., Any]] = {
    "Marsh": Marsh,
    "Kataria": Kataria,
    "Paedfusor": Paedfusor,
}


def compute_namedtuple(meta: dict[str, Any]) -> dict[str, Any]:
    m = NAMEDTUPLE_MODELS[meta["model"]](meta["covars"], **meta["options"])
    p = m.parameters
    params = params_from(p, PARAM_NAMES + MICRO_NAMES)
    if meta["model"] in MICRO_DEFINED:
        k = {n: params[n] for n in MICRO_NAMES}
        params = macro_from_micro(p.V1, k, p.ke0)
    n = m.system_matrix.shape[0]

    def rhs(t: float, y: FloatArray, r: float) -> Any:
        return m.ode(t, y, lambda _: r)

    return pk_output(
        params,
        m.system_matrix,
        rhs,
        n,
        meta,
        lambda y: (y[0] / p.V1, y[-1]),
    )


def compute_attribute(meta: dict[str, Any]) -> dict[str, Any]:
    covars = propofol_models.propofol.Covariates(**meta["covars"])
    m = ATTRIBUTE_MODELS[meta["model"]](covars)
    k = {n: float(getattr(m, n)) for n in MICRO_NAMES}
    params = macro_from_micro(m.v1, k, float(m.keo))
    S = micro_matrix(params)

    def rhs(t: float, y: FloatArray, r: float) -> FloatArray:
        dy = S @ y
        dy[0] += r
        return dy

    return pk_output(
        params, None, rhs, S.shape[0], meta, lambda y: (y[0] / m.v1, y[-1])
    )


def compute_sufentanil(meta: dict[str, Any]) -> dict[str, Any]:
    m = GeptsSufentanil()
    sp = m._params
    k = {n: sp[n] * 60 for n in MICRO_NAMES}
    params = macro_from_micro(sp["V1"], k, sp["ke0"] * 60)

    def rhs(t: float, y: FloatArray, r: float) -> Any:
        return m.pk(t, y, r, sp)

    return pk_output(
        params, None, rhs, 4, meta, lambda y: (y[0], y[3]), time_scale=60.0
    )


def compute_cefazolin(meta: dict[str, Any]) -> dict[str, Any]:
    m = Cefazolin(meta["options"]["CrCL"], y0=[0.0, 0.0])
    cp_ = m._params
    params = {
        "V1": cp_["Vc"],
        "V2": cp_["Vp"],
        "CL": cp_["CL"] * 60,
        "Q2": cp_["Q"] * 60,
        "ke0": 0.0,
        "k10": cp_["k10"] * 60,
        "k12": cp_["k12"] * 60,
        "k21": cp_["k21"] * 60,
    }

    def rhs(t: float, y: FloatArray, r: float) -> Any:
        return m.pk(t, y, r, cp_)

    return pk_output(
        params, None, rhs, 2, meta, lambda y: (y[0], y[0]), time_scale=60.0
    )


NORADRENALINE_PARAMS = (
    "V1", "V2", "CL", "Q", "EPROD", "KE0",
    "E0", "EC50", "EMAX", "GAMMA", "PKLAG", "PDLAG",
)  # fmt: skip


def compute_noradrenaline(meta: dict[str, Any]) -> dict[str, Any]:
    # Pre-refactor API (dict params, get_state), removed in plan step 4.
    m: Any = Noradrenaline(meta["covars"] or None)
    p = m.params
    lag = p["PKLAG"] + p["PDLAG"]
    load, maint = RATES["Noradrenaline"]
    infu_rate = (load, maint, 0.0) if meta["options"]["infusion"] else (0.0, 0.0, 0.0)
    meta["infu_time"], meta["infu_rate"] = list(INFU_TIME), list(infu_rate)
    rate = step_function(INFU_TIME, infu_rate)

    def rhs(t: float, y: FloatArray, r: float) -> Any:
        return m.ode(t, y, rate)

    y = solve_piecewise(
        rhs, m.get_state(), TIME, [t + lag for t in INFU_TIME], lambda _: 0.0
    )
    cp = y[0] / p["V1"]
    ce = y[2]
    params = {n: float(p[n]) for n in NORADRENALINE_PARAMS}
    k = {"k10": p["K10"], "k12": p["K12"], "k21": p["K21"]}
    S = micro_matrix({"V1": p["V1"], "ke0": p["KE0"]} | k)
    return {
        "params": params,
        "eigenvalues": sorted_eigenvalues(S),
        "cp": cp,
        "ce": ce,
        "effect": m.pd(ce),
    }


def compute_su(meta: dict[str, Any]) -> dict[str, Any]:
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


def compute(meta: dict[str, Any]) -> dict[str, Any]:
    """Outputs of the pre-refactor code for one case (meta is updated in place)."""
    model, kind = meta["model"], meta["kind"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if kind == "noradrenaline":
            return compute_noradrenaline(meta)
        if kind.startswith("su_"):
            return compute_su(meta)
        if model in NAMEDTUPLE_MODELS:
            return compute_namedtuple(meta)
        if model in ATTRIBUTE_MODELS:
            return compute_attribute(meta)
        if model == "GeptsSufentanil":
            return compute_sufentanil(meta)
        if model == "Cefazolin":
            return compute_cefazolin(meta)
    raise KeyError(model)


def case_path(meta: dict[str, Any]) -> Path:
    return GOLDEN_DIR / f"{meta['model']}_{meta['case']}.npz"


def to_arrays(out: dict[str, Any]) -> dict[str, Any]:
    arrays: dict[str, Any] = {"t": TIME}
    if "params" in out:
        arrays["param_names"] = np.array(list(out["params"]))
        arrays["param_values"] = np.array(list(out["params"].values()), dtype=float)
    for key in ("eigenvalues", "cp", "ce", "effect", "hr", "sv", "map"):
        if key in out:
            arrays[key] = np.asarray(out[key], dtype=float)
    return arrays


def main() -> None:
    for old in GOLDEN_DIR.glob("*.npz"):
        old.unlink()
    for meta in CASES:
        meta = json.loads(json.dumps(meta))
        out = compute(meta)
        np.savez(case_path(meta), meta=np.array(json.dumps(meta)), **to_arrays(out))
        print(case_path(meta).name)


if __name__ == "__main__":
    main()
