from __future__ import annotations

from importlib import resources
from math import nan
from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from pypmx import assets


def load_units():
    from pint import UnitRegistry

    ureg = UnitRegistry()
    # Q_ = ureg.Quantity
    # U_ = ureg.Unit
    asset_source = resources.files(assets)
    with resources.as_file(asset_source) as fp:
        ureg.load_definitions(fp / "units.txt")
    return ureg


def calculate_drug_dose(data, molar_mass=None):
    Q_ = load_units().Quantity
    rate = Q_(float(data["rate"]), data["rate_unit"])
    volume = Q_(float(data["volume"]), data["volume_unit"])
    mass_rate = dose = mol_rate = mol_dose = None
    if data["drug_concentration"] is not None:
        concentration = Q_(data["drug_concentration"])
        mass_rate = concentration * rate
        dose = concentration * volume
        if molar_mass:
            mol_rate = mass_rate / molar_mass
            mol_dose = dose / molar_mass
    return {
        "rate": rate,
        "mass_rate": mass_rate,
        "mol_rate": mol_rate,
        "volume": volume,
        "dose": dose,
        "mol_dose": mol_dose,
    }


class ConcentrationFunction(Protocol):
    """A concentration as a function of time, e.g. Cp(t) fed to a PD model."""

    def __call__(self, t: float, /) -> float: ...


def continuize_order_one(
    times: NDArray,
    values: NDArray,
    left: float | None = nan,
    right: float | None = None,
) -> ConcentrationFunction:
    """Linear interpolation of ``values`` sampled at ``times``.

    ``left`` is returned before ``times[0]`` (default nan), ``right`` after
    ``times[-1]`` (default: the last value).
    """

    def sub_func(x):
        return np.interp(x, times, values, left, right)

    return sub_func


def continuize_order_zero(infu_time, infu_rate, r0: float = 0):
    def run(t):
        if t < infu_time[0]:
            return r0
        idx = np.searchsorted(infu_time, t, side="right") - 1
        idx = np.clip(idx, 0, len(infu_rate) - 1)
        return infu_rate[idx]

    return run


def rate_at(t: Any, infu_time: Any, infu_rate: Any) -> NDArray[np.float64]:
    """Piecewise-constant infusion rate at time(s) ``t`` (vectorised).

    ``infu_rate[i]`` applies from ``infu_time[i]`` on, the last rate persists
    and the rate is 0 before ``infu_time[0]``.
    """
    t = np.asarray(t, dtype=float)
    infu_time = np.asarray(infu_time, dtype=float).ravel()
    infu_rate = np.asarray(infu_rate, dtype=float).ravel()
    idx = np.searchsorted(infu_time[1:], t, side="right")
    idx = np.clip(idx, 0, len(infu_rate) - 1)
    return np.where(t < infu_time[0], 0.0, infu_rate[idx])


def value_change_indices(data: NDArray):
    data = np.array(data)
    ddata = np.diff(data, prepend=data[0], append=data[-1])
    changes = np.flatnonzero(ddata)
    changes = np.insert(changes, 0, 0)
    return changes


def filter_valid(
    data: Mapping[str, Any] | NamedTuple, typeclass: type[NamedTuple]
) -> dict[str, Any]:
    """Keep only the entries of `data` that are fields of `typeclass`."""
    valid_fields = set(typeclass._fields)
    items = dict(data) if isinstance(data, Mapping) else data._asdict()
    return {k: v for k, v in items.items() if k in valid_fields}
