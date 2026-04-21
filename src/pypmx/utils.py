import bisect
from datetime import timedelta
from importlib import resources
from math import floor, log10, nan
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import signal

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


def continuize_order_one(
    times: NDArray,
    values: NDArray,
    left: float | None = nan,
    right: float | None = None,
) -> Callable[[float], float]:
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


def value_change_indices(data: NDArray):
    data = np.array(data)
    ddata = np.diff(data, prepend=data[0], append=data[-1])
    changes = np.flatnonzero(ddata)
    changes = np.insert(changes, 0, 0)
    return changes


class ConcentrationFunction(Protocol):
    def __call__(self, t: float, /) -> float: ...
