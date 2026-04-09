import bisect
from datetime import timedelta
from importlib import resources
from math import floor, log10, nan
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pint import UnitRegistry
from scipy import signal

from pmxpy import assets


def load_units():
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


def continuize_order_zero(
    times: NDArray, values: NDArray, left: float | None = None
) -> Callable[[float], float]:
    @np.vectorize(otypes=[float])
    def piecewise_rate(t):
        idx = bisect.bisect_right(times, t) - 1
        if idx < 0:
            return left if left is not None else values[idx]
        return values[idx]

    return piecewise_rate


def value_change_indices(data: NDArray):
    data = np.array(data)
    ddata = np.diff(data, prepend=data[0], append=data[-1])
    changes = np.flatnonzero(ddata)
    changes = np.insert(changes, 0, 0)
    return changes


def savgol(series, time_unit, window, order):
    conv = time_unit / timedelta(seconds=1)
    old_idx_conv = series.index.copy() * conv
    npoints = np.ceil(old_idx_conv.max() - old_idx_conv.min())
    new_idx = old_idx_conv.min() + np.arange(npoints)

    new_values = np.interp(new_idx, old_idx_conv, series.to_numpy())

    window = np.floor(window * conv)
    if not window % 2:
        # window is even, we need odd
        window += 1
    filtered = signal.savgol_filter(new_values, int(window), order)
    newname = f"{series.name}-filtered"
    newseries = pd.Series(filtered, index=new_idx / conv, name=newname)
    return newseries


def round_to_n_sig_figs(x, n=2):
    """Round to n significant figures"""
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


class ConcentrationFunction(Protocol):
    def __call__(self, t: float, /) -> float: ...
