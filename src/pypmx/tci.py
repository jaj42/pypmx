from typing import List, Tuple
import dataclasses
from collections import namedtuple

import numpy as np

UDFs = namedtuple("UDFs", ["p_udf", "e_udf"])
Coefs = namedtuple("Coefs", ["p_coef", "e_coef"])


@dataclasses.dataclass
class Parameters:
    vc: float  # L
    k10: float  # min⁻¹
    k12: float = 0  # min⁻¹
    k21: float = 0  # min⁻¹
    k13: float = 0  # min⁻¹
    k31: float = 0  # min⁻¹
    ke0: float = 0  # min⁻¹
    time_unit: str = "minutes"

    def per_seconds(self):
        if self.time_unit == "seconds":
            # Already in seconds
            result = self
        elif self.time_unit == "minutes":
            params = dataclasses.asdict(self).copy()
            for p in ["k10", "k12", "k13", "k21", "k31", "ke0"]:
                params[p] /= 60
            params["time_unit"] = "seconds"
            result = Parameters(**params)
        else:
            raise ValueError(f"Unsupported time unit: {self.time_unit}")
        return result

    def per_minutes(self):
        if self.time_unit == "minutes":
            # Already in minutes
            result = self
        elif self.time_unit == "seconds":
            params = dataclasses.asdict(self).copy()
            for p in ["k10", "k12", "k13", "k21", "k31", "ke0"]:
                params[p] *= 60
            params["time_unit"] = "seconds"
            result = Parameters(**params)
        else:
            raise ValueError(f"Unsupported time unit: {self.time_unit}")
        return result


def calculate_lambdas(parameters):
    p = parameters
    A = [
        [-(p.k10 + p.k12 + p.k13), p.k21, p.k31],
        [p.k12, -p.k21, 0],
        [p.k13, 0, -p.k31],
    ]
    lambdas = np.zeros(4)
    lambdas[0:3] = -np.sort(np.linalg.eigvals(A))
    lambdas[3] = p.ke0
    return lambdas


def calculate_coefficients(parameters: Parameters, lambdas: List[float]):
    k21 = parameters.k21
    k31 = parameters.k31
    ke0 = parameters.ke0
    vc = parameters.vc
    ls = lambdas

    p_coef = np.zeros(4)
    e_coef = np.zeros(4)

    # Calculate coefficients
    if k31 > 0:
        # Three compartment model
        p_coef[0] = (
            (k21 - ls[0])
            * (k31 - ls[0])
            / ((ls[0] - ls[1]) * (ls[0] - ls[2]) * vc * ls[0])
        )
        p_coef[1] = (
            (k21 - ls[1])
            * (k31 - ls[1])
            / ((ls[1] - ls[0]) * (ls[1] - ls[2]) * vc * ls[1])
        )
        p_coef[2] = (
            (k21 - ls[2])
            * (k31 - ls[2])
            / ((ls[2] - ls[1]) * (ls[2] - ls[0]) * vc * ls[2])
        )
        e_coef[0] = p_coef[0] / (ke0 - ls[0]) * ke0
        e_coef[1] = p_coef[1] / (ke0 - ls[1]) * ke0
        e_coef[2] = p_coef[2] / (ke0 - ls[2]) * ke0
        e_coef[3] = (
            (ke0 - k21)
            * (ke0 - k31)
            / ((ls[0] - ke0) * (ls[1] - ke0) * (ls[2] - ke0) * vc)
        )
    elif k21 > 0:
        # Two compartment model
        p_coef[0] = (k21 - ls[0]) / ((ls[1] - ls[0]) * vc * ls[0])
        p_coef[1] = (k21 - ls[1]) / ((ls[0] - ls[1]) * vc * ls[1])
        p_coef[2] = 0
        e_coef[0] = p_coef[0] / (ke0 - ls[0]) * ke0
        e_coef[1] = p_coef[1] / (ke0 - ls[1]) * ke0
        e_coef[2] = 0
        e_coef[3] = (k21 - ke0) / ((ls[0] - ke0) * (ls[1] - ke0) * vc)
    else:
        # One compartment model
        p_coef[0] = 1 / (ls[0] * vc)
        p_coef[1] = 0
        p_coef[2] = 0
        e_coef[0] = p_coef[0] / (ke0 - ls[0]) * ke0
        e_coef[1] = 0
        e_coef[2] = 0
        e_coef[3] = 1 / ((ls[0] - ke0) * vc)

    return Coefs(p_coef, e_coef)


def calculate_udfs(
    parameters: Parameters, lambdas: List[float], delta_seconds: int
) -> Tuple[UDFs, Coefs, int]:
    """Calculate Unit Disposition Functions
    Needs parameters in s-1
    """
    p_udf = np.zeros(200)
    e_udf = np.zeros(45 * 60)  # 45 minutes

    decay = np.exp(-lambdas)

    coefs = calculate_coefficients(parameters, lambdas)
    p_coef = coefs.p_coef
    e_coef = coefs.e_coef

    # Calculate plasma UDF (step response)
    A = np.zeros(4)
    for i in range(len(p_udf)):
        p_udf[i] = np.sum(A)
        A = A * decay + p_coef * (1 - decay)

    # Calculate effect site UDF until peak
    if np.sum(e_coef) > 0:
        A *= 0  # reinit state
        # Initial bolus of 1/s
        for i in range(delta_seconds):
            e_udf[i] = np.sum(A)
            A = A * decay + e_coef * (1 - decay)

        e_udf[delta_seconds] = np.sum(A)  # ← Store state at t=10

        # Wait until plateau to estimate peak_time
        prior = e_udf[i - 1]
        for i in range(delta_seconds + 1, len(e_udf)):
            A = A * decay.T
            e_udf[i] = np.sum(A)
            if prior >= e_udf[i]:
                break
            prior = e_udf[i]
        peak_time = i - 1
        if peak_time >= len(e_udf):
            print("Error: e_udf calculation exceeded maximum iterations")

    return UDFs(p_udf, e_udf), Coefs(p_coef, e_coef), peak_time


def model(
    Ap: List[float],
    Ae: List[float],
    lambdas: List[float],
    udfs: UDFs,
    coefs: Coefs,
    desired: float,
    peak_time: int,
    delta_seconds: int,
    target_effect: bool = True,
) -> float:
    """
    Calculate required infusion rate to achieve desired concentration
    Returns rate in units/second
    """
    temp1, temp2, temp3, _ = Ap
    temp1e, temp2e, temp3e, temp4e = Ae
    p_udf = udfs.p_udf
    e_udf = udfs.e_udf

    # Check if we should use plasma targeting or are very close to target
    target_delta_threshold = 0.05  # 5% close to target -> switch to plasma targetting
    close_to_target = abs(desired - np.sum(Ae)) < desired * target_delta_threshold
    if not target_effect or close_to_target:
        # Plasma targeting mode
        if temp1 == 0:
            # First dose - simple calculation
            return desired / p_udf[delta_seconds]
        else:
            # Ongoing infusion - predict and adjust

            result = virtual_model(Ap, delta_seconds, peak_time, lambdas, False)
            if desired > result:
                return (desired - result) / p_udf[delta_seconds]
            else:
                return 0.0

    else:
        # Effect site targeting mode
        if temp1 == 0:
            # First dose for effect site
            return desired / e_udf[peak_time]

        # Check if pump should be off (already at/above target)

        if virtual_model(Ae, delta_seconds, peak_time, lambdas, True) > desired:
            return 0.0

        # Set convergence criteria
        min_dif = desired * 0.0000001  # High precision for simulation

        # Initial settings
        temp_peak = peak_time
        if temp_peak <= delta_seconds:
            temp_peak = delta_seconds + 1

        # Initial rate calculation
        rate = (
            desired - virtual_model(Ae, temp_peak, peak_time, lambdas, True)
        ) / e_udf[temp_peak]

        temp_peak = find_peak(
            temp_peak, rate, Ae, peak_time, lambdas, e_udf, delta_seconds
        )
        current = (
            virtual_model(Ae, temp_peak, peak_time, lambdas, True)
            + e_udf[temp_peak] * rate
        )

        # Iterate until convergence
        iteration_count = 0
        max_iterations = 5000  # Safety limit

        while abs(current - desired) > min_dif and iteration_count < max_iterations:
            rate = (
                desired - virtual_model(Ae, temp_peak, peak_time, lambdas, True)
            ) / e_udf[temp_peak]

            temp_peak = find_peak(
                temp_peak, rate, Ae, peak_time, lambdas, e_udf, delta_seconds
            )
            current = (
                virtual_model(Ae, temp_peak, peak_time, lambdas, True)
                + e_udf[temp_peak] * rate
            )
            iteration_count += 1

        if iteration_count >= max_iterations:
            print(f"Warning: Model did not converge after {max_iterations} iterations")

        # Handle very small rates near peak time
        # XXX Maybe this is not needed here but should be handled by the pump rate resolution
        if rate < 0.00001 and temp_peak < delta_seconds * 2:
            rate = 0

    return max(0.0, rate)


def virtual_model(
    A: List[float],
    t: int,
    peak_time: int,
    lambdas: List[float],
    target_effect: bool,
) -> float:
    """
    Predict drug concentration at time t given current compartment amounts
    """
    # Probably this function should be replaced by update() with rate = 0 +/- lru_cache

    vm1, vm2, vm3, vm4 = A

    if target_effect and t > peak_time + 1:
        print("Error: Time error in virtual_model")
        raise ValueError("Error: Time error in virtual_model")
        return 0.0

    decay = np.exp(-lambdas * t)
    temp = np.sum(A * decay)

    return temp


def find_peak(
    current_time: int,
    rate: float,
    Ae: List[float],
    peak_time: int,
    lambdas: List[float],
    e_udf: List[float],
    delta_seconds: int,
) -> int:
    """Find the time when effect site concentration will peak"""

    # Calculate concentrations at three time points
    current = (
        virtual_model(Ae, current_time, peak_time, lambdas, True)
        + e_udf[current_time] * rate
    )
    earlier = (
        virtual_model(Ae, current_time - 1, peak_time, lambdas, True)
        + e_udf[current_time - 1] * rate
    )
    later = (
        virtual_model(Ae, current_time + 1, peak_time, lambdas, True)
        + e_udf[current_time + 1] * rate
    )

    # Search for peak
    while current < earlier or current < later:
        if current < earlier:
            if current_time == delta_seconds:
                return current_time
            current_time -= 1
            later = current
            current = earlier
            earlier = (
                virtual_model(Ae, current_time, peak_time, lambdas, True)
                + e_udf[current_time] * rate
            )
        else:
            current_time += 1
            earlier = current
            current = later
            later = (
                virtual_model(Ae, current_time + 1, peak_time, lambdas, True)
                + e_udf[current_time + 1] * rate
            )

    return current_time


def update(
    A: List[float],
    lambdas: List[float],
    coefs: List[float],
    rate: float,  # rate in units/second
    dt: int,  # time in seconds
):
    ls = np.exp(-lambdas * dt)
    return A * ls + coefs * rate * (1 - ls)
