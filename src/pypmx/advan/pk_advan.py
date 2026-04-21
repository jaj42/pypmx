"""
ADVAN-style Analytical Solutions for Pharmacokinetic Models

Implementation of the equations from:
Abuhelwa, A. Y., Foster, D. J., & Upton, R. N. (2015).
ADVAN-style analytical solutions for common pharmacokinetic models.
Journal of Pharmacological and Toxicological Methods, 73, 42-48.

This module provides closed form solutions for 1, 2, and 3 compartment
pharmacokinetic models with IV bolus, IV infusion, and first-order oral absorption.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Parameters:
    ka: float = 0
    k10: float = 0
    k12: float = 0
    k13: float = 0
    k20: float = 0
    k21: float = 0
    k30: float = 0
    k31: float = 0


def calculate_lambda(p: Parameters) -> NDArray:
    K = np.array(
        [
            [
                -(p.k10 + p.k12 + p.k13),
                p.k21,
                p.k31,
            ],
            [
                p.k12,
                -(p.k20 + p.k21),
                0,
            ],
            [
                p.k13,
                0,
                -(p.k30 + p.k31),
            ],
        ]
    )
    lambdas = np.linalg.eigvals(-K)
    return np.sort(np.real(lambdas))[::-1]


class OneCompartmentIVBolus:
    def __init__(self, params: Parameters):
        self.params = params

    def advance(self, A1: float, t: float) -> float:
        X1 = A1 * np.exp(-self.params.k10 * t)
        return X1


class OneCompartmentIVInfusion:
    def __init__(self, params: Parameters):
        self.params = params

    def advance(self, A1: float, t: float, R: float = 0.0) -> float:
        k10 = self.params.k10
        X1 = (R / k10) * (1 - np.exp(-k10 * t)) + A1 * np.exp(-k10 * t)
        return X1


class TwoCompartmentIVBolus:
    def __init__(self, params: Parameters):
        self.params = params
        self.E1 = params.k10 + params.k12
        self.E2 = params.k20 + params.k21
        self.lambda1, self.lambda2, _ = calculate_lambda(params)

    def advance(self, A1: float, A2: float, t: float) -> tuple[float, float]:
        k21 = self.params.k21
        k12 = self.params.k12
        E1, E2 = self.E1, self.E2
        l1, l2 = self.lambda1, self.lambda2

        # Central compartment (Eq. 11, first equation)
        num1 = (A1 * E2 + A2 * k21) - A1 * l1
        num2 = (A1 * E2 + A2 * k21) - A1 * l2
        X1 = (num1 * np.exp(-l1 * t) - num2 * np.exp(-l2 * t)) / (l2 - l1)

        # Peripheral compartment (Eq. 11, second equation)
        num1 = (A2 * E1 + A1 * k12) - A2 * l1
        num2 = (A2 * E1 + A1 * k12) - A2 * l2
        X2 = (num1 * np.exp(-l1 * t) - num2 * np.exp(-l2 * t)) / (l2 - l1)

        return X1, X2


class TwoCompartmentIVInfusion:
    def __init__(self, params: Parameters):
        self.params = params
        self.E1 = params.k10 + params.k12
        self.E2 = params.k20 + params.k21
        self.lambda1, self.lambda2, _ = calculate_lambda(params)

    def advance(
        self, A1: float, A2: float, t: float, R: float = 0.0
    ) -> tuple[float, float]:
        k21 = self.params.k21
        k12 = self.params.k12
        E1, E2 = self.E1, self.E2
        l1, l2 = self.lambda1, self.lambda2

        # Central compartment (Eq. 27)
        num1 = (A1 * E2 + R + A2 * k21) - A1 * l1
        num2 = (A1 * E2 + R + A2 * k21) - A1 * l2
        X1 = (num1 * np.exp(-l1 * t) - num2 * np.exp(-l2 * t)) / (l2 - l1)
        X1 += (
            R
            * E2
            * (
                1 / (l1 * l2)
                + np.exp(-l1 * t) / (l1 * (l1 - l2))
                - np.exp(-l2 * t) / (l2 * (l1 - l2))
            )
        )

        # Peripheral compartment (Eq. 28)
        num1 = (A2 * E1 + A1 * k12) - A2 * l1
        num2 = (A2 * E1 + A1 * k12) - A2 * l2
        X2 = (num1 * np.exp(-l1 * t) - num2 * np.exp(-l2 * t)) / (l2 - l1)
        X2 += (
            R
            * k12
            * (
                1 / (l1 * l2)
                + np.exp(-l1 * t) / (l1 * (l1 - l2))
                - np.exp(-l2 * t) / (l2 * (l1 - l2))
            )
        )

        return X1, X2


class ThreeCompartmentIVBolus:
    def __init__(self, params: Parameters):
        self.params = params
        self.E1 = params.k12 + params.k13 + params.k10
        self.E2 = params.k21 + params.k20
        self.E3 = params.k31 + params.k30
        self.lambda1, self.lambda2, self.lambda3 = calculate_lambda(params)

    def advance(
        self, A1: float, A2: float, A3: float, t: float
    ) -> tuple[float, float, float]:
        k21 = self.params.k21
        k31 = self.params.k31
        k12 = self.params.k12
        k13 = self.params.k13
        E1, E2, E3 = self.E1, self.E2, self.E3
        l1, l2, l3 = self.lambda1, self.lambda2, self.lambda3

        # Helper terms (Eq. 21)
        B = A2 * k21 + A3 * k31
        C = A2 * k21 * E3 + A3 * k31 * E2
        I = A1 * k12 * E3 - A2 * k13 * k31 + A3 * k31 * k12
        J = A1 * k13 * E2 - A3 * k12 * k21 + A2 * k13 * k21

        # Central compartment (Eq. 18)
        X1_A1 = A1 * (
            (E2 - l1) * (E3 - l1) / ((l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
            + (E2 - l2) * (E3 - l2) / ((l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
            + (E2 - l3) * (E3 - l3) / ((l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
        )

        X1_BC = (
            (C - B * l1) / ((l1 - l2) * (l1 - l3)) * np.exp(-l1 * t)
            + (B * l2 - C) / ((l1 - l2) * (l2 - l3)) * np.exp(-l2 * t)
            + (B * l3 - C) / ((l1 - l3) * (l3 - l2)) * np.exp(-l3 * t)
        )

        X1 = X1_A1 + X1_BC

        # First peripheral compartment (Eq. 19)
        X2_A2 = A2 * (
            (E1 - l1) * (E3 - l1) / ((l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
            + (E1 - l2) * (E3 - l2) / ((l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
            + (E1 - l3) * (E3 - l3) / ((l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
        )

        X2_I = (
            (I - A1 * k12 * l1) / ((l1 - l2) * (l1 - l3)) * np.exp(-l1 * t)
            + (A1 * k12 * l2 - I) / ((l1 - l2) * (l2 - l3)) * np.exp(-l2 * t)
            + (A1 * k12 * l3 - I) / ((l1 - l3) * (l3 - l2)) * np.exp(-l3 * t)
        )

        X2 = X2_A2 + X2_I

        # Second peripheral compartment (Eq. 20)
        X3_A3 = A3 * (
            (E1 - l1) * (E2 - l1) / ((l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
            + (E1 - l2) * (E2 - l2) / ((l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
            + (E1 - l3) * (E2 - l3) / ((l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
        )

        X3_J = (
            (J - A1 * k13 * l1) / ((l1 - l2) * (l1 - l3)) * np.exp(-l1 * t)
            + (A1 * k13 * l2 - J) / ((l1 - l2) * (l2 - l3)) * np.exp(-l2 * t)
            + (A1 * k13 * l3 - J) / ((l1 - l3) * (l3 - l2)) * np.exp(-l3 * t)
        )

        X3 = X3_A3 + X3_J

        return X1, X2, X3


class ThreeCompartmentIVInfusion:
    def __init__(self, params: Parameters):
        self.params = params
        self.E1 = params.k12 + params.k13 + params.k10
        self.E2 = params.k21 + params.k20
        self.E3 = params.k31 + params.k30
        self.lambda1, self.lambda2, self.lambda3 = calculate_lambda(params)

    def advance(
        self, A1: float, A2: float, A3: float, t: float, R: float = 0.0
    ) -> tuple[float, float, float]:
        k21 = self.params.k21
        k31 = self.params.k31
        k12 = self.params.k12
        k13 = self.params.k13
        E1, E2, E3 = self.E1, self.E2, self.E3
        l1, l2, l3 = self.lambda1, self.lambda2, self.lambda3

        # Helper terms
        B = A2 * k21 + A3 * k31
        C = A2 * k21 * E3 + A3 * k31 * E2
        I = A1 * k12 * E3 - A2 * k13 * k31 + A3 * k31 * k12
        J = A1 * k13 * E2 - A3 * k12 * k21 + A2 * k13 * k21

        # Central compartment (Eq. 29)
        X1_R = R * (
            E2 * E3 / (l1 * l2 * l3)
            - (E2 - l1) * (E3 - l1) / (l1 * (l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
            - (E2 - l2) * (E3 - l2) / (l2 * (l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
            - (E2 - l3) * (E3 - l3) / (l3 * (l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
        )

        X1_A1 = A1 * (
            (E2 - l1) * (E3 - l1) / ((l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
            + (E2 - l2) * (E3 - l2) / ((l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
            + (E2 - l3) * (E3 - l3) / ((l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
        )

        X1_BC = (
            (C - B * l1) / ((l1 - l2) * (l1 - l3)) * np.exp(-l1 * t)
            + (B * l2 - C) / ((l1 - l2) * (l2 - l3)) * np.exp(-l2 * t)
            + (B * l3 - C) / ((l1 - l3) * (l3 - l2)) * np.exp(-l3 * t)
        )

        X1 = X1_R + X1_A1 + X1_BC

        # First peripheral compartment (Eq. 30)
        X2_R = (
            R
            * k12
            * (
                E3 / (l1 * l2 * l3)
                - (E3 - l1) / (l1 * (l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
                - (E3 - l2) / (l2 * (l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
                - (E3 - l3) / (l3 * (l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
            )
        )

        X2_A2 = A2 * (
            (E1 - l1) * (E3 - l1) / ((l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
            + (E1 - l2) * (E3 - l2) / ((l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
            + (E1 - l3) * (E3 - l3) / ((l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
        )

        X2_I = (
            (I - A1 * k12 * l1) / ((l1 - l2) * (l1 - l3)) * np.exp(-l1 * t)
            + (A1 * k12 * l2 - I) / ((l1 - l2) * (l2 - l3)) * np.exp(-l2 * t)
            + (A1 * k12 * l3 - I) / ((l1 - l3) * (l3 - l2)) * np.exp(-l3 * t)
        )

        X2 = X2_R + X2_A2 + X2_I

        # Second peripheral compartment (Eq. 31)
        X3_R = (
            R
            * k13
            * (
                E2 / (l1 * l2 * l3)
                - (E2 - l1) / (l1 * (l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
                - (E2 - l2) / (l2 * (l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
                - (E2 - l3) / (l3 * (l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
            )
        )

        X3_A3 = A3 * (
            (E1 - l1) * (E2 - l1) / ((l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
            + (E1 - l2) * (E2 - l2) / ((l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
            + (E1 - l3) * (E2 - l3) / ((l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
        )

        X3_J = (
            (J - A1 * k13 * l1) / ((l1 - l2) * (l1 - l3)) * np.exp(-l1 * t)
            + (A1 * k13 * l2 - J) / ((l1 - l2) * (l2 - l3)) * np.exp(-l2 * t)
            + (A1 * k13 * l3 - J) / ((l1 - l3) * (l3 - l2)) * np.exp(-l3 * t)
        )

        X3 = X3_R + X3_A3 + X3_J

        return X1, X2, X3


class OneCompartmentOral:
    def __init__(self, params: Parameters):
        self.params = params

    def advance(self, A1: float, A2: float, t: float) -> tuple[float, float]:
        ka = self.params.ka
        k20 = self.params.k10

        # Absorption compartment (Eq. 32)
        X1 = A1 * np.exp(-ka * t)

        # Central compartment (Eq. 33)
        X2 = (A1 * ka / (ka - k20)) * (
            np.exp(-k20 * t) - np.exp(-ka * t)
        ) + A2 * np.exp(-k20 * t)

        return X1, X2


class TwoCompartmentOral:
    def __init__(self, params: Parameters):
        self.params = params
        self.E2 = params.k10 + params.k12
        self.E3 = params.k20 + params.k21
        self.lambda1, self.lambda2, _ = calculate_lambda(params)

    def advance(
        self, A1: float, A2: float, A3: float, t: float
    ) -> tuple[float, float, float]:
        ka = self.params.ka
        k23 = self.params.k12
        k32 = self.params.k21
        E2, E3 = self.E2, self.E3
        l1, l2 = self.lambda1, self.lambda2

        # Absorption compartment (Eq. 32)
        X1 = A1 * np.exp(-ka * t)

        # Central compartment (Eq. 34)
        X2_A1 = (
            A1
            * ka
            * (
                (E3 - ka) / ((l1 - ka) * (l2 - ka)) * np.exp(-ka * t)
                + (E3 - l1) / ((l2 - l1) * (ka - l1)) * np.exp(-l1 * t)
                + (E3 - l2) / ((l1 - l2) * (ka - l2)) * np.exp(-l2 * t)
            )
        )

        num1 = (A2 * E3 + A3 * k32) - A2 * l1
        num2 = (A2 * E3 + A3 * k32) - A2 * l2
        X2_A23 = (num1 * np.exp(-l1 * t) - num2 * np.exp(-l2 * t)) / (l2 - l1)

        X2 = X2_A1 + X2_A23

        # Peripheral compartment (Eq. 35)
        X3_A1 = (
            A1
            * ka
            * k23
            * (
                np.exp(-ka * t) / ((l1 - ka) * (l2 - ka))
                + np.exp(-l1 * t) / ((l2 - l1) * (ka - l1))
                + np.exp(-l2 * t) / ((l1 - l2) * (ka - l2))
            )
        )

        num1 = (A3 * E2 + A2 * k23) - A3 * l1
        num2 = (A3 * E2 + A2 * k23) - A3 * l2
        X3_A23 = (num1 * np.exp(-l1 * t) - num2 * np.exp(-l2 * t)) / (l2 - l1)

        X3 = X3_A1 + X3_A23

        return X1, X2, X3


class ThreeCompartmentOral:
    def __init__(self, params: Parameters):
        self.params = params
        self.E2 = params.k12 + params.k13 + params.k10
        self.E3 = params.k21 + params.k20
        self.E4 = params.k31 + params.k30
        self.lambda1, self.lambda2, self.lambda3 = calculate_lambda(params)

    def advance(
        self, A1: float, A2: float, A3: float, A4: float, t: float
    ) -> tuple[float, float, float, float]:
        ka = self.params.ka
        k23 = self.params.k12
        k32 = self.params.k21
        k24 = self.params.k13
        k42 = self.params.k31
        E2, E3, E4 = self.E2, self.E3, self.E4
        l1, l2, l3 = self.lambda1, self.lambda2, self.lambda3

        # Helper terms (modified Eq. 40 for oral)
        B = A3 * k32 + A4 * k42
        C = A3 * k32 * E4 + A4 * k42 * E3
        I = A2 * k23 * E4 - A3 * k24 * k42 + A4 * k42 * k23
        J = A2 * k24 * E3 - A4 * k23 * k32 + A3 * k24 * k32

        # Absorption compartment (Eq. 32)
        X1 = A1 * np.exp(-ka * t)

        # Central compartment (Eq. 37)
        X2_A1 = (
            A1
            * ka
            * (
                (E3 - ka)
                * (E4 - ka)
                / ((l1 - ka) * (l2 - ka) * (l3 - ka))
                * np.exp(-ka * t)
                + (E3 - l1)
                * (E4 - l1)
                / ((ka - l1) * (l2 - l1) * (l3 - l1))
                * np.exp(-l1 * t)
                + (E3 - l2)
                * (E4 - l2)
                / ((ka - l2) * (l1 - l2) * (l3 - l2))
                * np.exp(-l2 * t)
                + (E3 - l3)
                * (E4 - l3)
                / ((ka - l3) * (l1 - l3) * (l2 - l3))
                * np.exp(-l3 * t)
            )
        )

        X2_A2 = A2 * (
            (E3 - l1) * (E4 - l1) / ((l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
            + (E3 - l2) * (E4 - l2) / ((l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
            + (E3 - l3) * (E4 - l3) / ((l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
        )

        X2_BC = (
            (C - B * l1) / ((l1 - l2) * (l1 - l3)) * np.exp(-l1 * t)
            + (B * l2 - C) / ((l1 - l2) * (l2 - l3)) * np.exp(-l2 * t)
            + (B * l3 - C) / ((l1 - l3) * (l3 - l2)) * np.exp(-l3 * t)
        )

        X2 = X2_A1 + X2_A2 + X2_BC

        # First peripheral compartment (Eq. 38)
        X3_A1 = (
            A1
            * ka
            * k23
            * (
                (E4 - ka) / ((l1 - ka) * (l2 - ka) * (l3 - ka)) * np.exp(-ka * t)
                + (E4 - l1) / ((ka - l1) * (l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
                + (E4 - l3) / ((ka - l3) * (l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
            )
        )

        X3_A3 = A3 * (
            (E2 - l1) * (E4 - l1) / ((l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
            + (E2 - l2) * (E4 - l2) / ((l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
            + (E2 - l3) * (E4 - l3) / ((l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
        )

        X3_I = (
            (I - A2 * k23 * l1) / ((l1 - l2) * (l1 - l3)) * np.exp(-l1 * t)
            + (A2 * k23 * l2 - I) / ((l1 - l2) * (l2 - l3)) * np.exp(-l2 * t)
            + (A2 * k23 * l3 - I) / ((l1 - l3) * (l3 - l2)) * np.exp(-l3 * t)
        )

        X3 = X3_A1 + X3_A3 + X3_I

        # Second peripheral compartment (Eq. 39)
        X4_A1 = (
            A1
            * ka
            * k24
            * (
                (E3 - ka) / ((l1 - ka) * (l2 - ka) * (l3 - ka)) * np.exp(-ka * t)
                + (E3 - l1) / ((ka - l1) * (l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
                + (E3 - l2) / ((ka - l2) * (l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
                + (E3 - l3) / ((ka - l3) * (l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
            )
        )

        X4_A4 = A4 * (
            (E2 - l1) * (E3 - l1) / ((l2 - l1) * (l3 - l1)) * np.exp(-l1 * t)
            + (E2 - l2) * (E3 - l2) / ((l1 - l2) * (l3 - l2)) * np.exp(-l2 * t)
            + (E2 - l3) * (E3 - l3) / ((l1 - l3) * (l2 - l3)) * np.exp(-l3 * t)
        )

        X4_J = (
            (J - A2 * k24 * l1) / ((l1 - l2) * (l1 - l3)) * np.exp(-l1 * t)
            + (A2 * k24 * l2 - J) / ((l1 - l2) * (l2 - l3)) * np.exp(-l2 * t)
            + (A2 * k24 * l3 - J) / ((l1 - l3) * (l3 - l2)) * np.exp(-l3 * t)
        )

        X4 = X4_A1 + X4_A4 + X4_J

        return X1, X2, X3, X4
