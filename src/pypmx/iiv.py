"""Inter-individual variability (IIV): ``omega`` and sampling of individuals.

A model with IIV declares a class-level :class:`Omega`: the covariance of the
random effects ``eta ~ N(0, Ω)`` and how each one acts on its parameter,
exponentially (``P = P_typ·exp(η)``, the default) or additively
(``P = P_typ + η``).

Models inherit :class:`RandomEffects`, which gives them

- ``with_eta(eta)``: a copy with the etas applied (deterministic, e.g. for an
  EBE), and
- ``sample(rng)``: a copy with etas drawn from ``N(0, Ω)``.

The etas act on the model's **current** values, so they should be applied to
a typical (freshly built) model. Each model provides ``_get_values`` and
``_set_values`` to read and replace the parameters named in ``omega``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from math import exp
from typing import ClassVar, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
R = TypeVar("R", bound="RandomEffects")


def _symmetric(
    names: tuple[str, ...],
    diag: Iterable[float],
    off: Mapping[tuple[str, str], float],
) -> FloatArray:
    m = np.diag(np.asarray(list(diag), dtype=float))
    index = {n: i for i, n in enumerate(names)}
    for (a, b), value in off.items():
        i, j = index[a], index[b]
        if i == j:
            raise ValueError(f"off-diagonal entry ({a}, {b}) is on the diagonal")
        m[i, j] = m[j, i] = value
    return m


class Omega:
    """Covariance of the random effects, one eta per named parameter."""

    def __init__(
        self,
        names: Iterable[str],
        cov: ArrayLike,
        additive: Iterable[str] = (),
    ):
        self.names = tuple(names)
        self.cov = np.array(cov, dtype=float)
        self.additive = frozenset(additive)
        n = len(self.names)
        if self.cov.shape != (n, n):
            raise ValueError(f"cov must be {n}x{n}, got {self.cov.shape}")
        if not np.allclose(self.cov, self.cov.T):
            raise ValueError("cov must be symmetric")
        if unknown := self.additive - set(self.names):
            raise ValueError(f"additive names not in omega: {sorted(unknown)}")
        np.linalg.cholesky(self.cov)  # raises if not positive definite

    @classmethod
    def from_variances(
        cls,
        variances: Mapping[str, float],
        covariances: Mapping[tuple[str, str], float] | None = None,
        additive: Iterable[str] = (),
    ) -> Omega:
        """NONMEM-style ``$OMEGA``: variances and (optional) covariances."""
        names = tuple(variances)
        cov = _symmetric(names, variances.values(), covariances or {})
        return cls(names, cov, additive)

    @classmethod
    def from_cv(
        cls,
        cv_percent: Mapping[str, float],
        correlations: Mapping[tuple[str, str], float] | None = None,
    ) -> Omega:
        """Log-normal IIV given as CV% (``ω² = ln(1 + CV²)``) and correlations."""
        names = tuple(cv_percent)
        cv = np.array(list(cv_percent.values()), dtype=float) / 100
        sd = np.sqrt(np.log(cv**2 + 1))
        corr = _symmetric(names, np.ones(len(names)), correlations or {})
        return cls(names, corr * np.outer(sd, sd))

    @property
    def variances(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, np.diag(self.cov))))

    def draw(self, rng: np.random.Generator | None = None, n: int = 1) -> FloatArray:
        """``n`` eta vectors, shape ``(n, len(names))``."""
        rng = np.random.default_rng() if rng is None else rng
        return rng.multivariate_normal(np.zeros(len(self.names)), self.cov, size=n)

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        """One eta per parameter."""
        return dict(zip(self.names, map(float, self.draw(rng)[0])))

    def apply(
        self, values: Mapping[str, float], eta: Mapping[str, float]
    ) -> dict[str, float]:
        """Individual values from typical ``values`` and ``eta`` (missing → 0)."""
        if unknown := set(eta) - set(self.names):
            raise ValueError(f"etas not in omega: {sorted(unknown)}")
        out = {}
        for name in self.names:
            e = eta.get(name, 0.0)
            v = values[name]
            out[name] = v + e if name in self.additive else v * exp(e)
        return out

    def __repr__(self) -> str:
        return (
            f"Omega(names={self.names!r}, cov={self.cov.tolist()!r}, "
            f"additive={sorted(self.additive)!r})"
        )


class RandomEffects:
    """Mixin for models with IIV. Subclasses set ``omega`` and the two hooks."""

    omega: ClassVar[Omega | None] = None

    def _get_values(self, names: tuple[str, ...]) -> dict[str, float]:
        raise NotImplementedError

    def _set_values(self, values: Mapping[str, float]) -> None:
        """Replace the named parameters **in place** (called on a copy)."""
        raise NotImplementedError

    def _require_omega(self) -> Omega:
        if self.omega is None:
            raise NotImplementedError(f"{type(self).__name__} has no IIV (omega)")
        return self.omega

    def _apply_eta(self, eta: Mapping[str, float]) -> None:
        omega = self._require_omega()
        self._set_values(omega.apply(self._get_values(omega.names), eta))

    def with_eta(self: R, eta: Mapping[str, float]) -> R:
        """Copy of the model with the random effects ``eta`` applied."""
        new = copy.copy(self)
        new._apply_eta(eta)
        return new

    def sample(self: R, rng: np.random.Generator | None = None) -> R:
        """Copy of the model for an individual drawn from ``N(0, Ω)``."""
        return self.with_eta(self._require_omega().sample(rng))
