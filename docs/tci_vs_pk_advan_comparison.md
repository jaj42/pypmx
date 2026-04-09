# Comparison: `tci.py` vs `pk_advan.py` — PK Model Equations

## Parameters

|                        | `tci.py`                       | `pk_advan.py`                           |
| ---------------------- | ------------------------------ | --------------------------------------- |
| Volume                 | `vc` (central only)            | none (works in amounts)                 |
| Rates                  | `k10, k12, k21, k13, k31, ke0` | `k10, k12, k13, k20, k21, k30, k31, ka` |
| Peripheral elimination | **No** (`k20=k30=0` assumed)   | **Yes** (`k20, k30` supported)          |
| Effect site            | Yes (`ke0`)                    | No                                      |
| Oral absorption        | No                             | Yes (`ka`)                              |

---

## Lambda (eigenvalue) calculation

Both build the same rate matrix `K`, but `pk_advan.py` includes peripheral elimination terms
`k20`, `k30` that `tci.py` omits:

|       | `tci.py` diagonal | `pk_advan.py` diagonal   |
| ----- | ----------------- | ------------------------ |
| Row 1 | `-(k10+k12+k13)`  | `-(k10+k12+k13)`         |
| Row 2 | **`-k21`**        | **`-(k20+k21)`** = `-E2` |
| Row 3 | **`-k31`**        | **`-(k30+k31)`** = `-E3` |

---

## 1-Compartment

**`tci.py`** — Bateman coefficient for the unit disposition function (plasma concentration per
unit infusion rate):

```
p_coef = 1 / (λ₁ · Vc)
```

State update per time step:

```
A(t+dt) = A · e^(-λ·dt) + p_coef · rate · (1 - e^(-λ·dt))
```

**`pk_advan.py`** — direct closed-form for compartment amount:

```
IV Bolus:    X1 = A1 · e^(-k10·t)
IV Infusion: X1 = (R/k10) · (1 - e^(-k10·t)) + A1 · e^(-k10·t)
```

Equivalent with λ₁ = k10, but `tci.py` tracks **concentration** while `pk_advan.py` tracks
**amounts**.

---

## 2-Compartment

**`tci.py`** — Bateman-form coefficients (`k20=0` assumed, so `E2 = k21`):

```
p_coef[0] = (k21 - λ₁) / ((λ₂ - λ₁) · Vc · λ₁)
p_coef[1] = (k21 - λ₂) / ((λ₁ - λ₂) · Vc · λ₂)
```

**`pk_advan.py`** — full state-transition from arbitrary initial conditions (Eq. 11 / 27):

```
num1 = (A1·E2 + A2·k21) - A1·λ₁
num2 = (A1·E2 + A2·k21) - A1·λ₂
X1   = (num1·e^(-λ₁t) - num2·e^(-λ₂t)) / (λ₂ - λ₁)
```

where `E2 = k20 + k21` (generalises `tci.py`'s bare `k21`).

The IV infusion term adds:

```
X1 += R·E2 · [1/(λ₁λ₂) + e^(-λ₁t)/(λ₁(λ₁-λ₂)) - e^(-λ₂t)/(λ₂(λ₁-λ₂))]
```

The `tci.py` coefficients correspond to `pk_advan.py` evaluated with `A1 = 1/Vc`, `A2 = 0`,
`R = 0`, and `k20 = 0`.

---

## 3-Compartment

**`tci.py`** — Bateman-form coefficients (Shafer TCI form, `k20 = k30 = 0`):

```
p_coef[i] = (k21 - λᵢ) · (k31 - λᵢ)
           / ((λⱼ - λᵢ) · (λₖ - λᵢ) · Vc · λᵢ)
```

for each permutation of `{i,j,k} = {0,1,2}`.

**`pk_advan.py`** — full state-transition (Eq. 18 / 29), with `E2 = k20+k21`,
`E3 = k30+k31` and helper terms `B`, `C`:

```
B  = A2·k21 + A3·k31
C  = A2·k21·E3 + A3·k31·E2

X1 = A1 · Σᵢ [ (E2-λᵢ)·(E3-λᵢ) / (∏ⱼ≠ᵢ(λⱼ-λᵢ)) · e^(-λᵢt) ]
   +      Σᵢ [ (C - B·λᵢ)       / (∏ⱼ≠ᵢ(λⱼ-λᵢ)) · e^(-λᵢt) ]
```

The IV infusion term adds (Eq. 29):

```
X1 += R · E2·E3/(λ₁λ₂λ₃)
    - R · Σᵢ [ (E2-λᵢ)·(E3-λᵢ) / (λᵢ · ∏ⱼ≠ᵢ(λⱼ-λᵢ)) · e^(-λᵢt) ]
```

With `k20 = k30 = 0` and `A2 = A3 = 0`, the numerators of `pk_advan.py` reduce to
`(k21-λᵢ)·(k31-λᵢ)`, recovering `tci.py`'s form (the `1/(Vc·λᵢ)` factor arises from
the infusion-rate normalisation used in TCI).

---

## Effect-site coefficients (`tci.py` only)

The fourth coefficient tracks the effect-site response. For each compartment model:

| Model | `e_coef[3]`                                                  |
| ----- | ------------------------------------------------------------ |
| 1-cpt | `1 / ((λ₁ - ke0) · Vc)`                                      |
| 2-cpt | `(k21 - ke0) / ((λ₁ - ke0) · (λ₂ - ke0) · Vc)`               |
| 3-cpt | `(ke0 - k21)·(ke0 - k31) / ((λ₁ - ke0)·(λ₂ - ke0)·(λ₃ - ke0)·Vc)` |

The remaining `e_coef[0..2]` are rescaled from `p_coef`:

```
e_coef[i] = p_coef[i] · ke0 / (ke0 - λᵢ)
```

---

## A note on terminology: "Bateman equations"

The term *Bateman equations* originates from **nuclear physics**: Harry Bateman (1910) derived
analytical solutions for radioactive decay chains, describing how daughter nuclides accumulate
and decay in a serial first-order chain. The mathematical structure is identical to linear
compartmental PK models (both are systems of linear first-order ODEs), which is why the term
occasionally appears in pharmacokinetics literature.

In PK, "Bateman equation" most precisely refers to the **1-compartment oral absorption** equation:

```
C(t) = (F · D · ka) / (Vd · (ka - ke)) · (e^(-ke·t) - e^(-ka·t))
```

The multiexponential expressions in `tci.py` are more accurately called **polyexponential
disposition equations** or **Laplace-transform solutions** of the compartmental ODE system.
The specific coefficient form used there — normalised to a unit infusion rate — are the
**Shafer/Sheiner TCI coefficients**, from Shafer & Gregg (1992) on TCI algorithms.

The term "Bateman-form" in this document is used loosely to mean *expressed as a sum of
exponentials with pre-computed scalar coefficients*, which is the structural analogy with
Bateman's nuclear decay solutions, but is not standard PK nomenclature.

---

## Summary of key differences

| Aspect                     | `tci.py`                                         | `pk_advan.py`                                           |
| -------------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| **Peripheral elimination** | Assumes `k20 = k30 = 0`                          | Supports non-zero `k20`, `k30`                          |
| **Units**                  | Concentration (÷ Vc)                             | Amounts                                                 |
| **Effect site**            | Yes (`ke0`)                                      | No                                                      |
| **Oral absorption**        | No                                               | Yes (`ka`)                                              |
| **Equation form**          | Bateman / Sheiner step-response coefficients     | Full state-transition from arbitrary initial conditions |
| **Primary use**            | TCI dose calculation (response to unit infusion) | Simulation / NLME (propagate state forward in time)     |

The 3-compartment plasma coefficients are mathematically equivalent when `k20 = k30 = 0` and
the initial conditions are a unit dose in the central compartment with all other compartments
empty.
