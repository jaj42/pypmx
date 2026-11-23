# Unify the PK model API

## Context
The drug models in pypmx use at least four styles:
- **NamedTuple style** (Hannivoort, Dyck, Minto, Eleveld ×2, Schnider, GrassinDelyle, Allegaert, Zomorodi, Kuipers). Each one repeats the micro-constant derivation, the `system_matrix` literal, `partial(pk_Ncomp, …)` and the range warnings.
- **Attribute style** (Marsh, Kataria, Paedfusor). These have no Parameters and no matrix, and use the typo `keo`.
- **Stateful datetime/seconds style** (Sufentanil, Cefazolin). These use a concentration-state ODE.
- **Noradrenaline**. It uses dict params with uppercase keys, endogenous production (`EPROD`), a lag, a steady-state y0, an Emax PD and `solve_ivp`.

There is also a hidden inconsistency. Every `system_matrix` is written in **concentration space** (row 0 = `[-k123, k12, k13]`), but `.ode` (pk_3comp), `tci.py` and `pk_advan.py` work in **amounts**. The eigenvalues agree, so the tests never noticed.

Goal: one base class that every linear PK model inherits. Each drug then only maps covariates to macro parameters. The base class provides the matrix, the ODE and an eigen-based simulator. The simulator takes the forcing term from pmxmc `eig.py`, which lets Noradrenaline use it too.

## Key design decision: state space = amounts + Ce (mixed units)
A linear system does not need every state in the same unit. Ce needs no effect-site volume:
`Ce' = ke0·(A1/V1 − Ce)` gives the row `[ke0/V1, 0, 0, −ke0]`.
pk_3comp already does exactly this. The matrix stays block lower-triangular, so its eigenvalues are unchanged ({PK eigenvalues} ∪ {ke0}).
- B = e₀ (the dose goes into A1, in amount/time). The forcing is f = [EPROD, 0, …] (amount/time).
- Observation: Cp = x[0]/V1, Ce = x[-1].
- Models with no effect site (ke0 = 0) **omit the Ce state**. This removes the zero eigenvalue and the "drop the last mode" nan hack in the tests. The solver is also made λ=0-safe for generality.

## Steps
Each step is one commit on `main`. Before every commit:
- Run `ruff check`/`ruff format` on the touched files.
- Run `uv run ty check src tests` (type checker).
- Run `uv run pytest tests`.

Each step adds or updates pytest sanity checks (memory: tests-per-step).

### 0. Tooling and golden outputs from the current code (before any refactor) — DONE (f9301d7)
**Outcome / deviations** (the original spec follows below):
- `[tool.ty]`: python 3.10, roots `src` + `tests` (`tests` so that `from golden import generate` resolves).
- Diagnostics fixed: covariates typed `Mapping[str, Any]` (Schnider too), `tci.py` arrays are `NDArray`, `eigendecomposition(ke0: float | None)`, `utils.calculate_drug_dose` defines `Q_`, Su return types. `filter_valid` is **already moved** to `pypmx.utils` (single copy, used by both Su models); `utils.py` uses `from __future__ import annotations`.
- `[tool.ty.src] exclude` (legacy): `ode.py`, `drugs/alfentanil.py`, `drugs/cefazolin.py`, `drugs/noradrenaline_old.py`, `drugs/sufentanil.py`. **Each step that deletes/rewrites one of these removes it from the list** (ode.py → step 3, noradrenaline_old → step 4, the rest → step 5). Known bug there: Sufentanil's `rate` setter calls the non-existent `run_pk`.
- 60 snapshots. The `RATES` table in `generate.py` gives a per-drug (load, maintenance) rate; grid `linspace(0, 240, 481)`, breakpoints `(0, 2, 60)`. Each `.npz` holds `meta` (JSON: model, case, kind, covars, options, infu_time/infu_rate), `t`, `param_names`/`param_values`, `eigenvalues`, `cp`, `ce` (only if ke0 > 0), `effect` (NA MAP) or `hr`/`sv`/`map` (Su).
- Eigenvalues stored = PK block ∪ {−ke0 if ke0 > 0} (no zero eigenvalue), i.e. what the new amounts+optional-Ce matrix gives.
- Models defined by rate constants (Marsh, Kataria, Paedfusor, Sufentanil): V2 = Q2/k21, V3 = Q3/k31 (not the published, rounded volumes; Marsh 0.464·wt). Step 5 must derive them the same way.
- Noradrenaline: its `ode` is integrated directly with tight tolerances (its `__call__` uses default rtol 1e-3); effect = its `pd(ce)`. y0 = `get_state()`. Saved params: V1, V2, CL, Q, EPROD, KE0, E0, EC50, EMAX, GAMMA, PKLAG, PDLAG (step 4 adapter maps them to the new names).
- Su PD models run through their public method with their **own** solver settings (pure regression guard; not touched by the refactor).
- Allegaert "pregnancy" case uses `sex="f"`; generation runs with warnings as errors, so all cases are inside the valid ranges.
- `tests/test_golden.py` has `ADAPTERS: dict[model_key, meta -> outputs]`; models without an adapter fall back to `generate.compute` (self-check). **Every migration step adds the new-API adapter for the models it migrates** (outputs dict: `params`, `eigenvalues`, `cp`, `ce`/`effect`), and the old-API branch in `generate.py` stays untouched. Once old code is deleted, `generate.py` can no longer run; that is expected. Caveat: `test_golden.py` imports `generate` (for `CASES`, `case_path`, `to_arrays`), so when a step removes or renames an old class that `generate.py` imports (e.g. step 5), first move those helpers and the case list to `tests/golden/cases.py` and import the old models lazily inside the `compute_*` functions.

**Original spec:**
- **ty**:
  - Add `ty` to the `dev` dependency group (`uv add --dev ty`) and add a `[tool.ty]` section to `pyproject.toml` (src layout, python ≥3.10).
  - Fix or annotate the existing diagnostics so the baseline is clean, or record a short ignore list for legacy files that get deleted in step 5.
  - New code (`pk.py`, `eigen.py`, migrated drugs) is fully annotated: `NDArray[np.float64]`, `NamedTuple` fields, and `ClassVar` for `units`/`covariate_ranges`.
- **Golden snapshot**:
  - Write `tests/golden/generate.py`. It imports the **current** (pre-refactor) code and, for every drug model in `drugs/__init__.all` plus Marsh/Kataria/Paedfusor/Schnider/Minto/Eleveld-remi, saves `tests/golden/<model>_<case>.npz`.
  - Cases: default covariates, plus 1–2 edge covariate sets inside the valid range, plus every constructor variant (Hannivoort effect, Kuipers model A–D, Allegaert setting, Zomorodi bsa).
  - Each case saves the PK parameters (V, CL, Q, ke0) and the micro constants, the sorted eigenvalues, and Cp/Ce on a fixed time grid.
  - The time grid uses a standard regimen: a 2-min loading infusion then a maintenance infusion, a stop, and washout.
  - Simulation for the golden uses `solve_ivp(..., rtol=1e-10, atol=1e-12)` on each model's **own** old ODE.
  - Old models have their own quirks:
    - Attribute-style models (Marsh, Kataria, Paedfusor): use a generic amount-space 3-comp ODE built from their attributes.
    - Sufentanil/Cefazolin (seconds, concentration states): convert to minutes for the snapshot.
    - Noradrenaline: use its own `__call__` → t, Cp, Ce, MAP, both at baseline and with a lagged infusion.
    - Su PD models: given fixed Cp functions → HR/SV/MAP, with `random_effects=False`.
- `tests/test_golden.py` is parametrized over all snapshots and compares the **new** API output against them. Before migration it runs against the old code, as a self-check.
  - Parameters/eigenvalues: `rtol=1e-10`.
  - Cp/Ce/effect: `rtol=1e-6, atol=1e-9`. These tolerances absorb the solve_ivp vs eig_advan difference.
  - Snapshots are committed. They are regenerated only on purpose, and the commit message says why.
- This test must pass after **every** later step. Any intended numerical change has to be explicit (for example, if a bug fix moves a value, the snapshot is updated in that commit with a note).

### 1. NumPy eigen engine (`src/pypmx/eigen.py`) — DONE (1c324e7)
**Outcome / deviations** (the original spec follows below):
- Breakpoints are `{t0} ∪ (infu_time + lag) ∪ meas_time` (not the unshifted infusion times as in pmxmc), so lag is exact; `t0 = min(infu_time, meas_time)` is where `y0` holds.
- `Modes` are sorted slowest first (`lambdas` ascending). `calculate_impulse_response(modes, t, state=0)`.
- `utils.rate_at` is vectorised. `tci.stanpump_order(modes, ke0)` holds the reordering (PK fastest first, ke0 last); not yet used by tci (step 6).
- The 4 existing eigen tests (midazolam, TXA, paracetamol, lidocaine) were ported to the new API. The drug matrices are still concentration space, so they pass `B = e₀/V1` (`_unit_bolus` helper); step 3 replaces this. No more nan/last-mode dropping.

**Original spec:**
Port `/home/jaj/devel/pmxmc/src/pmxmc/advan/eig.py` to numpy:
- `eigendecomposition(S, B, f=None, y0=None, real_eigenvalues=True) -> Modes(lambdas, eigvecs, alpha_B, alpha_f, y0_mod)`.
  - Store the modal weights **without** dividing by λ.
  - Apply `phi(λ, dt) = -expm1(-λ·dt)/λ` (→ dt as λ→0) in the step. This avoids nan for zero eigenvalues.
- `eig_advan(S, B, meas_time, infu_time, infu_rate, y0=None, f=None, lag=0.0) -> (n_meas, n_states)`.
  - Breakpoints are the union of infusion and measurement times, giving a piecewise-constant rate.
  - The step is `A ← A·e^{−λdt} + (eigvecs·(α_B·rate + α_f))·phi`, with a plain loop or `np.cumprod`-free iteration.
  - Lag works by evaluating the rate at `t − lag`.
- Add `rate_at` (numpy) to `src/pypmx/utils.py`, taken from pmxmc `rate_at_numpy`.
- Move the STANPUMP ke0-last reordering out of `eigen.py` and into `tci.py`, which is its only consumer. Keep `calculate_impulse_response`/`convolve`, rewritten on `Modes`.
- Tests (`tests/test_eigen.py`):
  - Reconstruct S. Compare `eig_advan` with `solve_ivp` on a random 3-comp+Ce system for a bolus-like short infusion and a multi-step regimen.
  - With forcing, starting from steady state `-S⁻¹f`, the state stays constant.
  - A zero eigenvalue gives a finite result (pure integrator ⇒ linear growth).

### 2. Base class (`src/pypmx/pk.py`) — DONE (c344554)
**Outcome / deviations** (the original spec follows below):
- `PKParameters` field order is `V1, CL, V2=nan, V3=nan, Q2=0, Q3=0, ke0=0` (a NamedTuple field cannot be required after a defaulted one). `k21`/`k31` are 0 (not nan) when the matching Q is 0. `n_cmt` raises if Q3 ≠ 0 while Q2 = 0. There is also a `has_effect_site` property (ke0 ≠ 0).
- `pk.py` already defines `Units` and a default `Covariates` (age 35, weight 70, height 170, sex "m"). Step 3 uses them, or drug-specific NamedTuples, through `_set_covariates(covars, Cls)`.
- `forcing` and `lag` are properties, so subclasses override them with properties (step 4).
- `covariate_ranges: {name: (lo, hi)}`. `_check_ranges(values, stacklevel)` is also called directly for derived covariates (e.g. Kuipers QI in step 3).
- `_set_covariates` uses `filter_valid`, so unknown keys are now **ignored silently**. Before, `Covariates(**covars)` raised on them.
- `Prediction(t, cp, ce, states)`. The `effect` field is still to be added in step 4.
- `tests/test_pk.py` checks: micro constants, n_cmt, matrix shape and eigenvalues for 1/2/3 comp ± Ce, `ode` vs `system_matrix` and vs the legacy `pk_2comp`/`pk_3comp`, `simulate` vs `solve_ivp(m.ode)`, the 1-comp analytic solution, Ce → Cp at steady state, forcing steady state, lag shift, `with_parameters`, range warnings and `repr`. Step 3 has to drop the `standard_models` cross-check when it deletes that module.

**Original spec:**
- `PKParameters(NamedTuple)`: `V1, V2=nan, V3=nan, CL, Q2=0, Q3=0, ke0=0`. The micro constants `k10, k12, k21, k13, k31` are properties. `n_cmt` is inferred (Q3 == 0 ⇒ 2-comp, Q2 == 0 ⇒ 1-comp).
- Class `LinearPKModel`:
  - Class attributes: `units`, `covariate_ranges: dict[str, tuple]`.
  - Instance attributes: `covars`, `parameters` (the drug-specific NamedTuple of thetas, refs and PD), and `pk: PKParameters`.
  - `system_matrix` (amounts + optional Ce), `input_matrix`, `forcing` (default zeros), `lag` (default 0).
  - `steady_state()` returns `solve(S, -f)`, or zeros when f = 0. It is the default `y0`.
  - `ode(t, y, Inf)` = `S@y + B·Inf(t−lag) + f`. It keeps the `solve_ivp(m.ode, …, args=(inf,))` call signature.
  - `simulate(infu_time, infu_rate, meas_time, y0=None) -> Prediction(t, cp, ce, states)` via `eig_advan`.
  - `with_parameters(**kw)` returns a copy with `pk` fields replaced. This is the general form of Noradrenaline's `copy_with`/`update_parameters`.
  - Generic `__repr__`.
  - `_set_covariates(covars, Covariates)` builds the NamedTuple, uses `utils.filter_valid` (already moved in step 0) and warns outside `covariate_ranges`.
- Tests: `PKParameters` micro constants, matrix shape for 1/2/3 comp ± effect site, and `ode` vs `system_matrix` consistency.

### 3. Migrate the NamedTuple-style models — DONE (949259d)
**Outcome / deviations** (the original spec follows below):
- Migrated: Eleveld/Schnider propofol, Eleveld/Minto remifentanil, Hannivoort/Dyck dexmedetomidine, Grassin-Delyle TXA, Allegaert paracetamol, Zomorodi midazolam, Kuipers lidocaine. Marsh/Kataria/Paedfusor are untouched in `propofol.py` (step 5) and still use its `Covariates` (which now also feeds Eleveld/Schnider).
- `parameters` is **no longer declared on the base** (ty rejects narrowing a mutable attribute). Subclasses that have one declare `parameters: Parameters`: it holds only thetas/refs/PD and derived covariates (Eleveld propofol `ce50`/`bis_delay`, Hannivoort `ce50`, Kuipers `QI`, Zomorodi `BSA`). PK values and micro constants live only in `m.pk`. Dyck, Minto, Schnider and Eleveld remifentanil have no `parameters`.
- Base gained `options: ClassVar[tuple[str, ...]]`: constructor options shown by `repr` (`effect`, `setting`, `bsa_adjusted`, `model`).
- Drug-specific `Covariates` are kept where defaults differ from `pk.Covariates` (TXA, paracetamol, midazolam, lidocaine, propofol `opiates`). `age` is now typed `float`.
- Range warnings now come from `covariate_ranges` and use the covariate name (`weight`, `age`, `QI`, `cardiac_output`). Tests match on those names. Paracetamol's "pregnancy with sex m" warning stays hand-written.
- Hannivoort `units.effect` is no longer set per instance (`units` is a ClassVar); the endpoint is `m.effect`.
- Minto is written in its published macro form (CL, Q2, Q3), which is identical to the old k-form up to rounding. **Schnider** keeps the k-form (V1 + k10…k31, as in STANPUMP) with `V2 = Q2/k21`, `V3 = Q3/k31`. **The Schnider snapshots were regenerated** (`MICRO_DEFINED` in `generate.py`): only the reported V2/V3 change (25.938 → 25.907, 238 → 239.12 at default covariates). Cp/Ce/eigenvalues are bit-identical.
- `SIM_RTOL` in `test_golden.py` is **1e-6 → 2e-5**. The snapshots' solve_ivp reference is only accurate to ~8e-6 relative (checked against `expm`), while `eig_advan` matches `expm` to ~1e-14. Parameters and eigenvalues are still checked at 1e-10.
- `test_golden.py` has a generic `linear_pk_adapter(cls)` for all migrated models. `standard_models.py` and `ode.py` are deleted. The legacy cross-check in `test_pk.py` is replaced by a written-out 3-comp+Ce equations test. New `tests/test_linear_models.py` covers every migrated model: structure, `simulate` vs `solve_ivp(m.ode)`, steady state rate/CL, repr and derived parameters.

**Original spec:**
Pattern for each drug:
- The class inherits `LinearPKModel`. `__init__` computes `CL, Q2, Q3, V1–V3, ke0` and builds `self.pk`.
- Remove the per-drug `system_matrix`, k-derivation, `partial(pk_Ncomp)` and the hand-written range warnings. The ranges move to `covariate_ranges`.
- Representative files: `drugs/dexmedetomidine.py`, `drugs/propofol/propofol.py` (Eleveld, Schnider), `drugs/remifentanil.py`, `drugs/tranexamic_acid.py`, `drugs/paracetamol.py`, `drugs/midazolam.py`, `drugs/lidocaine.py`.
- Move the shared `Units`/`Covariates` base fields into `pk.py`. Drugs subclass or extend them where they need extras (Kuipers `cardiac_output`, Allegaert `setting`).
- Delete `standard_models.py` (pk_2comp/pk_3comp) and `ode.py`; drop `ode.py` from the ty exclude list. Update the `ebe.py` docstring reference.
- Add `ADAPTERS` entries in `tests/test_golden.py` for every migrated model.
- Update the existing tests:
  - `m.parameters.V1` becomes `m.pk.V1`.
  - The eigen check becomes `m.simulate(...)` vs `solve_ivp(m.ode)`, with no last-mode dropping.
  - All published-value checks (Cmax, t½, Css …) must still pass unchanged.

### 4. Noradrenaline on the base class (`drugs/noradrenaline.py`) — DONE
**Outcome / deviations** (the original spec follows below):
- Population values are a `Thetas` NamedTuple (`Noradrenaline.thetas`, ClassVar, 70 kg reference). `parameters` is a `Parameters` NamedTuple of **individual** non-PK values only: `EPROD` (weight-scaled), `E0, EC50, EMAX, GAMMA, PKLAG, PDLAG`. The PK values (V1, V2, CL, Q→Q2, KE0→ke0) live only in `m.pk`, as in step 3.
- Own `Covariates(weight=70)`. A missing weight now falls back to 70 kg (the old code raised KeyError). No `covariate_ranges` (none documented).
- Base: `Prediction.effect` (default None) and a `pd(ce)` hook on `LinearPKModel` that returns None. `simulate` fills `effect = pd(ce)` when there is an effect site.
- `Noradrenaline.with_parameters` overrides the base one. It accepts `pk` fields **and** `Parameters` fields (e.g. `EPROD`, `PKLAG`), and raises ValueError for unknown names. This replaces `copy_with(params=...)`. `pd(ce, E0=None)` keeps the optional E0 override.
- Units: nmol, L, min, nmol/L, MAP mmHg (from the 0.261 nmol/min = 5 µg/h note in the old file).
- `generate.compute_noradrenaline` still uses the old API (it can no longer run). It is typed `m: Any` so ty stays clean. `test_golden.py` has `noradrenaline_adapter`.
- `tests/test_noradrenaline.py` checks: structure and allometric scaling, the endogenous steady state (Cp = EPROD/CL, MAP at baseline), `simulate` vs `solve_ivp(m.ode)` with lag, the exact lag shift, the infusion response (Css, MAP rise and washout), `with_parameters` and `repr`.

**Original spec:**
- Replace `_thetas` with a `Parameters` NamedTuple: `CL, Q, V1, V2` map to `pk` (Q → Q2), plus `EPROD, E0, EC50, EMAX, GAMMA, PKLAG, PDLAG`. Keep the allometric scaling, including ke0·(wt/70)^0.25.
- `forcing = [EPROD, 0, 0]`, `lag = PKLAG + PDLAG`. The default y0 is `steady_state()`, which reproduces the current `EPROD/CL·V` init.
- `pd(ce)` is the Emax/Hill model. `simulate` adds a `map` (effect) field when the model defines `pd`. The base `Prediction` gets an optional `effect` field.
- Drop `noradrenaline_old.py` (and from the ty exclude list). Add the Noradrenaline golden adapter (map `Q`→`Q2`, `KE0`→`ke0`).
- This is a breaking API change. The old `Noradrenaline(covars, t0, y0, rate_fn, params)(t)` becomes `Noradrenaline(covars).simulate(infu_time, infu_rate, t)`, and `copy_with` becomes `with_parameters`.
- Tests (`tests/test_noradrenaline.py`):
  - With no infusion the state stays at the endogenous steady state (Cp = EPROD/CL), and MAP = E0 + Emax-term at baseline.
  - `simulate` matches `solve_ivp(m.ode)` for an infusion with lag.
  - The lag shifts the response by `PKLAG+PDLAG`.

### 5. Migrate the legacy models and remove dead code
- Convert Marsh, Kataria and Paedfusor to macro parameters (`Q2 = k12·V1`, `V2 = Q2/k21`, …) on the base class, and fix `keo` → `ke0`. Kataria and Paedfusor warnings move to `covariate_ranges`.
- Rewrite Sufentanil (Gepts) and Cefazolin as stateless base-class models **in minutes**. The datetime/`rate` setter/`give_bolus` state machine is dropped (the user can keep state in the app layer).
- Delete `drugs/alfentanil.py`, which imports a non-existent `.base`. Optionally re-add Maitre on the base class if the parameters are at hand. Otherwise list it as a TODO in the README.
- Update `drugs/__init__.py` exports and README. Empty the ty exclude list (cefazolin, sufentanil, alfentanil). Add golden adapters for Marsh/Kataria/Paedfusor/Sufentanil/Cefazolin (snapshot V2/V3 are already Q/k-derived; Cefazolin cases are `CrCL` 0/40/80/130).
- Tests: the shared sanity checks for each migrated model. For Marsh, the derived V2 is about 0.46·wt, which matches the original.

### 6. tci.py on the unified model
- `tci.py` takes a `LinearPKModel` (or `PKParameters`) instead of its own `Parameters` dataclass. Lambdas and coefficients come from `eigen.eigendecomposition` plus the STANPUMP ordering helper from step 1.
- Seconds conversion happens at this boundary only. This also fixes the bug where `per_minutes` sets `time_unit="seconds"`.
- Tests: `tci.model` plasma-target rate for Eleveld propofol is unchanged before and after.

### 7. (Optional follow-ups, not in scope unless requested)
- **Shared IIV API.** Su models have `random_effects`/`covariance`/`sample_etas`, and Eleveld remifentanil has `bsv`. Unify them as an `omega` + `sample(rng)` on the base.
- **PD-only models.** The Su models should consume `Prediction.cp_function()` via `utils.continuize_order_one`, and use the single `utils.ConcentrationFunction` (it is currently duplicated in `su_variant.py`).
- **Cross-check.** Check `advan/pk_advan.py` against `eig_advan` in tests, as a second independent solver.

## Critical files
`src/pypmx/eigen.py`, `src/pypmx/pk.py` (new), `src/pypmx/utils.py`, `src/pypmx/tci.py`, `src/pypmx/drugs/*.py`, `src/pypmx/drugs/__init__.py`, `tests/*`. Delete `standard_models.py`, `ode.py`, `drugs/noradrenaline_old.py` and `drugs/alfentanil.py`.
Reuse: pmxmc `eig.py` (algorithm), pmxmc `utils.rate_at_numpy`, `pypmx.utils.continuize_order_zero`/`filter_valid`, `pypmx.leanbodymass`.

## Verification
- `tests/test_golden.py` checks that every drug and variant gives the same parameters, eigenvalues and Cp/Ce (plus MAP and HR/SV) as the pre-refactor code, after every step.
- `uv run ty check src tests` is clean after every step.
- `uv run pytest tests` passes after every step. The existing published-value checks (TXA Cmax/time > 10 mg/L, paracetamol Css, midazolam t½ 905 min, lidocaine Css/CL) are the regression guard for steps 2–3.
- New tests compare `eig_advan` with `solve_ivp`, for both forcing and lag, and check the Noradrenaline steady state.
- `ruff check src tests`.
