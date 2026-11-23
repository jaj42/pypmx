# PyPMX - Pharmacometric Models in Python

**Early development stage, for educational purposes only!**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-ISC-blue.svg)](https://opensource.org/licenses/ISC)

**PyPMX** is a Python library for pharmacometric modeling, providing implementations of pharmacokinetic (PK) and pharmacodynamic (PD) models for various drugs.

## Features

- **Comprehensive drug models**: Propofol, Remifentanil, Noradrenaline, Sufentanil, and more (see below)
- **Standard PK/PD models**: 2- and 3-compartment models, effect-site equilibration
- **Physiological calculations**: Lean body mass, covariate adjustments
- **Numerical solvers**: ODE integration for drug concentration simulations
- **Clinical utilities**: Infusion rate calculations, piecewise rate functions

## Drug models

All models are exported from `pypmx.drugs`. Time is in minutes.

| Drug | Class | Source |
|---|---|---|
| Propofol | `EleveldPropofol` | Eleveld et al. |
| Propofol (haemodynamics) | `SuPropofolHemodynamics` | Su et al. |
| Propofol + remifentanil (haemodynamics) | `SuPropofolRemifentanilHemodynamics`, `SuPropofolRemifentanilHemodynamicsVariant` | Su et al. |
| Sufentanil | `GeptsSufentanil` | Gepts et al. Anesthesiology 1995;83:1194 |
| Dexmedetomidine | `HannivoortDexmedetomidine`, `DyckDexmedetomidine` | Hannivoort et al.; Dyck et al. |
| Noradrenaline | `Noradrenaline` | 2-compartment with endogenous production, lag and MAP Emax PD (`simulate(...).effect`) |
| Cefazolin | `Cefazolin` | Sci Rep 2021;11:19763 |
| Tranexamic acid | `GrassinDelyleTranexamicAcid` | Grassin-Delyle et al. BJA 2022;128:465 (2-compartment, IV only) |
| Paracetamol | `AllegaertParacetamol` | Allegaert et al. BMC Anesthesiol 2014;14:77 (3-compartment; pregnancy / surgery / ICU settings) |
| Midazolam | `ZomorodiMidazolam` | Zomorodi et al. Anesthesiology 1998;89:1418 (3-compartment; simple or BSA-adjusted) |
| Lidocaine | `KuipersLidocaine` | Kuipers et al. Anesthesiology 2001;94:566 (3-compartment; models A–D, cardiac-output covariate) |

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the ISC License.

## Support

For questions or issues, please open a GitHub issue.