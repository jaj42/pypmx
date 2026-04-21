import numpy as np

def eigendecomposition(S,V1,ke0: float=None):
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eig(S)

    # Sort by magnitude (most negative first)
    sort_idx = np.argsort(eigvals)
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    # lambda_i are positive (decay constants)
    lambdas = -eigvals

    if ke0 is not None:
        # Find ke0 index
        ke0_idx = np.argmin(np.abs(lambdas - ke0))

        # Create mask to seperate PK eigenvalues from ke0
        pk_mask = np.ones_like(lambdas, dtype=bool)
        pk_mask[ke0_idx] = False

        # Reconstruct in STANPUMP order: [λ1_pk, λ2_pk, λ3_pk, ke0]
        lambdas = np.append(
            lambdas[pk_mask],
            lambdas[ke0_idx],
        )
        eigvecs = np.column_stack(
            [
                eigvecs[:, pk_mask],
                eigvecs[:, ke0_idx],
            ]
        )

    # S can be reconstructed from eigenvalues and eigenvectors
    S_reconstructed = eigvecs @ np.diag(-lambdas) @ np.linalg.inv(eigvecs)
    assert np.allclose(S, S_reconstructed)

    # Initial condition for unit bolus in central compartment
    C0 = np.array([1.0 / V1, 0.0, 0.0, 0.0])  # µg/L

    # Modal coordinates
    alpha = np.linalg.inv(eigvecs) @ C0  # µg/L

    # Plasma coefficients (row 0 = Cp)
    # p_coef = eigvecs[0, :] * alpha / lambdas  # sec/L

    # Effect site coefficients (row 3 = Ce)
    # e_coef = eigvecs[3, :] * alpha / lambdas  # sec/L

    coefs = eigvecs @ np.diag(alpha / lambdas).T

    return lambdas, coefs


def calculate_impulse_response(lambdas, coefs, t):
    A_bolus = coefs * lambdas  # [L^-1]
    decay = np.exp(-lambdas[:, np.newaxis] * t[np.newaxis, :])
    return np.sum(A_bolus[:, np.newaxis] * decay, axis=0)

def convolve(uir, inf, dt: float):
    npoints = len(uir)
    e_res = np.convolve(uir, inf, mode="full")
    return e_res[:npoints] * dt