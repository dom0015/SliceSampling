"""
Pomocný modul s implementacemi MCMC samplerů:
  - Slice Sampling (univariate, stepping-out)
  - Random Walk Metropolis (RWM, speciální případ Metropolis-Hastings)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Slice Sampling  (univariate, stepping-out & shrinking)
# ---------------------------------------------------------------------------
def slice_sampling(log_pdf, x0, n_samples, w=1.0, rng=None):
    """
    Jednoduchý univariátní slice sampler (stepping-out).

    Parametry
    ---------
    log_pdf : callable  – logaritmus cílové hustoty (až na konstantu)
    x0      : float     – počáteční bod
    n_samples : int     – počet požadovaných vzorků
    w       : float     – šířka okna pro stepping-out
    rng     : np.random.Generator | None

    Vrací
    -----
    samples : np.ndarray, shape (n_samples,)
    n_evals : int – celkový počet vyhodnocení log_pdf
    """
    if rng is None:
        rng = np.random.default_rng()

    samples = np.empty(n_samples)
    x = float(x0)
    n_evals = 0

    for i in range(n_samples):
        log_y = log_pdf(x) + np.log(rng.uniform())  # horizontální řez
        n_evals += 1

        # --- stepping out ---
        L = x - w * rng.uniform()
        R = L + w
        while log_pdf(L) > log_y:
            L -= w
            n_evals += 1
        n_evals += 1  # poslední neúspěšný test
        while log_pdf(R) > log_y:
            R += w
            n_evals += 1
        n_evals += 1  # poslední neúspěšný test

        # --- shrinking ---
        while True:
            x_new = rng.uniform(L, R)
            n_evals += 1
            if log_pdf(x_new) > log_y:
                break
            if x_new < x:
                L = x_new
            else:
                R = x_new

        x = x_new
        samples[i] = x

    return samples, n_evals


def slice_sampling_one_step(log_pdf, x_curr, w, rng):
    """
    Provede jeden krok slice samplingu. Vrací nový bod a mezivýsledky
    pro vizualizaci (y_slice, L, R).
    """
    log_y = log_pdf(x_curr) + np.log(rng.uniform())

    # stepping out
    L = x_curr - w * rng.uniform()
    R = L + w
    while log_pdf(L) > log_y:
        L -= w
    while log_pdf(R) > log_y:
        R += w

    # shrinking
    L_orig, R_orig = L, R
    while True:
        x_new = rng.uniform(L, R)
        if log_pdf(x_new) > log_y:
            break
        if x_new < x_curr:
            L = x_new
        else:
            R = x_new

    return x_new, np.exp(log_y), L_orig, R_orig


# ---------------------------------------------------------------------------
# Random Walk Metropolis (RWM)
# ---------------------------------------------------------------------------
def metropolis_hastings(log_pdf, x0, n_samples, proposal_std=1.0, rng=None):
    """
    Jednoduchý Random Walk Metropolis (RWM) s normálním návrhovým rozdělením.

    Parametry
    ---------
    log_pdf      : callable – logaritmus cílové hustoty (až na konstantu)
    x0           : float    – počáteční bod
    n_samples    : int      – počet požadovaných vzorků
    proposal_std : float    – směrodatná odchylka normálního proposal
    rng          : np.random.Generator | None

    Vrací
    -----
    samples : np.ndarray, shape (n_samples,)
    accept_rate : float – podíl přijatých návrhů
    n_evals : int – celkový počet vyhodnocení log_pdf
    """
    if rng is None:
        rng = np.random.default_rng()

    samples = np.empty(n_samples)
    x = float(x0)
    log_px = log_pdf(x)
    accepted = 0
    n_evals = 1  # počáteční vyhodnocení

    for i in range(n_samples):
        x_prop = x + proposal_std * rng.normal()
        log_px_prop = log_pdf(x_prop)
        n_evals += 1

        if np.log(rng.uniform()) < log_px_prop - log_px:
            x = x_prop
            log_px = log_px_prop
            accepted += 1

        samples[i] = x

    return samples, accepted / n_samples, n_evals
