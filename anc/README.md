# Ancillary files

Exact computer checks for the paper *Meromorphic travelling waves for
pure-power equations: one pole orbit and a two-wave bound* (A. Migita).
The theorems are proved in the text; these scripts check displayed
identities and the examples. One statement of the paper rests on a
computer elimination: the proposition on a first-derivative term for
n = 2 and p ≤ 10 (`check_drift_n2.py`). The numbers quoted in the
examples without derivation (the Swift–Hohenberg table, the
compatibility number F_J, the coefficient A_5 for D⁶ + a₀) are finite
computations that can be done by hand; they are checked by
`check_examples.py`, `check_more_examples.py` and
`check_linear_terms.py`.

Requirements: Python 3 and SymPy (tested with Python 3.12.3 and SymPy 1.12,
and with Python 3.8.10 and SymPy 1.13.3).
From this folder, run

```sh
python3 run_all.py
```

It runs every script below in turn, reports each exit status, and ends
with a nonzero exit code if any script fails (about one minute in total).
Add `--verbose` to see the output of each script.

The scripts were written for an earlier arrangement of the paper, and
some of their comments and messages use its words: "selector" for the
equations for the free coefficient of Section 6 ("affine selector":
Proposition 6.4; "quadratic selector": Proposition 6.6), "seed" for the
expression H of Lemma 6.3, "bracket" for the combination (6.1), and
"sharpness" for the equation with exactly two waves. The table gives
the section of the present text for each script.

| Script | Checks | Section of the paper |
|---|---|---|
| `check_bracket.py` | The rule for principal-part polynomials under differentiation, the terms in h and h² in the lemma on the expression H and in Propositions 6.4 and 6.6, and the choice of the function T in the proof of Proposition 6.6; recurrence checks with a first-derivative term, a vanishing principal-part polynomial, simple poles and (n,q) = (2,2). Writes `check_bracket_certificate.json`. | 6, Equations for the free coefficient |
| `check_bracket_audit.py` | An independent implementation of the same checks. Writes `check_bracket_audit_certificate.json`. | 6, Equations for the free coefficient |
| `check_sharpness.py` | The equation with exactly two waves: differentiation on both curves, exact reduction of the ODE, and the two free coefficients. Writes `sharpness_checks.json`. | 2.6, Step 6: the two waves |
| `check_ks_residue.py` | Kuramoto–Sivashinsky: the Laurent coefficients, the normalization for ν ≠ 1, and the shift that removes the integration constant. Writes `ks_residue_checks.json`. | 8.2, The residue condition |
| `check_ks_family.py` | Kuramoto–Sivashinsky: the family of elliptic waves at speed s = 0. | 8.2, The residue condition |
| `check_examples.py` | The function `swift_hohenberg_control`: the remainders, the polynomial conditions of the Swift–Hohenberg table, and the physical normalization. The script also runs two further exact controls (`plus_nodal_control`, `minus_nodal_control`) that the paper does not use. | 9.3, The Swift–Hohenberg equation with dispersion |
| `check_text_identities.py` | The form D^j g_r = g_r Q_j(G), the effect of k ↦ −k on g_r, the reductions in the Kuramoto–Sivashinsky section and in the section on a first-derivative term, the scaling of the two-wave example, the end-value identity on the Swift–Hohenberg waves, the σ = 0 entries of the list of dispersion coefficients, the degree-two family, and, independently of the scripts above, the numbers ψ, κ, Δ₁, Δ₂ of Section 6. Writes `text_identities_checks.json`. | Several sections |
| `check_linear_terms.py` | The terms of the combination (6.1) that are linear in the free coefficient (pair u, H_n): homogeneity of the coefficients A_i; the residue formulas for A_0 and A_1; A_1 = 0 for even operators and the identities used in its proof; A_0 = c_0 a_{p-1} and A_1 = c_1 a_{p-1}^2; the explicit family of the operators P_ε (closed forms (C.4) and the values of A_0, A_1); the formula (C.5) for the first odd coefficient c_{j-1}, its agreement with the series and its sign for n = 2, 3 and p ≤ 120; the conclusion "one wave unless P is even" on the Swift–Hohenberg waves with σ ≠ 0; and the example of the even operator D⁶ + a₀ with n = 3, which has one wave. Imports `series_tools.py`. Writes `linear_terms_checks.json`. | 7.3, Operators that are not even; Appendix C |
| `check_more_examples.py` | The expression Ψ of Step 4 (n = 3, P = D⁴ − s); the example of Appendix D for the lemma "Bounded on the poles"; the two waves for n = 2, P = D⁴ − s (the coefficients ℓ_1, …, ℓ_4, the function Λ_4 = ℘² − g₂/10 also for the degenerate ℘, the conditions g₃ = 0 and g₂² = s²/23184, and the quadratic equation at the two free coefficients); the compatibility number F_J for n = 3, p = 4; for Swift–Hohenberg, the residue R(P), the twenty parameter values of the table and their symmetry ξ ↦ −ξ, the rates of the twisted waves, the two waves at σ² = 392/11, and the agreement of s and α with formulas (28)–(31) of Kudryashov–Sinelshchikov (2012); the two constants C = −8, −18 of the Kuramoto–Sivashinsky section; the wave c·G for n = 3, p = 2 with a first-derivative term; the two waves of D⁶ + a₀ for n = 2 (g₂ = 0, g₃² = a₀²/3643833600) and the quadratic equation at their free coefficients. Imports `series_tools.py`. Writes `more_examples_checks.json`. | 2.4, 8.2, 8.3, 9.2, 9.3, Appendix D |
| `check_drift_n2.py` | n = 2, P = P_e + a_1 D: exact elimination showing a_1 = 0 on every wave rational in an exponential for p = 4, 6, 8, 10, and that the statement does not hold for p = 2. | 9.2, Proposition 9.5 |
| `sh_dispersion.py` | Swift–Hohenberg: prints σ², α and κ² for all waves rational in an exponential (a computation, without assertions). | 9.3 |
| `tables_q2.py` | Prints the explicit waves for pole order q = 2 in every sector (n = 3, 4 by default; a computation, without assertions). Imports `sectors.py`. | 9.3 |
| `network_32_explicit.py` | Solves for and prints all waves of the untwisted Swift–Hohenberg sector with arbitrary background, in the parameter b = β + ξ/2. | 9.3 |

`series_tools.py` is the series module of `check_linear_terms.py` and
`check_more_examples.py`; it shares no code with `check_bracket.py` and
`check_bracket_audit.py`. The Kuramoto–Sivashinsky scripts write the
integration constant as `A`; in the paper it is `C = −A`.

`sectors.py` is a module used by `tables_q2.py`, which needs only its
function `count` (pure SymPy). Running `sectors.py` on its own counts
solutions with the external program msolve; that is not part of these
checks.

The JSON files in this folder are the certificates written by the last
run of `python3 run_all.py`.
