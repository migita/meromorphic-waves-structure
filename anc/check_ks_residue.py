#!/usr/bin/env python3
"""Exact principal-part and normalization check for the KS residue example.

The coefficients are derived by substitution into the full physical ODE.
Only the necessary elliptic residue condition is asserted. The sole output
file is ks_residue_checks.json beside this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as s


z, B, mu, a0, nu = s.symbols("z B mu a0 nu")
b1, b2, b3 = s.symbols("b1 b2 b3")
L, C2, R, C0 = s.symbols("L C2 R C0")


def coefficient(expression, exponent):
    return s.expand(expression).coeff(z, exponent)


def main():
    # Monic normalized equation: P(D)u+60u^2=0, since v=120u.
    u = z**-3 + b1 * z**-2 + b2 * z**-1 + b3
    normalized_residual = s.expand(s.diff(u, z, 3) + B * s.diff(u, z, 2)
                                  + mu * s.diff(u, z) + a0 * u + 60 * u**2)
    assert coefficient(normalized_residual, -6) == 0
    rows = [coefficient(normalized_residual, exponent) for exponent in (-5, -4, -3)]
    solution = {}
    for unknown, row in zip((b1, b2, b3), rows):
        current = s.expand(row.subs(solution))
        assert s.degree(current, unknown) == 1
        solution[unknown] = s.factor(s.solve(current, unknown)[0])
        assert s.expand(current.subs(unknown, solution[unknown])) == 0
    assert solution[b1] == -B / 8
    assert s.factor(solution[b2] - (16 * mu - B**2) / 608) == 0
    assert not solution[b1].has(a0) and not solution[b2].has(a0)
    assert s.diff(solution[b3], a0) == -s.Rational(1, 120)
    for exponent in (-6, -5, -4, -3):
        assert s.factor(coefficient(normalized_residual.subs(solution), exponent)) == 0

    # Independently derive the nonmonic physical Laurent coefficients.
    physical_v = L * z**-3 + C2 * z**-2 + R * z**-1 + C0
    physical_residual = s.expand(nu * s.diff(physical_v, z, 3)
                                + B * s.diff(physical_v, z, 2)
                                + mu * s.diff(physical_v, z) + a0 * physical_v
                                + physical_v**2 / 2)
    balance = s.factor(coefficient(physical_residual, -6))
    nonzero_leading = [root for root in s.solve(balance, L) if root != 0]
    assert nonzero_leading == [120 * nu]
    physical_solution = {L: nonzero_leading[0]}
    for unknown, exponent in ((C2, -5), (R, -4), (C0, -3)):
        row = s.factor(coefficient(physical_residual, exponent).subs(physical_solution))
        physical_solution[unknown] = s.factor(s.solve(row, unknown)[0])
        assert s.factor(row.subs(unknown, physical_solution[unknown])) == 0
    assert physical_solution[C2] == -15 * B
    residue = s.factor(physical_solution[R])
    assert s.factor(residue - 15 * (16 * mu * nu - B**2) / (76 * nu)) == 0
    scaled_monic_residue = 120 * nu * solution[b2].subs({B: B / nu, mu: mu / nu}, simultaneous=True)
    assert s.factor(residue - scaled_monic_residue) == 0
    assert not residue.has(a0)

    # A constant integration term is removed by an equilibrium shift.
    A, shift, w, w1, w2, w3 = s.symbols("A shift w w1 w2 w3")
    translated = (nu * w3 + B * w2 + mu * w1
                  + a0 * (w + shift) + (w + shift)**2 / 2 + A)
    shifted_equation = nu * w3 + B * w2 + mu * w1 + (a0 + shift) * w + w**2 / 2
    equilibrium_polynomial = shift**2 / 2 + a0 * shift + A
    assert s.expand(translated - shifted_equation - equilibrium_polynomial) == 0

    directory = Path(__file__).resolve().parent
    report = {
        "monic": {
            "physical_normalization": "v=120u",
            "normalized_equation": "u'''+B*u''+mu*u'+a0*u+60*u^2=0",
            "derived_rows": [str(row) for row in rows],
            "derived_coefficients": {str(unknown): str(value) for unknown, value in solution.items()},
            "elliptic_residue": str(solution[b2]),
            "necessary_condition": "B^2=16mu",
            "independent_of_a0": True,
        },
        "nonmonic": {
            "assumption": "nu!=0",
            "leading_balance": str(balance),
            "physical_normalization": "v=120nu*u",
            "physical_principal_part": {"z^-3": str(physical_solution[L]),
                                        "z^-2": str(physical_solution[C2]), "z^-1": str(residue)},
            "necessary_condition": "B^2=16mu*nu",
        },
        "constant_shift": {
            "substitution": "v=w+s",
            "equilibrium_condition": "s^2/2+a0*s+A=0",
            "new_linear_coefficient": "a0+s",
            "exact_identity_remainder": 0,
        },
        "primary_source_checked": {
            "bibliography_key": "Eremenko2006",
            "author": "Alexandre Eremenko",
            "title": "Meromorphic traveling wave solutions of the Kuramoto–Sivashinsky equation",
            "full_text": "https://arxiv.org/pdf/nlin/0504053",
            "checked_on": "2026-09-23",
            "locations": ["Equation (1), PDF page 2: integrated KS equation with constant A",
                          "Equation (3), PDF page 2: physical Laurent principal part",
                          "Theorem 1(i), PDF page 3: necessary condition b^2=16mu*nu",
                          "Proof of Theorem 1, Case 2, PDF pages 4–5: single-pole residue argument"],
        },
        "scope": "Necessary elliptic obstruction only; no sufficiency or new KS classification is asserted",
        "status": "all exact recurrence, normalization, and shift checks passed",
    }
    output = directory / "ks_residue_checks.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "monic_residue": str(solution[b2]),
                      "physical_residue": str(residue), "certificate": output.name}, indent=2))


if __name__ == "__main__":
    main()
