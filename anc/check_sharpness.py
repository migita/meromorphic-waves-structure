#!/usr/bin/env python3
"""Independent direct-curve proof of the structural sharpness example.

Only elementary polynomial derivations and Weierstrass Laurent identities
are used. No coefficient-classification or reconstruction module is
imported. The only file written is sharpness_checks.json beside this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as s


x, y, g, r = s.symbols("x y g r")


def reduce_quadratic(expression, variable, square):
    return s.Poly(s.expand(expression), variable).rem(
        s.Poly(variable**2 - square, variable)
    ).as_expr().expand()


def iterate(derivative, expression, count):
    for _ in range(count):
        expression = derivative(expression)
    return expression


def wp_laurent(g2, g3, top=10):
    """Coefficients of wp from wp''=6wp^2-g2/2 and its first integral."""
    result = {-2: s.S.One, 2: s.Rational(g2, 20), 4: s.Rational(g3, 28)}
    for exponent in range(6, top + 1, 2):
        lower = sum(coefficient * result.get(exponent - 2 - j, s.S.Zero)
                    for j, coefficient in result.items() if j >= 2)
        result[exponent] = s.factor(6 * lower / (exponent * (exponent - 1) - 12))
    return result


def main():
    # u_+=wp(z;-6,0), on y^2=4x^3+6x.
    cubic = 4 * x**3 + 6 * x

    def D_plus(expression):
        return reduce_quadratic(s.diff(expression, x) * y
                                + s.diff(expression, y) * (6 * x**2 + 3), y, cubic)

    fourth_plus = iterate(D_plus, x, 4)
    assert s.expand(fourth_plus - 120 * x**3 - 108 * x) == 0
    plus_residual = s.expand(fourth_plus - 108 * x - 120 * x**3)
    assert plus_residual == 0

    # g^2=wp(z;4,0), r=g'. Its quartic is r^2=g^4-1 and Dr=2g^3.
    quartic = g**4 - 1

    def D_minus(expression):
        return reduce_quadratic(s.diff(expression, g) * r
                                + s.diff(expression, r) * 2 * g**3, r, quartic)

    u_minus = -r
    fourth_minus = iterate(D_minus, u_minus, 4)
    square_minus = reduce_quadratic(u_minus**2, r, quartic)
    assert s.expand(square_minus - g**4 + 1) == 0
    assert reduce_quadratic(fourth_minus - (120 * g**4 - 12) * u_minus, r, quartic) == 0
    minus_residual = reduce_quadratic(fourth_minus - 108 * u_minus - 120 * u_minus**3, r, quartic)
    assert minus_residual == 0

    # The quartic derivation maps to the stated Weierstrass curve, so the
    # computation checks precisely the displayed square-root profile.
    X, Y = g**2, 2 * g * r
    assert reduce_quadratic(Y**2 - 4 * X**3 + 4 * X, r, quartic) == 0
    assert reduce_quadratic(D_minus(X) - Y, r, quartic) == 0
    assert reduce_quadratic(D_minus(Y) - 6 * X**2 + 2, r, quartic) == 0
    assert (-6)**3 != 0 and 4**3 != 0

    plus = wp_laurent(-6, 0)
    wp_minus = wp_laurent(4, 0)
    # The residue-one root g=z^-1 sum d_j z^j is obtained from g^2=wp,
    # not from the fourth-order equation or its candidate polynomial.
    normalized_root = [s.S.One]
    for j in range(1, 13):
        target = wp_minus.get(j - 2, s.S.Zero)
        known = sum(normalized_root[i] * normalized_root[j - i] for i in range(1, j))
        normalized_root.append(s.factor((target - known) / 2))
    minus = {j - 2: s.factor(-(j - 1) * coefficient)
             for j, coefficient in enumerate(normalized_root)}
    for j in range(8):
        assert plus.get(j - 2, 0) == minus.get(j - 2, 0)
    h_plus, h_minus = plus[6], minus[6]
    assert h_plus == s.Rational(3, 100)
    assert h_minus == -s.Rational(7, 600)
    assert h_plus != h_minus
    assert plus[-2] == minus[-2] == 1
    assert -s.Integer(-360) / 3 == 120

    directory = Path(__file__).resolve().parent
    report = {
        "operator": "D^4-108",
        "physical_normalization": "v=c*u, c^2=-360",
        "normalized_equation": "u''''-108*u-120*u^3=0",
        "plus": {
            "curve": "y^2=4*x^3+6*x",
            "profile": "u=x=wp(z;-6,0)",
            "fourth_derivative": str(fourth_plus),
            "full_ODE_residual": str(plus_residual),
            "Laurent_coefficients_by_exponent": {str(j): str(plus.get(j, 0)) for j in range(-2, 10)},
            "h": str(h_plus),
        },
        "minus": {
            "quartic": "r^2=g^4-1, Dg=r, Dr=2*g^3",
            "profile": "u=-r=-D sqrt(wp(z;4,0))",
            "Weierstrass_curve_map": "x=g^2, y=2*g*r; y^2=4*x^3-4*x",
            "fourth_derivative": str(fourth_minus),
            "full_ODE_residual": str(minus_residual),
            "Laurent_coefficients_by_exponent": {str(j): str(minus.get(j, 0)) for j in range(-2, 10)},
            "h": str(h_minus),
        },
        "distinct_normalized_germs": True,
        "status": "all independent direct-curve and Laurent checks passed",
    }
    output = directory / "sharpness_checks.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "h_plus": str(h_plus), "h_minus": str(h_minus),
                      "both_full_ODE_residuals": 0,
                      "certificate": output.name}, indent=2))


if __name__ == "__main__":
    main()
