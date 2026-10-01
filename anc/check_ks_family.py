#!/usr/bin/env python3
"""Exact checks for the Kuramoto--Sivashinsky example.

Requires Python 3 and SymPy. Run from this folder:
    python3 check_ks_family.py

Checks:
1. An infinite fixed-speed family of elliptic travelling waves for one KS
   PDE, whose integration constant varies.
2. The equilibrium shift that removes a quadratic integration constant
   changes the constant coefficient of the operator.

No files are written.
"""

import sympy as sp


def check_fixed_speed_counterexample():
    X, Y, g3 = sp.symbols("X Y g3")
    g2 = sp.Rational(1, 12)
    curve = Y**2 - 4 * X**3 + g2 * X + g3

    def reduce(expr):
        return sp.rem(sp.expand(expr), curve, Y)

    def D(expr):
        return reduce(
            Y * sp.diff(expr, X)
            + (6 * X**2 - g2 / 2) * sp.diff(expr, Y)
        )

    # The derivation preserves the Weierstrass relation.
    assert D(curve) == 0
    w = -60 * Y - 60 * X - 1
    w1 = D(w)
    w2 = D(w1)
    w3 = D(w2)
    w4 = D(w3)
    integrated = reduce(w3 + 4 * w2 + w1 + w**2 / 2)
    assert integrated == 13 - 1080 * g3, integrated
    pde_residual = reduce(w * w1 + w2 + 4 * w3 + w4)
    assert pde_residual == 0, pde_residual
    assert D(integrated) == 0
    assert sp.diff(integrated, g3) == -1080

    # Two explicit admissible lattice parameters already have different C;
    # the polynomial identity above holds for all nondegenerate parameters.
    discriminant = g2**3 - 27 * g3**2
    assert discriminant.subs(g3, 0) != 0
    assert discriminant.subs(g3, 1) != 0
    assert integrated.subs(g3, 0) != integrated.subs(g3, 1)

    print("KS integrated residual:", integrated)
    print("KS stationary PDE residual:", pde_residual)
    print("Fixed-speed counterexample: exact identity verified.")


def check_quadratic_equilibrium_shift():
    v0, v1, v2, v3 = sp.symbols("v v1 v2 v3")
    a1, a2, a3, speed, C, w0 = sp.symbols("a1 a2 a3 speed C w0")
    derivative_terms = a3 * v3 + a2 * v2 + a1 * v1
    shifted = (
        derivative_terms - speed * (v0 + w0) + (v0 + w0)**2 / 2 - C
    )
    target = derivative_terms + (w0 - speed) * v0 + v0**2 / 2
    equilibrium_condition = w0**2 / 2 - speed * w0 - C
    assert sp.expand(shifted - target - equilibrium_condition) == 0

    # On the equilibrium condition, the square of the new linear
    # coefficient depends on C even when the speed is fixed.
    new_linear_coefficient = w0 - speed
    assert sp.expand(
        new_linear_coefficient**2 - (speed**2 + 2 * C)
        - 2 * equilibrium_condition
    ) == 0
    print("Equilibrium shift: exact identity verified.")
    print("Before monic normalization, the new D^0 coefficient is w0 - speed.")


def main():
    check_fixed_speed_counterexample()
    check_quadratic_equilibrium_shift()
    print("All Kuramoto-Sivashinsky example checks passed.")


if __name__ == "__main__":
    main()
