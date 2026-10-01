#!/usr/bin/env python3
"""A first-derivative term for the quadratic nonlinearity (n = 2).

Remark after the corollary "A first-derivative term excludes waves":
for n = 2 and P = P_e + a_1 D with P_e even of degree p, a wave that is
rational in an exponential has a_1 = 0 when p = 4, 6, 8, 10, while for p = 2
waves with a_1 != 0 exist (the front of the Korteweg-de Vries-Burgers
equation).

By the theorem on the one-pole forms, such a wave is, at rate k = 1 and with
the pole at 0,

    u = beta + sum_{j=0}^{p-1} d_j D^j G,   G' = G (1 - G),
    d_{p-1} = (-1)^(p-1)/(p-1)!   (normalization u = z^(-p) + ...).

Scaling reduces any rate to k = 1 and keeps the form P_e + a_1 D. All
coefficients of P_e, beta and d_0, ..., d_{p-2} are unknowns. The script
shows that the polynomial system of P(D)u = K u^2, K = (p)_p, together with
1 - t*a_1 generates the unit ideal (so a_1 = 0 on every solution), and that
for p = 2 it does not. For p = 4, 6 it also checks that solutions with
a_1 = 0 exist, so that the system itself is consistent.
"""

import time

import sympy as sp

G, t, beta, a1 = sp.symbols("G t beta a1")


def D(f):
    return sp.expand(sp.diff(f, G) * G * (1 - G))


def a1_vanishes(p, control):
    start = time.time()
    even = {i: sp.Symbol(f"a{i}") for i in range(0, p, 2)}
    d = {j: sp.Symbol(f"d{j}") for j in range(p - 1)}
    g = [G]
    for _ in range(1, p):
        g.append(D(g[-1]))
    K = sp.rf(p, p)
    u = beta + sum(d[j] * g[j] for j in d) + sp.Rational((-1) ** (p - 1), sp.factorial(p - 1)) * g[p - 1]
    Du = [u]
    for _ in range(p):
        Du.append(D(Du[-1]))
    residual = sp.expand(Du[p] + sum(even[i] * Du[i] for i in even) + a1 * Du[1] - K * u ** 2)
    equations = [residual.coeff(G, k) for k in range(2 * p + 1)]
    assert equations[2 * p] == 0                       # the leading balance
    equations = [e for e in equations[:2 * p] if e != 0]
    gens = (t, a1) + tuple(even.values()) + (beta,) + tuple(d.values())
    forced = sp.groebner(equations + [1 - t * a1], *gens, order="grevlex").exprs == [1]
    line = f"p = {p:2d}: {len(gens) - 1} unknowns, {len(equations)} equations; a_1 = 0 on every solution: {forced}"
    if control:
        consistent = sp.groebner(equations + [a1], *gens[1:], order="grevlex").exprs != [1]
        assert consistent
        line += "; solutions with a_1 = 0 exist"
    print(line, f"({time.time() - start:.0f} s)", flush=True)
    return forced


def main():
    assert a1_vanishes(2, True) is False              # Korteweg-de Vries-Burgers front
    # the front itself: u = G^2 at rate 1 solves u'' - 5u' + 6u = 6 u^2  (a_0 = 6 a_1^2/25)
    u = G ** 2
    assert sp.expand(D(D(u)) - 5 * D(u) + 6 * u - 6 * u ** 2) == 0
    for p, control in ((4, True), (6, True), (8, False), (10, False)):
        assert a1_vanishes(p, control)
    print("PASS n = 2: a_1 = 0 for every wave rational in an exponential when p = 4, 6, 8, 10; not for p = 2.")


if __name__ == "__main__":
    main()
