#!/usr/bin/env python3
"""Further examples of the paper, checked exactly.

1. Section "An example" (n = 3, P = D^4 - s): the series f(z;h), the
   expression Psi, its value at a pole and its z^8 coefficient; Phi = -Psi'.
2. The example after the lemma "Bounded on the poles":
   v = -12 k^2 e^{-2kz} wp(e^{-kz} - zeta0; 0, g3) and (v' + 2kv)^2 + v^3/3.
3. Section "Equations with exactly two waves", n = 2, P = D^4 - s: the
   coefficients l_1, ..., l_4; Lambda_4 = wp^2 - g2/10, also for the
   degenerate wp; the substitution u = wp^2 - g2/10 - s/1680, which gives
   g3 = 0 and g2^2 = s^2/23184; and the quadratic selector at the two free
   coefficients.
4. The compatibility number F_J for n = 3, P = D^4 + a2 D^2 + a1 D + a0.
5. Swift-Hohenberg with dispersion: the residue R(P); the twenty solutions of
   the table (5 twisted, 15 untwisted, a_2 != 0); the symmetry xi -> -xi;
   R(P) = 0 on the twisted waves; the rates of the twisted waves (real for
   xi = 0, +-1, imaginary for xi^2 = -17/7); the two waves at sigma^2 = 392/11;
   agreement of (s, alpha) of the untwisted waves with formulas (28)-(31) of
   Kudryashov-Sinelshchikov (2012) at zero quadratic coefficient.
6. Kuramoto-Sivashinsky example: for the two excluded constants C = -8 and
   C = -18 the formula of the family gives a wave that is rational in an
   exponential.
7. Remark on a first-derivative term, degree two: v = c G with c^2 = -6 solves
   v'' - 3k v' + 2k^2 v + v^3/3 = 0.
8. Section "Scope": n = 2, P = D^6 + a0 has exactly two waves,
   u = wp^3 - 3 g2 wp/20 - 3 g3/28 + a0/665280 with g2 = 0 and
   g3^2 = a0^2/3643833600; their free coefficients are the two roots of the
   quadratic selector.

Writes more_examples_checks.json.
"""

import json
from fractions import Fraction as Fr
from pathlib import Path

import sympy as sp
from sympy import QQ

from series_tools import (Family, make_ring, phi_u_Hn, s_add, s_diff, s_mul, s_scale,
                          to_fraction)


def example_selector():
    R, h, extra = make_ring(["s"])
    s = extra["s"]
    fam = Family(3, 2, R, h, [-s, R(0), R(0), R(0)], 22)
    b = fam.b
    assert fam.F_J == 0 and b[4] == -s * QQ(1, 360) and b[8] == h
    assert all(b[j] == 0 for j in range(1, 21) if j % 4)
    f = fam.series()
    f1, f2 = s_diff(f, 1), s_diff(f, 2)
    Psi = s_add(s_add(s_mul(s_mul(f, f), f2), s_scale(s_mul(f, s_mul(f1, f1)), -QQ(3, 2))), s_scale(f2, s * QQ(1, 108)))
    Psi = {e: c for e, c in Psi.items() if e <= 10}
    assert all(e >= 0 for e in Psi)                                       # no pole
    H, S = sp.symbols("h s")
    assert sp.expand(Psi[0].as_expr() - (72 * H + sp.Rational(7, 97200) * S ** 2)) == 0
    assert all(e in (0, 8) for e in Psi)                                  # nothing at z^2, z^4, z^6, z^10
    target = sp.Rational(7, 365316480000) * (388800 * H - S ** 2) * (6998400 * H + 7 * S ** 2)
    assert sp.expand(Psi[8].as_expr() - target) == 0
    phi = phi_u_Hn(fam, 9)                                                # bracket of (u, H_3)
    assert all(phi.get(e, R(0)) == -(e + 1) * Psi.get(e + 1, R(0)) for e in range(9))
    roots = [sp.Rational(108 ** 2, 388800), -sp.Rational(7 * 108 ** 2, 6998400)]
    assert roots == [sp.Rational(3, 100), -sp.Rational(7, 600)]
    return {"Psi_at_pole": "72*h + 7*s^2/97200", "z8_coefficient": str(sp.factor(Psi[8].as_expr())),
            "roots_at_s_108": [str(r) for r in roots], "Phi_equals_minus_Psi_prime": True}


def example_hypothesis_needed():
    k, g3, zeta = sp.symbols("k g3 zeta")
    W = sp.Function("W")(zeta)                       # W = wp(zeta - zeta0; 0, g3)
    Dz = lambda F: -k * zeta * sp.diff(F, zeta)      # zeta = e^{-kz}
    v = -12 * k ** 2 * zeta ** 2 * W
    residual = sp.expand(Dz(Dz(v)) + 5 * k * Dz(v) + 6 * k ** 2 * v + v ** 2 / 2)
    residual = residual.subs(sp.Derivative(W, (zeta, 2)), 6 * W ** 2)     # wp'' = 6 wp^2
    assert sp.simplify(residual) == 0
    w1 = sp.Symbol("w1")
    expr = sp.expand(((Dz(v) + 2 * k * v) ** 2 + v ** 3 / 3).subs(sp.Derivative(W, zeta), w1))
    expr = sp.expand(expr.subs(w1 ** 2, 4 * W ** 3 - g3))                 # wp'^2 = 4 wp^3 - g3
    assert sp.expand(expr + 144 * g3 * k ** 6 * zeta ** 6) == 0
    return {"(v'+2kv)^2 + v^3/3": "-144*g3*k^6*exp(-6*k*z)"}


def wp_series(g2, g3, terms):
    """Coefficients c_k of wp = z^-2 + sum_{k>=2} c_k z^(2k-2)."""
    c = {2: g2 / 20, 3: g3 / 28}
    for k in range(4, terms + 1):
        c[k] = sp.Rational(3, (2 * k + 1) * (k - 3)) * sum(c[i] * c[k - i] for i in range(2, k - 1))
    return c


def example_two_waves_n2():
    x, g2, g3, s = sp.symbols("x g2 g3 s")

    def D2(F):                                        # second z-derivative of a polynomial in x = wp
        return sp.expand(sp.diff(F, x, 2) * (4 * x ** 3 - g2 * x - g3) + sp.diff(F, x) * (6 * x ** 2 - g2 / 2))

    # the first Laurent coefficients: u = z^-4 - s/1680 + O(z^4), so l_1 = l_2 = l_3 = 0, l_4 = -s/1680 (m = 1)
    R, h, extra = make_ring(["s"])
    fam = Family(2, 4, R, h, [-extra["s"], R(0), R(0), R(0)], 30)
    assert fam.F_J == 0
    assert fam.b[1] == fam.b[2] == fam.b[3] == 0 and fam.b[4] == -extra["s"] * QQ(1, 1680)
    # Lambda_4 = (wp'' - [z^0] wp'')/6 = wp^2 - g2/10, since wp'' = 6 wp^2 - g2/2 and wp^2 = z^-4 + g2/10 + O(z^2)
    c = wp_series(g2, g3, 4)
    assert sp.expand(2 * c[2] - g2 / 10) == 0                                   # [z^0] wp^2
    lam4 = sp.expand((D2(x) - (6 * g2 / 10 - g2 / 2)) / 6)
    assert sp.expand(lam4 - (x ** 2 - g2 / 10)) == 0
    # the degenerate wp = k^2/12 - DG satisfies wp'^2 = 4 wp^3 - g2 wp - g3 with g2 = k^4/12, g3 = -k^6/216
    G, k = sp.symbols("G k")
    DG = lambda F: sp.expand(sp.diff(F, G) * G * (k - G))
    wpd = k ** 2 / 12 - DG(G)
    assert sp.expand(DG(wpd) ** 2 - (4 * wpd ** 3 - k ** 4 / 12 * wpd + k ** 6 / 216)) == 0
    assert sp.expand((k ** 4 / 12) ** 3 - 27 * (k ** 6 / 216) ** 2) == 0
    # substitution of u = wp^2 - g2/10 - s/1680
    u = x ** 2 - g2 / 10 - s / 1680
    residual = sp.expand(D2(D2(u)) - s * u - 840 * u ** 2)
    assert sp.expand(residual - (-120 * g3 * x - sp.Rational(69, 10) * (g2 ** 2 - s ** 2 / 23184))) == 0
    assert 12 ** 2 * 161 == 23184
    # the same conditions in the form used in the referee report: u = wp^2 + gamma
    gam = -g2 / 10 - s / 1680
    quad = 690 * gam ** 2 + sp.Rational(23, 28) * s * gam - s ** 2 / 18816
    assert sp.expand(quad - sp.Rational(69, 10) * (g2 ** 2 - s ** 2 / 23184)) == 0
    # free coefficients h = b_12 of the two waves, and the quadratic selector Q_P
    Q = phi_u_Hn(fam, 11)[11].as_expr()               # A_0 = 0 here, so Q_P = [z^(J-1)] phi, J = 12
    Hs, Ss = sp.symbols("h s")
    assert sp.Poly(Q, Hs).degree() == 2
    checked = []
    for sign in (1, -1):
        g2v = sign / (12 * sp.sqrt(161))              # s = 1
        c = wp_series(g2v, 0, 6)
        wp = {-2: sp.Integer(1)}
        wp.update({2 * j - 2: c[j] for j in c})
        sq = {}
        for e1, c1 in wp.items():
            for e2, c2 in wp.items():
                if e1 + e2 <= 8:
                    sq[e1 + e2] = sq.get(e1 + e2, 0) + c1 * c2
        assert sp.simplify(sq[0] - g2v / 10) == 0 and sp.simplify(sq[4]) == sp.simplify(fam.b[8].as_expr().subs(Ss, 1))
        hval = sp.nsimplify(sp.expand(sq[8]))         # coefficient of z^8 = z^(J-q)
        assert sp.simplify(Q.subs({Ss: 1, Hs: hval})) == 0
        checked.append(str(sp.nsimplify(sp.simplify(hval))))
    assert len(set(checked)) == 2
    return {"u": "wp^2 - g2/10 - s/1680", "g3": "0", "g2^2": "s^2/23184",
            "free_coefficients_at_s_1": checked, "selector_vanishes_at_both": True}


def example_compatibility():
    R, h, extra = make_ring(["a0", "a1", "a2"])
    fam = Family(3, 2, R, h, [extra["a0"], extra["a1"], extra["a2"], R(0)], 8)
    a1, a2 = sp.symbols("a1 a2")
    assert sp.expand(fam.F_J.as_expr() + a1 ** 2 * a2 / 900) == 0
    return {"F_J for n=3, P=D^4+a2 D^2+a1 D+a0": "-a1^2*a2/900"}


def example_swift_hohenberg():
    R, h, extra = make_ring(["sigma", "s", "alpha"])
    fam = Family(3, 2, R, h, [-extra["alpha"], -extra["s"], R(2), -extra["sigma"]], 4)
    b = {j: c for j, c in enumerate(fam.b)}
    residue = s_mul(b, b)[3].as_expr()                # l_3 = [z^3] (sum b_j z^j)^2
    sg, s = sp.symbols("sigma s")
    assert sp.expand(residue - (1372 * s + 39 * sg ** 3 - 392 * sg) / 123480) == 0
    speed = lambda sigma: sigma * (392 - 39 * sigma ** 2) / 1372

    xi, be = sp.symbols("xi beta")
    tw = dict(a3=-14 * xi, a2=46 * xi ** 2 + 10, a1=-2 * xi * (7 * xi ** 2 + 10), a0=-14 * xi ** 4 - 80 * xi ** 2 - 11)
    un = dict(a3=-14 * xi, a2=46 * xi ** 2 - 120 * be - 20, a1=-2 * xi * (7 * xi ** 2 - 300 * be - 20),
              a0=1440 * be ** 2 - 120 * be * xi ** 2 + 480 * be - 14 * xi ** 4 + 160 * xi ** 2 + 64)
    twisted = sp.roots(sp.Poly(xi * (xi ** 2 - 1) * (7 * xi ** 2 + 17), xi))
    assert len(twisted) == 5
    untwisted = set()
    for line, poly, var in ((be + xi / 2, xi * (xi - 4) * (xi ** 2 - 4) * (7 * xi - 2), xi),
                            (be - xi / 2, xi * (xi + 4) * (xi ** 2 - 4) * (7 * xi + 2), xi),
                            (xi, be * (30 * be ** 2 + 15 * be + 2), be),
                            (be, xi * (xi ** 2 - 4) * (7 * xi ** 2 + 8), xi)):
        for root in sp.roots(sp.Poly(poly, var)):
            other = sp.solve(line.subs(var, root), be if var == xi else xi)[0]
            untwisted.add((sp.nsimplify(other), sp.nsimplify(root)) if var == xi else (sp.nsimplify(root), sp.nsimplify(other)))
    assert len(untwisted) == 15
    # z -> -z: the tables are even (a2, a0) or odd (a3, a1) in xi, and the solution sets are symmetric
    for tab in (tw, un):
        assert all(sp.expand(tab[key].subs(xi, -xi) - sign * tab[key]) == 0
                   for key, sign in (("a3", -1), ("a2", 1), ("a1", -1), ("a0", 1)))
    assert {(b0, -x0) for b0, x0 in untwisted} == untwisted and {-x0 for x0 in twisted} == set(twisted)
    # the remaining condition: both end values w = beta -+ xi/2 satisfy w (a0 - 480 w^2) = 0
    for b0, x0 in untwisted:
        for end in (b0 - x0 / 2, b0 + x0 / 2):
            assert sp.simplify(end * (un["a0"].subs({be: b0, xi: x0}) - 480 * end ** 2)) == 0
        assert sp.simplify(un["a2"].subs({be: b0, xi: x0})) != 0
    sigma2 = {"twisted": set(), "untwisted": set()}
    for x0 in twisted:
        a = {k: sp.simplify(v.subs(xi, x0)) for k, v in tw.items()}
        assert a["a2"] != 0
        lam2 = 2 / a["a2"]
        sigma2["twisted"].add(sp.nsimplify(sp.simplify(lam2 * a["a3"] ** 2)))
        if x0 != 0:                                   # R(P) = 0:  s / sigma = (392 - 39 sigma^2)/1372
            assert sp.simplify(lam2 * a["a1"] / a["a3"] - (392 - 39 * lam2 * a["a3"] ** 2) / 1372) == 0
    assert sigma2["twisted"] == {0, 7, sp.Rational(833, 89)}
    # rates of the twisted waves: lambda^2 = 2/a2 is positive for xi = 0, +-1 and negative for xi^2 = -17/7
    lam2_tw = {sp.nsimplify(x0 ** 2): sp.nsimplify(sp.simplify(2 / tw["a2"].subs(xi, x0))) for x0 in twisted}
    assert lam2_tw == {0: sp.Rational(1, 5), 1: sp.Rational(1, 28), -sp.Rational(17, 7): -sp.Rational(7, 356)}
    # untwisted waves with sigma != 0 against Kudryashov-Sinelshchikov (28)-(31) at beta_KS = 0
    ks = {"(28),(29)": (lambda t: -(201 * t - 1960) / 2744,
                        lambda t: (81 * t * (9505216 - 1565928 * t + 82933 * t ** 2) - 697019904) / (2963520 * (-383 * t - 784))),
          "(30),(31)": (lambda t: (171 * t - 980) / 686,
                        lambda t: (27 * t * (101286528 - 27836424 * t + 2136139 * t ** 2) - 1858719744) / (185220 * (7056 - 1563 * t)))}
    matches = {}
    at_392_11 = {}
    for b0, x0 in untwisted:
        a = {k: sp.simplify(v.subs({be: b0, xi: x0})) for k, v in un.items()}
        lam2 = 2 / a["a2"]
        t = sp.nsimplify(sp.simplify(lam2 * a["a3"] ** 2))                  # sigma^2
        sigma2["untwisted"].add(t)
        if x0 == 0:
            continue
        ours = (sp.simplify(lam2 * a["a1"] / a["a3"]), sp.simplify(-lam2 ** 2 * a["a0"]))   # (s/sigma, alpha)
        if t == sp.Rational(392, 11):
            at_392_11.setdefault(ours, set()).add((b0, x0))
        hit = [name for name, (c0, al) in ks.items()
               if sp.simplify(c0(t) - ours[0]) == 0 and sp.simplify(al(t) - ours[1]) == 0]
        assert len(hit) == 1
        matches.setdefault(str(t), set()).add(hit[0])
    assert sigma2["untwisted"] == {0, sp.Rational(392, 71), sp.Rational(1568, 239), sp.Rational(392, 11),
                                   sp.Rational(392, 41), sp.Rational(784, 127)}
    assert matches["392/11"] == {"(28),(29)", "(30),(31)"}
    # sigma^2 = 392/11: two waves for each sign of sigma, given by the mirror pairs (1, +-2) and (-1/7, +-2/7),
    # with different (s/sigma, alpha)
    assert sorted(at_392_11.values(), key=str) == sorted(
        [{(sp.Integer(1), sp.Integer(2)), (sp.Integer(1), sp.Integer(-2))},
         {(-sp.Rational(1, 7), sp.Rational(2, 7)), (-sp.Rational(1, 7), -sp.Rational(2, 7))}], key=str)
    assert len({al for _, al in at_392_11}) == 2 and len({c0 for c0, _ in at_392_11}) == 2
    return {"R(P)": "(1372*s + 39*sigma^3 - 392*sigma)/123480", "twisted": 5, "untwisted": 15,
            "sigma^2_twisted": sorted(str(v) for v in sigma2["twisted"]),
            "sigma^2_untwisted": sorted(str(v) for v in sigma2["untwisted"]),
            "lambda^2_twisted_by_xi^2": {str(key): str(val) for key, val in sorted(lam2_tw.items(), key=lambda kv: str(kv[0]))},
            "waves_at_sigma^2_392/11": {str(key): sorted(str(v) for v in val) for key, val in sorted(at_392_11.items(), key=str)},
            "KS2012_formulas": {k: sorted(v) for k, v in sorted(matches.items())}}


def example_ks_degenerate():
    G, k = sp.symbols("G k")
    D = lambda F: sp.expand(sp.diff(F, G) * G * (k - G))          # DG = G (k - G)
    wp = k ** 2 / 12 - D(G)                                       # degenerate wp: g2 = k^4/12, g3 = -k^6/216
    w = -60 * D(wp) - 60 * wp - 1
    lhs = sp.expand(D(D(D(w))) + 4 * D(D(w)) + D(w) + w ** 2 / 2)
    out = {}
    for k2 in (1, -1):                                            # k^4 = 1, so g2 = 1/12
        g3 = -sp.Rational(k2 ** 3, 216)
        assert sp.Rational(1, 12) ** 3 == 27 * g3 ** 2            # the excluded values of g3
        C = 1080 * g3 - 13
        remainder = sp.rem(sp.Poly(lhs + C, k), sp.Poly(k ** 2 - k2, k))
        assert remainder.is_zero
        out["k^2 = %d" % k2] = "g3 = %s, C = %s" % (g3, C)
    assert sorted(out.values()) == ["g3 = -1/216, C = -18", "g3 = 1/216, C = -8"]
    return out


def example_drift_degree_two():
    G, k, c = sp.symbols("G k c")
    D = lambda F: sp.expand(sp.diff(F, G) * G * (k - G))
    v = c * G
    residual = sp.expand(D(D(v)) - 3 * k * D(v) + 2 * k ** 2 * v + v ** 3 / 3)
    assert sp.expand(residual - c * G ** 3 * (2 + c ** 2 / 3)) == 0      # zero exactly for c^2 = -6
    # n = 3, p = 2, q = 1: K = (q)_p = 2 and c^2 = -nK = -6
    return {"v = c G, c^2 = -6": "v'' - 3k v' + 2k^2 v + v^3/3 = 0"}


def example_two_waves_sextic():
    x, g2, g3, a0 = sp.symbols("x g2 g3 a0")

    def D2(F):
        return sp.expand(sp.diff(F, x, 2) * (4 * x ** 3 - g2 * x - g3) + sp.diff(F, x) * (6 * x ** 2 - g2 / 2))

    K = 332640                                        # (6)_6
    R, h, extra = make_ring(["a0"])
    fam = Family(2, 6, R, h, [extra["a0"]] + [R(0)] * 5, 44)
    assert fam.F_J == 0 and all(fam.b[j] == 0 for j in range(1, 6)) and fam.b[6] == extra["a0"] * QQ(1, 2 * K)
    # Lambda_6 = (wp - [z^0] wp)/120, with wp = 120 wp^3 - 18 g2 wp - 12 g3 and [z^0] wp = 6 g3/7
    wp4 = D2(D2(x))
    assert sp.expand(wp4 - (120 * x ** 3 - 18 * g2 * x - 12 * g3)) == 0
    lam6 = sp.expand((wp4 - sp.Rational(6, 7) * g3) / 120)
    assert sp.expand(lam6 - (x ** 3 - sp.Rational(3, 20) * g2 * x - sp.Rational(3, 28) * g3)) == 0
    u = lam6 + a0 / (2 * K)
    residual = sp.Poly(sp.expand(D2(D2(D2(u))) + a0 * u - K * u ** 2), x)
    assert residual.degree() == 2
    assert sp.expand(residual.coeff_monomial(x ** 2) + sp.Rational(8316, 5) * g2 ** 2) == 0
    assert sp.expand(residual.coeff_monomial(x) + 4536 * g2 * g3) == 0
    assert sp.expand(1330560 * residual.coeff_monomial(1) - (a0 ** 2 - 16765056 * g2 ** 3 - 3643833600 * g3 ** 2)) == 0
    assert 2160 ** 2 * 781 == 3643833600
    # the two waves and the quadratic selector (J = 18; all A_i, B_i with 1 <= i <= 16 vanish here)
    phi = phi_u_Hn(fam, 17)
    assert all(phi.get(i, R(0)) == 0 for i in range(1, 17))
    Hs, As = sp.symbols("h a0")
    Q = sp.Poly(phi[17].as_expr(), Hs)
    assert Q.degree() == 2 and Q.coeff_monomial(Hs) == 0
    c = wp_series(0, g3, 10)                          # wp = z^-2 + sum c_k z^(2k-2), g2 = 0
    wp = {-2: sp.Integer(1)}
    wp.update({2 * k - 2: c[k] for k in c if c[k] != 0})
    cube = {}
    for e1, c1 in wp.items():
        for e2, c2 in wp.items():
            for e3, c3 in wp.items():
                if e1 + e2 + e3 <= 12:
                    cube[e1 + e2 + e3] = sp.expand(cube.get(e1 + e2 + e3, 0) + c1 * c2 * c3)
    assert sp.expand(cube[0] - sp.Rational(3, 28) * g3) == 0          # constant term of wp^3
    assert sp.expand(cube[6] - fam.b[12].as_expr().subs(As, a0)).subs(g3 ** 2, a0 ** 2 / 3643833600) == 0
    hval = sp.expand(cube[12]).subs(g3 ** 3, g3 * a0 ** 2 / 3643833600)     # h = b_18, odd in g3
    assert sp.simplify(sp.expand(Q.as_expr().subs(Hs, hval)).subs(g3 ** 2, As ** 2 / 3643833600).subs(a0, As)) == 0
    return {"u": "wp^3 - 3*g2*wp/20 - 3*g3/28 + a0/665280", "g2": "0", "g3^2": "a0^2/3643833600",
            "h": str(sp.factor(hval)), "selector_vanishes_at_both": True}


def main():
    report = {}
    for name, check in (("selector_example", example_selector),
                        ("hypothesis_needed", example_hypothesis_needed),
                        ("two_waves_n2", example_two_waves_n2),
                        ("compatibility", example_compatibility),
                        ("swift_hohenberg", example_swift_hohenberg),
                        ("ks_degenerate", example_ks_degenerate),
                        ("drift_degree_two", example_drift_degree_two),
                        ("two_waves_sextic", example_two_waves_sextic)):
        report[name] = check()
        print("PASS", name, flush=True)
    target = Path(__file__).with_name("more_examples_checks.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {target.name}.")


if __name__ == "__main__":
    main()
