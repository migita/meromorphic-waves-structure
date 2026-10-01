#!/usr/bin/env python3
"""Exact checks for identities stated in the text.

This script checks identities stated in the introduction and in the
sections on the global structure, consequences and examples, and re-derives
the universal leading coefficients of the two selectors (the section on the
free coefficient) from the pure operator D^p, independently of the other
scripts.

Run from this folder:

    python3 check_text_identities.py

The only output file is text_identities_checks.json beside this script.
"""

from __future__ import annotations

import json
from math import comb, factorial
from pathlib import Path

import sympy as s

z, k, y, t, G = s.symbols("z k y t G")
HERE = Path(__file__).resolve().parent
results = {}


def record(name, value="ok"):
    results[name] = value
    print(f"{name}: {value}")


# ----------------------------------------------------------------------
# 1. Periodic generators: g_r, orientation rule, ansatz form D^j g_r = g_r Q_j(G)
# ----------------------------------------------------------------------
def generator(r, m, kk=k):
    return kk * s.exp(r * kk * z / m) / (s.exp(kk * z) - 1)


def check_generators():
    for m in range(1, 5):
        Gk = generator(m, m)
        # DG = G (k - G)
        assert s.simplify(s.diff(Gk, z) - Gk * (k - Gk)) == 0
        # G = (k/2)(1 + coth(kz/2))
        assert s.simplify((Gk - k / 2 * (1 + s.coth(k * z / 2))).rewrite(s.exp)) == 0
        for r in range(1, m + 1):
            g = generator(r, m)
            # logarithmic derivative r k / m - G
            assert s.simplify(s.diff(g, z) / g - (s.Rational(r, m) * k - Gk)) == 0
            # orientation rule
            if r < m:
                assert s.simplify(g - generator(m - r, m, -k)) == 0
            else:
                assert s.simplify(generator(m, m, -k) - (g - k)) == 0
            # D^j g_r = g_r Q_j(G) with deg Q_j = j and leading coefficient (-1)^j j!
            Q = s.Integer(1)
            for j in range(1, 6):
                # D(g Q(G)) = g [(r k/m - G) Q + G (k - G) Q'(G)]
                Q = s.expand((s.Rational(r, m) * k - G) * Q + G * (k - G) * s.diff(Q, G))
                assert s.degree(Q, G) == j
                assert s.LC(s.Poly(Q, G)) == (-1) ** j * factorial(j)
                direct = s.diff(g, z, j) / g
                assert s.simplify(direct - Q.subs(G, Gk)) == 0
    # translations used in the introduction (m = 2, r = 1 and G)
    g1 = generator(1, 2)
    Gk = generator(2, 2)
    shift = {z: z + s.I * s.pi / k}
    assert s.simplify((g1.subs(shift) + s.I * k / 2 * s.sech(k * z / 2)).rewrite(s.exp)) == 0
    assert s.simplify((Gk.subs(shift) - k / 2 * (1 + s.tanh(k * z / 2))).rewrite(s.exp)) == 0
    record("generators_orientation_and_ansatz_form")


# ----------------------------------------------------------------------
# 2. Travelling-wave reductions used in Sections 8.2 and 9.2
# ----------------------------------------------------------------------
def check_reductions():
    x, tt, sp = s.symbols("x t s")
    xi = x - sp * tt
    w = s.Function("w")
    W = w(xi)
    mu, delta, nu, A, w0 = s.symbols("mu delta nu A w0")

    # Kuramoto--Sivashinsky with dispersion
    pde = (s.diff(W, tt) + W * s.diff(W, x) + mu * s.diff(W, x, 2)
           + delta * s.diff(W, x, 3) + nu * s.diff(W, x, 4))
    ode = (nu * s.diff(W, x, 3) + delta * s.diff(W, x, 2) + mu * s.diff(W, x)
           - sp * W + W**2 / 2)
    assert s.simplify(pde - s.diff(ode, x)) == 0

    # the shift w = v + w0 removes A when w0^2/2 - s w0 - A = 0
    v = s.Function("v")(z)
    shifted = (nu * s.diff(v, z, 3) + delta * s.diff(v, z, 2) + mu * s.diff(v, z)
               - sp * (v + w0) + (v + w0) ** 2 / 2 - A)
    target = (nu * s.diff(v, z, 3) + delta * s.diff(v, z, 2) + mu * s.diff(v, z)
              + (w0 - sp) * v + v**2 / 2)
    assert s.expand(shifted - target - (w0**2 / 2 - sp * w0 - A)) == 0

    # residue of the normalized wave for nu D^3 + delta D^2 + mu D + a0
    a0 = s.symbols("a0")
    c1, c2 = s.symbols("c1 c2")
    u = z**-3 + c1 * z**-2 + c2 * z**-1
    V = 120 * nu * u
    res = s.expand(nu * s.diff(V, z, 3) + delta * s.diff(V, z, 2) + mu * s.diff(V, z)
                   + a0 * V + V**2 / 2)
    assert s.expand(res.coeff(z, -6)) == 0
    sol = s.solve([res.coeff(z, -5), res.coeff(z, -4)], [c1, c2], dict=True)[0]
    assert s.simplify(sol[c2] - (16 * mu * nu - delta**2) / (608 * nu**2)) == 0

    # modified Kawahara equation with a diffusion term
    eps, b = s.symbols("epsilon b")
    pde2 = (s.diff(W, tt) + W**2 * s.diff(W, x) + eps * s.diff(W, x, 2)
            + b * s.diff(W, x, 3) + s.diff(W, x, 5))
    ode2 = (s.diff(W, x, 4) + b * s.diff(W, x, 2) + eps * s.diff(W, x)
            - sp * W + W**3 / 3)
    assert s.simplify(pde2 - s.diff(ode2, x)) == 0
    record("ks_and_drift_reductions")


# ----------------------------------------------------------------------
# 3. Scaling: lambda^q u(lambda z) for P_lambda(rho) = lambda^p P(rho/lambda)
# ----------------------------------------------------------------------
def check_scaling():
    # The identities are polynomial in u and its derivatives, so a generic
    # polynomial test function with symbolic coefficients suffices.
    lam, X = s.symbols("lambda X", nonzero=True)
    cs = s.symbols("c0:9")

    def u_of(var):
        return sum(cs[i] * var**i for i in range(9))

    rho = s.symbols("rho")
    for n, q in ((2, 3), (3, 2), (4, 1)):
        p = (n - 1) * q
        K = (-1) ** p * s.rf(q, p)
        a = s.symbols(f"a0:{p}")
        P = rho**p + sum(a[j] * rho**j for j in range(p))
        Pl = s.expand(lam**p * P.subs(rho, rho / lam))

        def apply(poly, f, var):
            return sum(poly.coeff(rho, j) * s.diff(f, var, j) for j in range(p + 1))

        ul = lam**q * u_of(lam * z)
        lhs = apply(Pl, ul, z) - K * ul**n
        rhs = lam ** (p + q) * (apply(P, u_of(X), X) - K * u_of(X) ** n)
        assert s.expand(lhs - rhs.subs(X, lam * z)) == 0

    # the two-wave equation u'''' = 108 u + 120 u^3 scales to 108 lambda^4
    U = lam**2 * u_of(lam * z)
    ode = s.diff(U, z, 4) - 108 * lam**4 * U - 120 * U**3
    uX = u_of(X)
    scaled = lam**6 * (s.diff(uX, X, 4) - 108 * uX - 120 * uX**3)
    assert s.expand(ode - scaled.subs(X, lam * z)) == 0
    record("scaling_rule_and_two_wave_family")


# ----------------------------------------------------------------------
# 4. Universal selector coefficients from the pure operator D^p
# ----------------------------------------------------------------------
def pure_germ(n, q, order_h):
    """Formal family f(z;h) for P = D^p up to degree order_h in h."""
    p = (n - 1) * q
    J = (n + 1) * q
    K = s.rf(q, p) * (-1) ** p
    h = s.symbols("h")
    cs = [s.Integer(1), s.Integer(1)]
    x = s.symbols("x")
    for d in range(2, order_h + 1):
        cd = s.symbols("cd")
        phi = sum(cs[i] * x**i for i in range(d)) + cd * x**d
        # coefficient of z^{dJ - q - p}: I(dJ) c_d = K [x^d] phi^n (without the linear term)
        I = s.ff(d * J - q, p) - n * K
        rhs = K * (s.expand(phi**n).coeff(x, d) - n * cd)
        cs.append(s.nsimplify(s.solve(s.Eq(I * cd, rhs), cd)[0]))
    f = z**-q * sum(cs[i] * (h * z**J) ** i for i in range(order_h + 1))
    return f, h, cs


def principal_poly(series, rho):
    ser = s.expand(series)
    out = 0
    order = -s.Poly(s.expand(ser * z**200), z).monoms()[-1][0] + 200
    for r in range(1, order + 1):
        coeff = ser.coeff(z, -r)
        if coeff != 0:
            out += coeff * (-rho) ** (r - 1) / factorial(r - 1)
    return s.expand(out)


def apply_poly(poly, rho, series):
    poly = s.Poly(poly, rho)
    total = 0
    for (deg,), coeff in poly.terms():
        total += coeff * s.diff(series, z, deg)
    return s.expand(total)


def H_of(u, q):
    return s.expand(u * s.diff(u, z, 2) - s.Rational(q + 1, q) * s.diff(u, z) ** 2)


def check_seed_symbol():
    q, j, eps = s.symbols("q j epsilon", positive=True)
    u = z**-q * (1 + eps * z**j)
    H = u * s.diff(u, z, 2) - (q + 1) / q * s.diff(u, z) ** 2
    lin = s.simplify(s.diff(H, eps).subs(eps, 0) / z ** (j - 2 * q - 2))
    assert s.simplify(lin - j * (j + 1)) == 0
    assert s.simplify(H.subs(eps, 0)) == 0
    record("seed_linear_symbol_j(j+1)")


def check_selectors():
    rho = s.symbols("rho")
    rows = []
    # quadratic selector: n = 2 (q even >= 4) and n = 3 (q >= 2)
    for n, q in ((2, 4), (2, 6), (3, 2), (3, 3), (3, 4)):
        p = (n - 1) * q
        J = (n + 1) * q
        f, h, cs = pure_germ(n, q, 2)
        K = s.rf(q, p)
        X = s.rf((n + 2) * q + 1, p) / s.rf(q, p)
        c2 = s.Rational(comb(n, 2)) / (X - n)
        assert s.simplify(cs[2] - c2) == 0
        Hn = s.expand(f ** (n - 2) * H_of(f, q))
        kappa = J * (2 * (2 * J + 1) * c2 + (n - 2) * J - 2)
        h2 = s.expand(Hn).coeff(h, 2)
        low = s.expand(h2 / z ** ((n + 2) * q - 2))
        assert s.simplify(low.subs(z, 0) - kappa) == 0 if low.is_polynomial(z) else False
        # bracket G = fhat(D) Hn - Hnhat(D) f
        fhat = principal_poly(f.subs(h, 0), rho)
        Fhat = principal_poly(Hn.subs(h, 0), rho)
        Gs = s.expand(apply_poly(fhat, rho, Hn) - apply_poly(Fhat, rho, f))
        a_star = s.Rational((-1) ** (q - 1), factorial(q - 1))
        Delta2 = a_star * kappa * s.ff((n + 2) * q - 2, q - 1)
        coeff_h2 = s.expand(Gs.coeff(h, 2))
        # no negative powers, lowest power z^{J-1} with coefficient Delta2
        for e in range(-3 * J, J - 1):
            assert s.simplify(coeff_h2.coeff(z, e)) == 0
        assert s.simplify(coeff_h2.coeff(z, J - 1) - Delta2) == 0
        assert Delta2 != 0
        rows.append({"n": n, "q": q, "psi": str(c2), "kappa": str(kappa),
                     "Delta2": str(Delta2)})
    # affine selector: p even, p > 2q + 1
    for n, q in ((4, 2), (5, 1), (5, 2), (7, 1)):
        p = (n - 1) * q
        if p % 2:
            continue
        J = (n + 1) * q
        f, h, cs = pure_germ(n, q, 1)
        H = H_of(f, q)
        Ehat = principal_poly(s.expand(f.subs(h, 0) ** 2), rho)
        Fhat = principal_poly(H.subs(h, 0), rho)
        Gs = s.expand(apply_poly(Ehat, rho, H) - apply_poly(Fhat, rho, s.expand(f**2)))
        Delta1 = -s.Rational(J * (J + 1), factorial(2 * q - 1)) * s.ff(p - 2, 2 * q - 1)
        lin = s.expand(Gs.coeff(h, 1))
        for e in range(-3 * J, p - 2 * q - 1):
            assert s.simplify(lin.coeff(z, e)) == 0
        assert s.simplify(lin.coeff(z, p - 2 * q - 1) - Delta1) == 0
        rows.append({"n": n, "q": q, "Delta1": str(Delta1)})
    # the inequality (5/2)^q > 6q + 3 and X - 2 > 6q + 1 for even q >= 4 (n = 2)
    for q in range(4, 61, 2):
        X = s.rf(4 * q + 1, q) / s.rf(q, q)
        assert X > s.Rational(5, 2) ** q > 6 * q + 3
    # (n, q) = (2, 2): X = 15 and kappa = 0
    X22 = s.rf(9, 2) / s.rf(2, 2)
    assert X22 == 15
    J22 = 6
    assert J22 * (2 * (2 * J22 + 1) / (X22 - 2) - 2) == 0
    record("selector_leading_coefficients", rows)


# ----------------------------------------------------------------------
# 5. Resonance lemma: roots of I(j) for small n, q
# ----------------------------------------------------------------------
def check_resonances():
    for n in range(2, 7):
        for q in range(1, 7):
            p = (n - 1) * q
            K = (-1) ** p * s.rf(q, p)
            roots = [j for j in range(1, 6 * (n + 1) * q)
                     if s.ff(j - q, p) - n * K == 0]
            expected = [(n + 1) * q] if p % 2 == 0 else []
            assert roots == expected, (n, q, roots)
            assert s.ff(-1 - q, p) - n * K == 0  # I(-1) = 0
    record("resonance_lemma_small_cases")


# ----------------------------------------------------------------------
# 6. End-value identity and residues on the Swift--Hohenberg waves (m = 2)
# ----------------------------------------------------------------------
def residue_R(a3, a2, a1, a0):
    """R(P) = l_{p-1} for P = rho^4 + a3 rho^3 + ... and P(D)u = 120 u^3."""
    bs = s.symbols("b1:4")
    u = z**-2 * (1 + bs[0] * z + bs[1] * z**2 + bs[2] * z**3)
    res = s.expand(s.diff(u, z, 4) + a3 * s.diff(u, z, 3) + a2 * s.diff(u, z, 2)
                   + a1 * s.diff(u, z) + a0 * u - 120 * u**3)
    sol = {}
    for i, e in enumerate((-5, -4, -3)):
        row = s.expand(res.coeff(z, e).subs(sol))
        sol[bs[i]] = s.solve(row, bs[i])[0]
    b1, b2, b3 = (sol[bs[0]], sol[bs[1]], sol[bs[2]])
    return s.simplify(2 * b3 + 2 * b1 * b2)


def check_sh_end_values():
    xi, beta = s.symbols("xi beta")
    op1 = dict(a3=-14 * xi, a2=46 * xi**2 + 10, a1=-2 * xi * (7 * xi**2 + 10),
               a0=-14 * xi**4 - 80 * xi**2 - 11)
    op2 = dict(a3=-14 * xi, a2=46 * xi**2 - 120 * beta - 20,
               a1=-2 * xi * (7 * xi**2 - 300 * beta - 20),
               a0=1440 * beta**2 - 120 * beta * xi**2 + 480 * beta
               - 14 * xi**4 + 160 * xi**2 + 64)
    twisted = [0, 1, -1, s.I * s.sqrt(s.Rational(17, 7)), -s.I * s.sqrt(s.Rational(17, 7))]
    for x0 in twisted:
        vals = {kk: s.nsimplify(vv.subs(xi, x0)) for kk, vv in op1.items()}
        assert s.simplify(residue_R(**vals)) == 0  # both ends vanish
    points = set()
    lines = [
        (lambda X: -X / 2, xi * (xi - 4) * (xi**2 - 4) * (7 * xi - 2)),
        (lambda X: X / 2, xi * (xi + 4) * (xi**2 - 4) * (7 * xi + 2)),
    ]
    for bfun, poly in lines:
        for x0 in s.solve(poly, xi):
            points.add((x0, bfun(x0)))
    for b0 in s.solve(beta * (30 * beta**2 + 15 * beta + 2), beta):
        points.add((s.Integer(0), b0))
    for x0 in s.solve(xi * (xi**2 - 4) * (7 * xi**2 + 8), xi):
        points.add((x0, s.Integer(0)))
    for x0, b0 in points:
        vals = {kk: s.simplify(vv.subs({xi: x0, beta: b0})) for kk, vv in op2.items()}
        R = residue_R(**vals)
        # u = -2 w2, u_+ = -2(beta - xi/2), u_- = -2(beta + xi/2), k = 2
        u_plus, u_minus = -2 * (b0 - x0 / 2), -2 * (b0 + x0 / 2)
        assert s.simplify(u_plus**2 - u_minus**2 - 2 * R) == 0
    # sigma = 0 entries: sigma = -lambda a3 = 14 lambda xi, so xi = 0, then a1 = 0
    assert op1["a1"].subs(xi, 0) == 0 and op2["a1"].subs(xi, 0) == 0
    record("sh_end_value_identity_points", len(points) + len(twisted))


# ----------------------------------------------------------------------
# 7. Degree two: v = -12 wp(z; a0^2/12, g3) - a0 solves v'' + a0 v + v^2/2 = 0
# ----------------------------------------------------------------------
def check_order_two_family():
    a0, g3, W = s.symbols("a0 g3 W")
    g2 = a0**2 / 12
    # use wp'' = 6 wp^2 - g2/2, i.e. v'' = -12 (6 W^2 - g2/2) with W = wp
    v = -12 * W - a0
    v2 = -12 * (6 * W**2 - g2 / 2)
    assert s.expand(v2 + a0 * v + v**2 / 2) == 0
    # Laurent series of wp from the recursion wp = z^-2 + sum c_k z^(2k-2)
    N = 6
    c = {2: g2 / 20, 3: g3 / 28}
    for kk in range(4, N + 1):
        c[kk] = s.Rational(3, (2 * kk + 1) * (kk - 3)) * sum(
            c[j] * c[kk - j] for j in range(2, kk - 1))
    wp = z**-2 + sum(c[kk] * z ** (2 * kk - 2) for kk in range(2, N + 1))
    # normalized wave for n = 2, q = 2: c = -12, u = v / c = wp + a0/12
    u = s.expand(wp + a0 / 12)
    res = s.expand(s.diff(u, z, 2) + a0 * u - 6 * u**2)  # P(D)u = K u^2, K = 6
    for e in range(-4, 2 * N - 6):
        assert s.simplify(res.coeff(z, e)) == 0, e
    b = s.expand(u * z**2)
    assert s.simplify(b.coeff(z, 6) - g3 / 28) == 0  # free coefficient h = b_6
    assert s.simplify(b.coeff(z, 2) - a0 / 12) == 0
    assert s.simplify(b.coeff(z, 4) - a0**2 / 240) == 0
    record("degree_two_family_free_coefficient_g3_over_28")


def main():
    check_generators()
    check_reductions()
    check_scaling()
    check_seed_symbol()
    check_resonances()
    check_selectors()
    check_sh_end_values()
    check_order_two_family()
    out = HERE / "text_identities_checks.json"
    out.write_text(json.dumps({"status": "all identity checks passed",
                               "checks": results}, indent=2) + "\n")
    print("All identity checks passed; wrote", out.name)


if __name__ == "__main__":
    main()
