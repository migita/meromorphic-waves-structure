#!/usr/bin/env python3
"""The terms of the bracket that are linear in the free coefficient.

Checks for the section "Operators that are not even" and the appendix "The
terms linear in the free coefficient" (pair E = u, F = H_n):

1. A_i is weighted-homogeneous of weight i + 1 in the coefficients of P.
2. A_1 = 0 for even operators (and A_0 = A_2 = 0 there); the residue formulas
   for A_0, A_1 and the two identities used in the proof, for any operator.
3. A_0 = c_0 a_{p-1} and A_1 = c_1 a_{p-1}^2 with the stated constants.
4. The explicit family of P_eps = prod (rho + (q+i) eps): the closed forms of
   f, f_h, eta = H_n(f), eta_h at the wave (h' = 0), and the values of A_0, A_1.
5. The first odd coefficient: formula for c_{j-1}, its agreement with the
   series, the inequality used in the proof, and its sign for p <= 120.
6. One wave unless P is even, on the Swift-Hohenberg waves with sigma != 0:
   the free coefficient of each wave solves A_i h + B_i = 0 for 1 <= i <= 6,
   and no two different waves share an operator.
7. The even operator D^6 + a0 with n = 3: A_5 = -13 a0/888, the wave
   -wp'(z; 0, -a0/4320)/2, and the second root of the quadratic selector,
   which is not the free coefficient of a wave.

All arithmetic is exact. Writes linear_terms_checks.json.
"""

import json
import random
from fractions import Fraction as Fr
from math import comb, factorial
from pathlib import Path

import sympy as sp

from series_tools import (Family, falling, linear_terms, make_ring, needed_order, phi_u_Hn,
                          poly_from_roots, rising, s_add, s_diff, s_mul, s_pow, seed, to_fraction)

CASES = [(3, 2), (2, 4), (3, 3), (2, 6), (3, 4)]


def constants(n, q):
    J = (n + 1) * q
    c0 = Fr((-1) ** q * 2 * (n + 1) * (n * q + 1) * (J + 1), (n - 1) * (J - 1))
    c1 = -Fr(n * q + 2, (n - 1) * (J - 1)) * c0
    return c0, c1


def random_operator(p, rng, even=False):
    return [Fr(rng.randint(-5, 5), rng.randint(1, 4)) if (k % 2 == 0 or not even) else Fr(0) for k in range(p)]


def check_homogeneity(rng):
    for n, q in CASES[:3]:
        p = (n - 1) * q
        a = random_operator(p, rng)
        lam = Fr(3, 2)
        scaled = [a[k] * lam ** (p - k) for k in range(p)]        # a_{p-k} has weight k
        base, _ = linear_terms(n, q, a, 3)
        moved, _ = linear_terms(n, q, scaled, 3)
        for i in range(4):
            assert moved[i][0] == lam ** (i + 1) * base[i][0], (n, q, i)
    return "A_i(P_lambda) = lambda^(i+1) A_i(P) for i <= 3"


def check_even_operators(rng):
    for n, q in CASES:
        p = (n - 1) * q
        for _ in range(3):
            terms, fam = linear_terms(n, q, random_operator(p, rng, even=True), 2)
            assert fam.F_J == 0                                   # even operators are compatible
            assert terms[0][0] == 0 and terms[1][0] == 0 and terms[2][0] == 0
    return "A_0 = A_1 = A_2 = 0 and F_J = 0 for random even operators"


def scaled(A, factor):
    return {e: c * factor for e, c in A.items() if c * factor != 0}


def check_residue_identities(rng):
    """Appendix: A_0, A_1 as residues, and the identities in the proof that A_1 = 0 for even P."""

    def h_part(series, degree):
        out = {}
        for e, c in series.items():
            value = sum((to_fraction(cc) for mon, cc in c.terms() if mon[0] == degree), Fr(0))
            if value != 0:
                out[e] = value
        return out

    def reflect(A):
        return {e: c * (-1) ** (e % 2) for e, c in A.items()}

    def residue(A):
        return A.get(-1, Fr(0))

    def power(A, k):
        out = {0: Fr(1)}
        for _ in range(k):
            out = s_mul(out, A)
        return out

    for n, q in CASES[:3]:
        p, J = (n - 1) * q, (n + 1) * q
        K = rising(q, p)
        for even in (False, True):
            a = random_operator(p, rng, even=even)
            terms, fam = linear_terms(n, q, a, 1)
            A0, A1 = terms[0][0], terms[1][0]
            f = fam.series()
            Hn = s_mul(s_pow(f, n - 2, fam.R), seed(f, q))
            top = len(fam.b) - 1
            f0, f1 = ({e: c for e, c in h_part(f, d).items() if e <= top - q} for d in (0, 1))
            e0, e1 = ({e: c for e, c in h_part(Hn, d).items() if e <= top - n * q - 2} for d in (0, 1))
            # residue formulas for A_0 and A_1
            assert residue(s_add(s_mul(f0, reflect(e1)), s_mul(e0, reflect(f1)), -1)) == A0
            assert residue(s_add(s_mul(s_diff(f0), reflect(e1)), s_mul(s_diff(e0), reflect(f1)), -1)) == A1
            # Res[f0' eta1 - eta0' f1] = (J+2)/q * Res[f1 G]
            d1, d2, d3 = s_diff(f0), s_diff(f0, 2), s_diff(f0, 3)
            G = scaled(s_mul(power(f0, n - 2), s_mul(d1, d2)), 3)
            if n >= 3:
                G = s_add(G, scaled(s_mul(power(f0, n - 3), s_mul(d1, s_mul(d1, d1))), n - 2))
            lhs = residue(s_add(s_mul(d1, e1), s_mul(s_diff(e0), f1), -1))
            assert lhs == Fr(J + 2, q) * residue(s_mul(f1, G))
            # n(n-1)K Res[f1 G] = Res[f1 M d3], where d3 is the third derivative of f0
            # and M = P(D) - nK f0^(n-1) is the linearized operator
            Md3 = scaled(s_mul(power(f0, n - 1), d3), -n * K)
            for k, coefficient in enumerate(a + [Fr(1)]):
                Md3 = s_add(Md3, scaled(s_diff(d3, k), coefficient))
            assert n * (n - 1) * K * residue(s_mul(f1, G)) == residue(s_mul(f1, Md3))
            if even:
                assert lhs == 0 and A1 == 0
    return "residue formulas for A_0, A_1 and the identities in the proof of A_1 = 0 for even P"


def check_A0_A1(rng):
    for n, q in CASES:
        p = (n - 1) * q
        c0, c1 = constants(n, q)
        for _ in range(3):
            a = random_operator(p, rng)
            terms, _ = linear_terms(n, q, a, 1)
            assert terms[0][0] == c0 * a[p - 1]
            assert terms[1][0] == c1 * a[p - 1] ** 2
    return "A_0 = c_0 a_{p-1}, A_1 = c_1 a_{p-1}^2 for random operators"


def check_explicit_family():
    z = sp.Symbol("z")
    for n, q in CASES:
        p, J = (n - 1) * q, (n + 1) * q
        for eps in (Fr(1), Fr(-2, 3)):
            roots = [-(q + i) * eps for i in range(p)]
            coeffs = poly_from_roots(roots)
            a = coeffs[:p]
            assert a[p - 1] == eps * p * (J - 1) / 2
            terms, fam = linear_terms(n, q, a, 1)
            assert fam.F_J == 0
            A0, A1 = terms[0][0], terms[1][0]
            assert A0 == (-1) ** q * eps * J * (J + 1) * (n * q + 1)
            assert A1 == (-1) ** (q + 1) * eps ** 2 * Fr(q * J * (J + 1) * (n * q + 1) * (n * q + 2), 2)
            c0, c1 = constants(n, q)
            assert A0 == c0 * a[p - 1] and A1 == c1 * a[p - 1] ** 2
    # closed forms of the appendix, compared with the series of the recurrence
    for n, q in ((3, 2), (2, 4)):
        p, J = (n - 1) * q, (n + 1) * q
        eps = Fr(1, 2)
        e = sp.Rational(eps.numerator, eps.denominator)
        a = poly_from_roots([-(q + i) * eps for i in range(p)])[:p]
        R, h, _ = make_ring()
        order = J + 4
        fam = Family(n, q, R, h, a, order)
        x = sp.exp(-e * z) - 1
        f_closed = (-e) ** q * (1 + x) ** q * x ** (-q)
        fh_closed = (-e) ** q * (1 + x) ** q * x ** (n * q)
        eta_closed = q * e ** 2 * (-e) ** (n * q) * (1 + x) ** (n * q) * x ** (-n * q - 1)
        etah_closed = e ** 2 * (-e) ** (n * q) * (1 + x) ** (n * q) * x ** (q - 2) * (
            J * (J + 1) + (2 * J ** 2 + J + n * q) * x + J ** 2 * x ** 2)

        def coefficients(expr, shift, count):
            series = sp.series(sp.expand(expr * z ** shift), z, 0, count + 1).removeO()
            return [sp.nsimplify(series.coeff(z, k)) for k in range(count + 1)]

        def h_part(poly, degree):
            return sum((to_fraction(c) for mon, c in poly.terms() if mon[0] == degree), Fr(0))

        # f and eps^J f_h at h' = 0; the coefficients compared do not depend on h
        got = coefficients(f_closed, q, 5)
        assert all(sp.Rational(h_part(fam.b[k], 0).numerator, h_part(fam.b[k], 0).denominator) == got[k] for k in range(6))
        got = coefficients(fh_closed / e ** J, -n * q, 3)
        assert all(sp.Rational(h_part(fam.b[J + k], 1).numerator, h_part(fam.b[J + k], 1).denominator) == got[k] for k in range(4))
        # eta = H_n(f) and eps^J eta_h at h' = 0, again in the range where they do not depend on h
        from series_tools import s_mul, s_pow, seed
        f = fam.series()
        Hn = s_mul(s_pow(f, n - 2, R), seed(f, q))
        got = coefficients(eta_closed, n * q + 1, 3)
        for k in range(4):
            value = h_part(Hn.get(-n * q - 1 + k, R(0)), 0)
            assert sp.Rational(value.numerator, value.denominator) == got[k]
        assert all(h_part(c, 0) == 0 for e_, c in Hn.items() if e_ < -n * q - 1)
        got = coefficients(etah_closed / e ** J, -(q - 2), 3)
        for k in range(4):
            value = h_part(Hn.get(q - 2 + k, R(0)), 1)
            assert sp.Rational(value.numerator, value.denominator) == got[k]
    return "explicit family: closed forms and the values of A_0, A_1"


def first_odd_coefficient(n, q, j):
    """c_{j-1} from the formula of the appendix (j odd, 1 <= j <= p - 1)."""
    p, J = (n - 1) * q, (n + 1) * q
    nK = rising(q + 1, p)
    indicial = lambda t: falling(t - q, p) - nK
    assert indicial(j) < 0 and indicial(J + j) > 0
    beta = Fr(rising(q, p - j), indicial(j))
    gamma = (Fr(-factorial(n * q), factorial(q + j)) + (n - 1) * Fr(factorial(n * q), factorial(q)) * beta) / indicial(J + j)
    assert beta < 0 and gamma < 0
    s = (n - 2) * (J * (J + 1) + j * (j + 1)) + (J - j) * (J - j + 1) - 2 * n * j
    value = (-comb(q + j - 2, j - 1) * (s * beta + (J + j) * (J + j + 1) * gamma)
             + J * (J + 1) * comb(q - 2, j - 1) * beta
             - j * (j + 1) * comb(n * q, j - 1) * beta)
    return (-1) ** q * value, s


def check_first_odd_coefficient(rng):
    # formula against the series of D^p + a D^(p-j), to first order in a
    for n, q in CASES:
        p = (n - 1) * q
        for j in range(1, p, 2):
            R, h, extra = make_ring(["a"])
            a = [R(0)] * p
            a[p - j] = extra["a"]
            fam = Family(n, q, R, h, a, max(needed_order(n, q, j - 1), (n + 1) * q + j + 1))
            phi = phi_u_Hn(fam, j - 1)
            slope = sum((to_fraction(c) for mon, c in phi.get(j - 1, R(0)).terms() if mon == (1, 1)), Fr(0))
            assert slope == first_odd_coefficient(n, q, j)[0], (n, q, j)
    # with random operators: if a_{p-i} = 0 for odd i < j then A_{j-1} = c_{j-1} a_{p-j}
    for n, q in CASES:
        p = (n - 1) * q
        for j in range(3, p, 2):
            a = random_operator(p, rng)
            for i in range(1, j, 2):
                a[p - i] = Fr(0)
            a[p - j] = Fr(rng.choice([-3, -2, -1, 1, 2, 3]), rng.randint(1, 4))
            terms, _ = linear_terms(n, q, a, j - 1)
            assert all(terms[i][0] == 0 for i in range(0, j - 1, 2))
            assert terms[j - 1][0] == first_odd_coefficient(n, q, j)[0] * a[p - j]
    # sign and the inequality of the proof, for all n in {2, 3} and even p <= 120
    count = 0
    for n in (2, 3):
        for q in range(2, 121):
            p, J = (n - 1) * q, (n + 1) * q
            if p % 2 or p < 4 or p > 120:
                continue
            c0, _ = constants(n, q)
            assert first_odd_coefficient(n, q, 1)[0] == c0
            for j in range(3, p, 2):
                value, s = first_odd_coefficient(n, q, j)
                assert (-1) ** q * value > 0
                assert s > 0 and comb(q + j - 2, j - 1) * s >= J * (J + 1) * comb(q - 2, j - 1)
                count += 1
    return "c_{j-1}: formula = series; (-1)^q c_{j-1} > 0 in %d cases (n = 2, 3; p <= 120)" % count


def check_swift_hohenberg_waves():
    """Waves with rational parameters and sigma != 0, at rate 2, pole at i*pi/2."""
    t = sp.Symbol("t")
    csch, coth = 1 / sp.sinh(t), sp.cosh(t) / sp.sinh(t)
    waves = [("twisted", Fr(1), Fr(0)), ("twisted", Fr(-1), Fr(0))]
    waves += [("untwisted", xi, -xi / 2) for xi in (Fr(4), Fr(2), Fr(-2), Fr(2, 7))]
    waves += [("untwisted", xi, xi / 2) for xi in (Fr(-4), Fr(2), Fr(-2), Fr(-2, 7))]
    waves += [("untwisted", Fr(2), Fr(0)), ("untwisted", Fr(-2), Fr(0))]
    operators = {}
    for kind, xi, be in waves:
        X, Bq = sp.Rational(xi.numerator, xi.denominator), sp.Rational(be.numerator, be.denominator)
        if kind == "twisted":
            a = [-14 * xi ** 4 - 80 * xi ** 2 - 11, -2 * xi * (7 * xi ** 2 + 10), 46 * xi ** 2 + 10, -14 * xi]
            u = csch * coth + X * csch                        # -2i w_1 at the pole
        else:
            a = [1440 * be ** 2 - 120 * be * xi ** 2 + 480 * be - 14 * xi ** 4 + 160 * xi ** 2 + 64,
                 -2 * xi * (7 * xi ** 2 - 300 * be - 20), 46 * xi ** 2 - 120 * be - 20, -14 * xi]
            u = csch ** 2 + X * coth - 2 * Bq                 # -2 w_2 at the pole
        series = sp.series(u * t ** 2, t, 0, 13).removeO()
        wave = [Fr(int(sp.numer(c)), int(sp.denom(c))) for c in (sp.nsimplify(series.coeff(t, k)) for k in range(13))]
        terms, fam = linear_terms(3, 2, a, 6)
        assert fam.F_J == 0
        hw = wave[8]
        for k in range(13):
            value = sum((to_fraction(c) * hw ** mon[0] for mon, c in fam.b[k].terms()), Fr(0))
            assert value == wave[k]                           # the wave is f(z; h) with h = hw
        assert terms[0][0] != 0 and terms[1][0] != 0          # a_3 = -14 xi != 0
        assert all(terms[i][0] * hw + terms[i][1] == 0 for i in range(1, 7))
        lam2 = 2 / a[2]                                       # lambda^2 a_2 = 2
        key = (lam2 * a[3] ** 2, lam2 * a[1] / a[3], -lam2 ** 2 * a[0])   # (sigma^2, s/sigma, alpha)
        operators.setdefault(key, set()).add((kind, abs(xi), abs(be)))
    assert len(operators) == 6 and all(len(v) == 1 for v in operators.values())
    assert sum(1 for key in operators if key[0] == Fr(392, 11)) == 2
    return "12 Swift-Hohenberg waves with sigma != 0: h = -B_1/A_1; 6 operators up to reflection, one wave each"


def check_even_sextic():
    """n = 3, P = D^6 + a0: one wave although P is even."""
    n, q, J = 3, 3, 12
    R, h, extra = make_ring(["a0"])
    a0 = extra["a0"]
    fam = Family(n, q, R, h, [a0] + [R(0)] * 5, needed_order(n, q, J - 1))
    assert fam.F_J == 0
    phi = phi_u_Hn(fam, J - 1)
    H, A = sp.symbols("h a0")
    slopes = {}
    for i in range(J - 1):
        poly = sp.Poly(phi.get(i, R(0)).as_expr(), H)
        assert poly.degree() <= 1
        slopes[i] = (poly.coeff_monomial(H), poly.coeff_monomial(1))
    assert all(slopes[i] == (0, 0) for i in range(J - 1) if i != 5)
    assert sp.simplify(slopes[5][0] + sp.Rational(13, 888) * A) == 0
    # the wave u = -wp'(z; 0, g3)/2 with g3 = -a0/4320:  u = z^-3 - (g3/14) z^3 - (5 g3^2/10192) z^9 + ...
    g3 = -A / 4320
    assert sp.simplify(fam.b[6].as_expr() + g3 / 14) == 0
    h_wave = -5 * g3 ** 2 / 10192
    assert sp.simplify(slopes[5][0] * h_wave + slopes[5][1]) == 0
    # u = -wp'/2 solves u^(6) + a0 u = 20160 u^3: with X = wp, X'^2 = 4X^3 - g3, X'' = 6X^2,
    # (X')^(6) = (20160 X^3 - 720 g3) X' and (X')^2/4 = X^3 - g3/4
    X = sp.Symbol("X")
    assert sp.expand((20160 * X ** 3 - 720 * g3) + A - 20160 * (X ** 3 - g3 / 4)) == 0
    # quadratic selector: two roots, only one of them belongs to a wave
    Q = sp.Poly(phi[J - 1].as_expr(), H)
    assert Q.degree() == 2 and sp.simplify(Q.eval(h_wave)) == 0
    other = [r for r in sp.roots(Q) if sp.simplify(r - h_wave) != 0]
    assert len(other) == 1 and sp.simplify(slopes[5][0] * other[0] + slopes[5][1]) != 0
    return "n = 3, P = D^6 + a0: A_5 = -13 a0/888; one wave, -wp'(z; 0, -a0/4320)/2"


def main():
    rng = random.Random(20260930)
    report = {}
    for name, check in (("homogeneity", lambda: check_homogeneity(rng)),
                        ("even_operators", lambda: check_even_operators(rng)),
                        ("residue_identities", lambda: check_residue_identities(rng)),
                        ("A0_A1", lambda: check_A0_A1(rng)),
                        ("explicit_family", check_explicit_family),
                        ("first_odd_coefficient", lambda: check_first_odd_coefficient(rng)),
                        ("swift_hohenberg", check_swift_hohenberg_waves),
                        ("even_sextic", check_even_sextic)):
        report[name] = check()
        print("PASS", report[name], flush=True)
    report["examples_of_c"] = {f"n={n},q={q}": [str(first_odd_coefficient(n, q, j)[0]) for j in range(1, (n - 1) * q, 2)]
                               for n, q in CASES}
    target = Path(__file__).with_name("linear_terms_checks.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {target.name}.")


if __name__ == "__main__":
    main()
