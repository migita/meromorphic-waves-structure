#!/usr/bin/env python3
"""Exact Laurent series at a pole for P(D)u = K u^n (module for the checks).

The series f(z;h) = z^(-q) * sum_j b_j(h) z^j of the paper is built from the
recurrence at all indices j != J, with b_J = h, for ANY monic operator P of
even degree p = (n-1)q; the compatibility value F_J is recorded. Coefficients
live in a polynomial ring over QQ whose first generator is h.

Used by check_linear_terms.py and check_more_examples.py. It shares no code
with check_bracket.py and check_bracket_audit.py.
"""

from fractions import Fraction
from math import factorial

from sympy import QQ
from sympy.polys.rings import ring


def falling(x, k):
    result = 1
    for i in range(k):
        result *= x - i
    return result


def rising(x, k):
    result = 1
    for i in range(k):
        result *= x + i
    return result


def make_ring(extra=()):
    """Polynomial ring QQ[h, *extra]; returns (R, h, dict of the extra generators)."""
    names = ["h"] + list(extra)
    objects = ring(names, QQ)
    R, gens = objects[0], objects[1:]
    return R, gens[0], dict(zip(extra, gens[1:]))


def to_ring(R, value):
    if isinstance(value, Fraction):
        return R(QQ(value.numerator, value.denominator))
    if isinstance(value, int):
        return R(value)
    return value


def to_fraction(c):
    return Fraction(int(c.numerator), int(c.denominator))


class Family:
    """b_0..b_M of f(z;h) for P = D^p + a[p-1] D^(p-1) + ... + a[0]."""

    def __init__(self, n, q, R, h, a, M):
        self.n, self.q, self.R, self.h = n, q, R, h
        p = (n - 1) * q
        assert len(a) == p
        self.p, self.J = p, (n + 1) * q
        self.K = (-1) ** p * rising(q, p)
        K = self.K
        coeffs = [to_ring(R, x) for x in a] + [R(1)]
        b = [R(1)]
        self.F_J = None
        for j in range(1, M + 1):
            nonlinear = self._power_coefficient(b, n, j)      # [z^j] (b_0 + ... + b_{j-1} z^{j-1})^n
            lower = R(0)
            for k in range(p):
                i = j - p + k
                if i >= 0:
                    lower += coeffs[k] * falling(i - q, k) * b[i]
            rhs = nonlinear * K - lower                      # F_j of the paper
            indicial = falling(j - q, p) - n * K
            if indicial == 0:
                assert j == self.J
                self.F_J = rhs
                b.append(h)
            else:
                b.append(rhs * QQ(1, indicial))
        self.b = b

    def _power_coefficient(self, b, n, j):
        R = self.R
        base = list(b[:j]) + [R(0)]
        current = [R(1)] + [R(0)] * j
        for _ in range(n):
            new = [R(0)] * (j + 1)
            for i, ci in enumerate(current):
                if ci == 0:
                    continue
                for k in range(j + 1 - i):
                    if base[k] != 0:
                        new[i + k] += ci * base[k]
            current = new
        return current[j]

    def series(self):
        return {j - self.q: c for j, c in enumerate(self.b) if c != 0}


# ---- truncated Laurent series as dictionaries {exponent: coefficient} ----

def s_mul(A, B):
    out = {}
    for ea, ca in A.items():
        for eb, cb in B.items():
            out[ea + eb] = out.get(ea + eb, 0) + ca * cb
    return {e: c for e, c in out.items() if c != 0}


def s_add(A, B, factor=1):
    out = dict(A)
    for e, c in B.items():
        out[e] = out.get(e, 0) + c * factor
    return {e: c for e, c in out.items() if c != 0}


def s_scale(A, factor):
    return {e: c * factor for e, c in A.items() if c != 0}


def s_diff(A, k=1):
    out = {}
    for e, c in A.items():
        factor = falling(e, k)
        if factor != 0:
            out[e - k] = c * factor
    return out


def s_pow(A, n, R):
    out = {0: R(1)}
    for _ in range(n):
        out = s_mul(out, A)
    return out


def seed(f, q):
    """H(f) = f f'' - (q+1)/q (f')^2."""
    f1, f2 = s_diff(f, 1), s_diff(f, 2)
    return s_add(s_mul(f, f2), s_scale(s_mul(f1, f1), -QQ(q + 1, q)))


def hat_apply(principal, S, top):
    """Apply hat(D) to S, where principal = {-r: coefficient} is a principal part."""
    out = {}
    for e_pp, a in principal.items():
        r = -e_pp
        sign, fact = (-1) ** (r - 1), factorial(r - 1)
        for e, c in S.items():
            target = e - (r - 1)
            if target > top:
                continue
            factor = falling(e, r - 1)
            if factor != 0:
                out[target] = out.get(target, 0) + a * c * QQ(sign * factor, fact)
    return {e: c for e, c in out.items() if c != 0}


def bracket(E, F, top):
    """hat E(D) F - hat F(D) E up to z^top; the principal parts must cancel."""
    Epp = {e: c for e, c in E.items() if e < 0}
    Fpp = {e: c for e, c in F.items() if e < 0}
    phi = s_add(hat_apply(Epp, F, top), hat_apply(Fpp, E, top), -1)
    assert all(e >= 0 for e in phi)
    return phi


def phi_u_Hn(family, top):
    """phi(z;h) for the pair E = u, F = H_n = u^(n-2) H(u), exact up to z^top."""
    n, q, R = family.n, family.q, family.R
    # exactness: knowing b_0..b_M, F is exact to exponent M - nq - 2 and hat E lowers by q - 1
    assert len(family.b) - 1 - n * q - 2 - (q - 1) >= top
    f = family.series()
    Hn = s_mul(s_pow(f, n - 2, R), seed(f, q))
    return bracket(f, Hn, top)


def needed_order(n, q, top):
    """Number of Laurent coefficients needed for phi up to z^top (pair (u, H_n))."""
    return top + n * q + 2 + (q - 1)


def linear_terms(n, q, a, top):
    """[(A_i, B_i) for i <= top] and the family, for numerical coefficients a (Fractions)."""
    R, h, _ = make_ring()
    fam = Family(n, q, R, h, a, max(needed_order(n, q, top), (n + 1) * q + 1))
    phi = phi_u_Hn(fam, top)
    out = []
    for i in range(top + 1):
        terms = {mon[0]: c for mon, c in phi.get(i, R(0)).terms()}
        assert set(terms) <= {0, 1}, "coefficients below z^(J-1) are affine in h"
        out.append((to_fraction(terms.get(1, QQ(0))), to_fraction(terms.get(0, QQ(0)))))
    return out, fam


def poly_from_roots(roots):
    """Coefficients [c_0, ..., c_d] of prod (rho - root)."""
    coeffs = [Fraction(1)]
    for root in roots:
        new = [Fraction(0)] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            new[i + 1] += c
            new[i] -= root * c
        coeffs = new
    return coeffs
