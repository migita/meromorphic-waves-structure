#!/usr/bin/env python3
"""Independent exact audit of canonical principal-part cancellation.

Only this directory is written.  The check uses exact Laurent recurrences,
not the older regularized-family implementation or its nullspace.  The
uniform identities are symbolic; the finite examples exercise both endpoint
branches, lower odd drift, vanished principal data, and the excluded order.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


HVAR = sp.Symbol("h")
XI = sp.Symbol("xi")
X = sp.Symbol("x")


def falling(x, degree):
    return sp.prod(x - j for j in range(degree))


def conv(a, b, maximum):
    return [
        sp.expand(sum(a[j] * b[k - j] for j in range(k + 1)))
        for k in range(maximum + 1)
    ]


def power(a, exponent, maximum):
    out = [sp.Integer(1)] + [sp.Integer(0)] * maximum
    for _ in range(exponent):
        out = conv(out, a, maximum)
    return out


def uniform_checks():
    n, q, j, r = sp.symbols("n q j r", integer=True, positive=True)
    J = (n + 1) * q
    c2, t, eps, beta1, b1 = sp.symbols("c2 t eps beta1 b1")

    # Principal-part transform: the coefficient identity for D(z^-r).
    assert sp.simplify(
        (-r) * (-1) ** r / sp.factorial(r)
        - (-1) ** (r - 1) / sp.factorial(r - 1)
    ) == 0

    f = 1 + t + c2 * t**2
    df = sp.expand(-q * f + J * t * sp.diff(f, t))
    ddf = sp.expand((-q - 1) * df + J * t * sp.diff(df, t))
    H = sp.expand(f * ddf - (q + 1) * df**2 / q)
    multiplier = 1 + (n - 2) * t + ((n - 2) * c2 + (n - 2) * (n - 3) / 2) * t**2
    S = sp.expand(multiplier * H)
    kappa = J * (2 * (2 * J + 1) * c2 + (n - 2) * J - 2)
    assert S.coeff(t, 0) == 0
    assert sp.expand(S.coeff(t, 1) - J * (J + 1)) == 0
    assert sp.expand(S.coeff(t, 2) - kappa) == 0

    # Independently compute the generic one-mode first symbol.
    f0 = 1 + eps
    f1 = -q + (j - q) * eps
    f2 = q * (q + 1) + (j - q) * (j - q - 1) * eps
    first = sp.expand((1 + (n - 2) * eps) * (f0 * f2 - (q + 1) * f1**2 / q))
    assert sp.factor(first.coeff(eps, 1)) == j * (j + 1)

    w = n * q + 2
    first_h2_S = (n + 2) * q - 2
    assert sp.expand(first_h2_S - (q - 1) - (J - 1)) == 0
    assert sp.expand((2 * J - q) - n * q - J) == 0
    assert sp.expand(first_h2_S - q - (J - 2)) == 0
    assert sp.expand(w - 2 - n * q) == 0  # deg pp(S) transform <= nq.

    # The generic A0 uses only the first pre/post-resonance coefficients.
    mixed_H = 2 + J * (J + 1) - 2 * (q + 1) * J / q
    mixed_S = sp.expand(mixed_H + (n - 2) * (2 + J * (J + 1)))
    L1 = (J + 1) * (J + 2) * beta1 + mixed_S * b1
    A0_without_sign = sp.factor(L1 - J * (J + 1) * b1 + 2 * b1)
    generic_A0 = (J + 1) * ((J + 2) * beta1 + ((n - 2) * J - 2) * b1)
    assert sp.expand(A0_without_sign - generic_A0) == 0
    expected_A0 = 2 * (n + 1) * (n * q + 1) * (J + 1) * b1
    assert sp.expand(generic_A0.subs(beta1, (n + 2) * b1) - expected_A0) == 0

    # Normalize the recurrence by K=(q)_p to check the two coefficients.
    alpha = sp.Symbol("alpha")
    I1_over_K = sp.factor((q - 1) / (n * q - 1) - n)
    assert sp.factor(I1_over_K + (n - 1) * (J - 1) / (n * q - 1)) == 0
    solved_b1 = sp.factor(alpha / ((n * q - 1) * I1_over_K))
    assert sp.factor(solved_b1 + alpha / ((n - 1) * (J - 1))) == 0
    solved_beta1 = sp.factor(((n - 1) * (q + 1) * b1 - alpha) / ((n - 1) * q))
    assert sp.factor(solved_beta1.subs(alpha, -(n - 1) * (J - 1) * b1) - (n + 2) * b1) == 0

    # Affine bracket protection uses an exponent one step before all rivals.
    p = sp.Symbol("p", integer=True, positive=True)
    assert sp.expand((p - 2) - (2 * q - 1) - (p - 2 * q - 1)) == 0
    assert sp.expand(p - 2 * q - (p - 2 * q - 1)) == 1
    return {
        "transform_derivative_identity": "T(Df)=xi*T(f)",
        "S_linear_symbol": "j*(j+1)",
        "S_quadratic_symbol": str(sp.factor(kappa)),
        "deg_a": "q-1",
        "deg_b_upper_bound": "n*q",
        "quadratic_first_exponent_G": "J-1",
        "quadratic_first_exponent_Gprime": "J-2",
        "A0": "(-1)^(q-1)*2*(n+1)*(n*q+1)*(J+1)*b1",
        "b1": "-alpha_(p-1)/((n-1)*(J-1))",
        "beta1": "(n+2)*b1",
        "affine_first_exponent": "p-2*q-1",
    }


def recurrence(n, q, polynomial, maximum):
    p = (n - 1) * q
    assert p % 2 == 0
    J = (n + 1) * q
    polynomial = sp.Poly(polynomial, X)
    assert polynomial.degree() == p and polynomial.LC() == 1
    K = sp.rf(q, p)
    b = [sp.Integer(1)] + [sp.Integer(0)] * maximum
    for j in range(1, maximum + 1):
        nonlinear = power(b[: j + 1], n, j)[j]
        lower = sum(
            polynomial.nth(r) * falling(j - p + r - q, r) * b[j - p + r]
            for r in range(p)
            if j - p + r >= 0
        )
        numerator = sp.expand(K * nonlinear - lower)
        divisor = falling(j - q, p) - n * K
        if j == J:
            assert divisor == 0 and numerator == 0, (n, q, polynomial, numerator)
            b[j] = HVAR
        else:
            assert divisor != 0
            b[j] = sp.expand(numerator / divisor)
        assert sp.Poly(b[j], HVAR).degree() <= j // J
    return b


def seed(b, n, q):
    maximum = len(b) - 1
    up = [b[j] * (j - q) for j in range(maximum + 1)]
    upp = [b[j] * falling(j - q, 2) for j in range(maximum + 1)]
    H = [sp.expand(a - sp.Rational(q + 1, q) * d) for a, d in zip(conv(b, upp, maximum), conv(up, up, maximum))]
    return H, conv(power(b, n - 2, maximum), H, maximum)


def transform(coefficients, pole_weight):
    return {
        pole_order - 1: sp.expand(coefficients[pole_weight - pole_order] * (-1) ** (pole_order - 1) / sp.factorial(pole_order - 1))
        for pole_order in range(1, pole_weight + 1)
        if coefficients[pole_weight - pole_order] != 0
    }


def poly_string(polynomial):
    return str(sp.factor(sum(c * XI**r for r, c in polynomial.items())))


def coefficient_after_operator(operator, coefficients, pole_weight, exponent):
    return sp.expand(sum(
        constant * falling(exponent + derivative, derivative) * coefficients[index]
        for derivative, constant in operator.items()
        if 0 <= (index := pole_weight + exponent + derivative) < len(coefficients)
    ))


def canonical_bracket(left, left_weight, right, right_weight, largest_exponent):
    a, b = transform(left, left_weight), transform(right, right_weight)
    assert all(not c.has(HVAR) for c in (*a.values(), *b.values()))
    assigned_weight = max(
        right_weight + max(a, default=-right_weight),
        left_weight + max(b, default=-left_weight),
    )
    out = {}
    for e in range(-assigned_weight, largest_exponent + 1):
        out[e] = sp.expand(
            coefficient_after_operator(a, right, right_weight, e)
            - coefficient_after_operator(b, left, left_weight, e)
        )
        if e < 0:
            assert out[e] == 0
    return a, b, out


def actual_monomial_h(n, q):
    p, m, J = (n - 1) * q, n - 1, (n + 1) * q
    # z^q u = exp(pz)*(mz/(exp(mz)-1))^q.
    bernoulli = [sp.bernoulli(j, 0) * m**j / sp.factorial(j) for j in range(J + 1)]
    exponential = [sp.Rational(p**j, sp.factorial(j)) for j in range(J + 1)]
    h = conv(exponential, power(bernoulli, q, J), J)[J]

    t = sp.Symbol("t")
    profile = m**q * t**q / (t - 1) ** q
    result = profile
    for j in range(p):
        result = sp.factor(m * t * sp.diff(result, t) - (p + m * j) * result)
    assert sp.factor(result - sp.rf(q, p) * profile**n) == 0
    return h


def sharp_profiles():
    """Verify both realizations of the two roots for the sharp operator."""
    W, g2, g3 = sp.symbols("W g2 g3")
    fourth = 120 * W**3 - 18 * g2 * W - 12 * g3
    assert sp.expand((fourth - 108 * W - 120 * W**3).subs({g2: -6, g3: 0})) == 0
    h1 = sp.Rational((-6) ** 2, 1200)
    assert h1 == sp.Rational(3, 100)

    U, V = sp.symbols("U V")
    relation = V**2 - U**4 + 1
    expression = -V
    for _ in range(4):
        expression = sp.rem(
            sp.expand(V * sp.diff(expression, U) + 2 * U**3 * sp.diff(expression, V)),
            relation, V,
        )
    assert sp.rem(sp.expand(expression - 108 * (-V) - 120 * (-V) ** 3), relation, V) == 0
    z = sp.Symbol("z")
    g = z ** (-1) * sp.series(sp.sqrt(1 + z**4 / 5 + z**8 / 75), z, 0, 9).removeO()
    h2 = sp.expand(-sp.diff(g, z)).coeff(z, 6)
    assert h2 == -sp.Rational(7, 600)
    return [str(h1), str(h2)]


def affine_audit(n, q, polynomial, coefficients=None, expected_h=None):
    p, J = (n - 1) * q, (n + 1) * q
    assert p > 2 * q + 1
    b = coefficients if coefficients is not None else recurrence(n, q, polynomial, 2 * J)
    H, _ = seed(b, n, q)
    E = conv(b, b, len(b) - 1)
    e = p - 2 * q - 1
    a, d, G = canonical_bracket(E, 2 * q, H, 2 * q + 2, e)
    assert max(a) == 2 * q - 1
    assert max(d, default=-1) <= 2 * q
    assert not G[0].has(HVAR)
    detector = sp.factor(G[e])
    leading = -J * (J + 1) * falling(p - 2, 2 * q - 1) / sp.factorial(2 * q - 1)
    assert sp.Poly(detector, HVAR).degree() == 1
    assert sp.expand(detector).coeff(HVAR) == leading != 0
    if expected_h is not None:
        assert detector.subs(HVAR, expected_h) == 0
    return {
        "n": n, "q": q, "p": p,
        "operator": str(sp.expand(polynomial)),
        "a": poly_string(a), "b": poly_string(d),
        "constant": str(G[0]),
        "detector_exponent": e,
        "detector": str(detector),
        "actual_h": str(expected_h) if expected_h is not None else None,
    }


def quadratic_audit(name, n, q, polynomial, monomial=False):
    p, J = (n - 1) * q, (n + 1) * q
    assert q >= 2
    b = recurrence(n, q, polynomial, 2 * J)
    _, S = seed(b, n, q)
    a, d, G = canonical_bracket(b, q, S, n * q + 2, J - 1)
    assert max(a) == q - 1
    assert a[q - 1] == sp.Rational((-1) ** (q - 1), sp.factorial(q - 1))
    assert max(d, default=-1) <= n * q
    assert sp.Poly(G[0], HVAR).degree() <= 1
    assert sp.Poly(G[1], HVAR).degree() <= 1
    A0, A1 = sp.expand(G[0]).coeff(HVAR), sp.expand(G[1]).coeff(HVAR)
    expected_A0 = (-1) ** (q - 1) * 2 * (n + 1) * (n * q + 1) * (J + 1) * b[1]
    assert sp.expand(A0 - expected_A0) == 0
    assert sp.expand(b[J + 1]).coeff(HVAR) == (n + 2) * b[1]
    assert b[1] == -sp.Poly(polynomial, X).nth(p - 1) / ((n - 1) * (J - 1))

    R = sp.rf((n + 2) * q + 1, p) / sp.rf(q, p)
    c2 = sp.binomial(n, 2) / (R - n)
    kappa = sp.cancel(J * (2 * (2 * J + 1) * c2 + (n - 2) * J - 2))
    leading_G = sp.cancel(a[q - 1] * kappa * falling((n + 2) * q - 2, q - 1))
    assert sp.expand(G[J - 1]).coeff(HVAR, 2) == leading_G

    if A0 == 0:
        branch = "G"
        exponent = J - 1
        constant, detector, expected = G[0], G[exponent], leading_G
    else:
        branch = "A0*Gprime-A1*G"
        exponent = J - 2
        constant = sp.expand(A0 * G[1] - A1 * G[0])
        detector = sp.expand(A0 * (exponent + 1) * G[exponent + 1] - A1 * G[exponent])
        expected = A0 * (J - 1) * leading_G
    assert not constant.has(HVAR)
    assert sp.expand(detector).coeff(HVAR, 2) == expected
    assert sp.Poly(detector, HVAR).degree() <= 2
    if (n, q) != (2, 2):
        assert expected != 0
        assert sp.Poly(detector, HVAR).degree() == 2
    else:
        assert kappa == 0 and expected == 0

    actual_h = actual_monomial_h(n, q) if monomial else None
    if monomial:
        assert sp.factor(detector.subs(HVAR, actual_h)) == 0
    result = {
        "name": name, "n": n, "q": q, "p": p,
        "operator": str(sp.expand(polynomial)),
        "a": poly_string(a), "b": poly_string(d),
        "b1": str(b[1]), "A0": str(A0), "A1": str(A1),
        "kappa": str(kappa), "completion": branch,
        "constant_after_completion": str(constant),
        "detector_exponent": exponent,
        "detector": str(sp.factor(detector)),
        "actual_h": str(actual_h) if actual_h is not None else None,
        "all_principal_coefficients_zero": True,
    }

    if name == "sharp_cubic":
        assert a == {1: sp.Integer(-1)}
        assert d == {3: sp.Integer(1)}
        sharp = -sp.Rational(126, 18125) * (100 * HVAR - 3) * (600 * HVAR + 7)
        assert sp.expand(detector - sharp) == 0
        assert set(sp.solve(detector, HVAR)) == {sp.Rational(3, 100), -sp.Rational(7, 600)}
        result["canonical_identity"] = "G = -D(S+u'')"
        result["both_realized_h_roots"] = sharp_profiles()
    if name == "excluded_quadratic_order_two":
        assert not d and all(coefficient == 0 for coefficient in G.values())
        # The identity is exact on u''=6u^2, not merely truncated:
        # D(H)=u*u'''-2u'*u'' and u'''=12u*u'.
        U, U1 = sp.symbols("U U1")
        assert sp.expand(U * (12 * U * U1) - 2 * U1 * (6 * U**2)) == 0
        result["exact_zero_identity"] = "G=-D(H)=0 on u''=6u^2"
    if p > 2 * q + 1:
        result["affine_bracket"] = affine_audit(n, q, polynomial, b, actual_h)
    return result


def monomial_operator(n, q):
    p, m = (n - 1) * q, n - 1
    return sp.expand(sp.prod(X - p - m * j for j in range(p)))


def main():
    report = {"uniform": uniform_checks(), "quadratic_audits": []}
    print("PASS uniform transform, first/second symbols, response positions, and A0 formula.", flush=True)
    cases = [
        ("sharp_cubic", 3, 2, X**4 - 108, False),
        ("mixed_cubic_top_drift", 3, 2, monomial_operator(3, 2), True),
        ("mixed_cubic_lower_drift", 3, 2, X**4 + X + 2, False),
        ("degenerate_pure_cubic", 3, 2, X**4, False),
        ("mixed_cubic_q3", 3, 3, monomial_operator(3, 3), True),
        ("mixed_quadratic_q4", 2, 4, monomial_operator(2, 4), True),
        ("mixed_quartic_q2", 4, 2, monomial_operator(4, 2), True),
        ("excluded_quadratic_order_two", 2, 2, X**2, False),
    ]
    for case in cases:
        report["quadratic_audits"].append(quadratic_audit(*case))
        print(f"PASS {case[0]}: principal cancellation, completion, and exact detector.", flush=True)
    n, q = 5, 1
    report["simple_pole_affine_audit"] = affine_audit(n, q, monomial_operator(n, q), expected_h=actual_monomial_h(n, q))
    print("PASS simple-pole affine bracket at n=5,q=1,p=4.", flush=True)
    target = Path(__file__).with_name("check_bracket_audit_certificate.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {target.name}", flush=True)


if __name__ == "__main__":
    main()
