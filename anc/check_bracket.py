#!/usr/bin/env python3
"""Exact checks for the canonical principal-part bracket proofs.

All nonvanishing arguments are symbolic identities or exact inequalities.
Arbitrary-jet checks audit cancellation without choosing an operator.  Full
recurrence checks independently exercise both constant-slope branches, the
zero-transform case, the excluded second-order detector, and actual fronts.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


XI, H = sp.symbols("xi h")


def falling(x, degree: int):
    return sp.prod(x - i for i in range(degree))


def convolution(left, right, maximum: int):
    return [
        sp.expand(sum(left[i] * right[j - i] for i in range(max(0, j - len(right) + 1), min(j, len(left) - 1) + 1)))
        for j in range(maximum + 1)
    ]


def power(series, exponent: int, maximum: int):
    result = [sp.Integer(1)] + [sp.Integer(0)] * maximum
    for _ in range(exponent):
        result = convolution(result, series, maximum)
    return result


def seed_coefficients(b, n: int, q: int, maximum: int):
    jets = [[sp.expand(b[j] * falling(j - q, r)) for j in range(maximum + 1)] for r in range(3)]
    first = convolution(jets[0], jets[2], maximum)
    second = convolution(jets[1], jets[1], maximum)
    quadratic = [sp.expand(a - sp.Rational(q + 1, q) * c) for a, c in zip(first, second)]
    return convolution(power(b, n - 2, maximum), quadratic, maximum)


def laurent(coefficients, initial_exponent: int):
    return {j + initial_exponent: coefficient for j, coefficient in enumerate(coefficients) if coefficient != 0}


def differentiate(series, degree: int = 1):
    return {
        exponent - degree: sp.expand(coefficient * falling(exponent, degree))
        for exponent, coefficient in series.items()
        if falling(exponent, degree) != 0
    }


def transform(series):
    return sp.Poly(sp.expand(sum(coefficient * (-XI) ** (-exponent - 1) / sp.factorial(-exponent - 1) for exponent, coefficient in series.items() if exponent < 0)), XI)


def act(operator, series, largest_power: int):
    result = {}
    for (degree,), multiplier in operator.terms():
        if multiplier == 0:
            continue
        for exponent, coefficient in series.items():
            target = exponent - degree
            factor = falling(exponent, degree)
            if target <= largest_power and factor != 0:
                result[target] = result.get(target, 0) + multiplier * coefficient * factor
    return {exponent: sp.expand(coefficient) for exponent, coefficient in result.items()}


def bracket(E, F, largest_power: int):
    a, b = transform(E), transform(F)
    left, right = act(a, F, largest_power), act(b, E, largest_power)
    result = {e: sp.expand(left.get(e, 0) - right.get(e, 0)) for e in set(left) | set(right)}
    assert all(value == 0 for e, value in result.items() if e < 0)
    return a, b, {e: value for e, value in result.items() if value != 0}


def uniform_checks():
    r = sp.Symbol("r", integer=True, positive=True)
    assert sp.simplify(r / sp.factorial(r) - 1 / sp.factorial(r - 1)) == 0
    arbitrary = {e: sp.Symbol(f"c_{'m' if e < 0 else 'p'}{abs(e)}") for e in range(-7, 5)}
    assert sp.expand(transform(differentiate(arbitrary)).as_expr() - XI * transform(arbitrary).as_expr()) == 0
    E = {e: sp.Symbol(f"e_{'m' if e < 0 else 'p'}{abs(e)}") for e in range(-3, 6)}
    F = {e: sp.Symbol(f"f_{'m' if e < 0 else 'p'}{abs(e)}") for e in range(-4, 6)}
    bracket(E, F, 0)

    n, q, d = sp.symbols("n q d", integer=True, positive=True)
    j, epsilon, t, c2, b1, d1 = sp.symbols("j epsilon t c2 b1 d1")
    p, J = (n - 1) * q, (n + 1) * q
    f0, f1, f2 = 1 + epsilon, -q + (j - q) * epsilon, q * (q + 1) + (j - q) * (j - q - 1) * epsilon
    lin = sp.expand((1 + (n - 2) * epsilon) * (f0 * f2 - (q + 1) * f1**2 / q))
    assert lin.coeff(epsilon, 0) == 0
    assert sp.factor(lin.coeff(epsilon, 1)) == j * (j + 1)

    f0 = 1 + t + c2 * t**2
    f1 = sp.expand(-q * f0 + J * t * sp.diff(f0, t))
    f2 = sp.expand((-q - 1) * f1 + J * t * sp.diff(f1, t))
    Ht = sp.expand(f0 * f2 - (q + 1) * f1**2 / q)
    St = sp.expand((1 + (n - 2) * t) * Ht)
    kappa = J * (2 * (2 * J + 1) * c2 + (n - 2) * J - 2)
    assert sp.expand(St.coeff(t, 2) - kappa) == 0

    # Direct first-occurrence orders; no weight-space assertion is used.
    T = (n + 2) * q - 2
    assert sp.expand(T - (q - 1) - (J - 1)) == 0
    assert sp.expand(2 * J - q - n * q - J) == 0
    assert sp.expand(d * J - n * q - 2 - (q - 1) - ((d - 1) * J - 1)) == 0
    assert sp.expand((p - 2) - (2 * q - 1) - (p - 2 * q - 1)) == 0
    assert sp.expand(2 * J - 4 * q - 1 - (2 * p - 1)) == 0

    x, y = sp.symbols("x y", nonnegative=True)
    positive = sp.Poly(sp.expand(((n - 2) * J - 2).subs({n: x + 3, q: y + 2})), x, y)
    assert all(coefficient > 0 for coefficient in positive.coeffs())
    i = sp.Symbol("i", integer=True)
    factor_gap = sp.expand(2 * (4 * q + 1 + i) - 5 * (q + i))
    assert factor_gap.subs(i, q - 1) == 5
    assert sp.Rational(5, 2) ** 4 - 27 == sp.Rational(193, 16)
    assert sp.expand(sp.Rational(5, 2) * (6 * q + 3) - (6 * (q + 1) + 3)) == 9 * q - sp.Rational(3, 2)
    assert sp.cancel((q + p) / q) == n
    assert sp.expand(2 * J - q - p + 1 - ((n + 2) * q + 1)) == 0
    assert kappa.subs({n: 2, q: 2, c2: sp.Rational(1, 13)}) == 0

    # Optional degeneracy calculation, independently derived from jets.
    mixed_H = (1 - q) * (-q) + (J - q) * (J - q - 1) - 2 * (q + 1) * (1 - q) * (J - q) / q
    next_S = (J + 1) * (J + 2) * d1 + (mixed_H + (n - 2) * (J * (J + 1) + 2)) * b1
    A0_without_sign = sp.factor(next_S - (J * (J + 1) - 2) * b1)
    A0_expected = (J + 1) * ((J + 2) * d1 + ((n - 2) * J - 2) * b1)
    assert sp.expand(A0_without_sign - A0_expected) == 0
    assert sp.expand(A0_expected.subs(d1, (n + 2) * b1) - 2 * (n + 1) * (n * q + 1) * (J + 1) * b1) == 0

    return {
        "principal_transform_derivative_identity": True,
        "arbitrary_principal_bracket_cancellation": True,
        "linear_seed_symbol": "j*(j+1)",
        "quadratic_seed_symbol": str(sp.factor(kappa)),
        "positive_certificate_n_equals_x_plus_3_q_equals_y_plus_2": str(positive.as_expr()),
        "quadratic_factor_cross_product_gap": str(factor_gap),
        "quadratic_induction_base_gap": "193/16",
        "quadratic_induction_step_gap": "9*q-3/2",
        "A0_before_recurrence_without_parity_sign": str(A0_expected),
        "A0_after_recurrence": "(-1)^(q-1)*2*(n+1)*(n*q+1)*(J+1)*b1",
        "excluded_n2_q2_kappa": "0",
    }


def arbitrary_universal_audit(n: int, q: int):
    assert q >= 2 and (n - 1) * q % 2 == 0
    J, maximum = (n + 1) * q, (n + 1) * q + 2
    alpha1, alpha2, d1, d2 = sp.symbols("alpha1 alpha2 d1 d2")
    coefficients = [sp.Integer(1), *sp.symbols(f"b1:{J}"), H, alpha1 + d1 * H, alpha2 + d2 * H]
    E = laurent(coefficients, -q)
    F = laurent(seed_coefficients(coefficients, n, q, maximum), -n * q - 2)
    a, b, G = bracket(E, F, 1)
    assert a.degree() == q - 1 and a.LC() == sp.Rational((-1) ** (q - 1), sp.factorial(q - 1))
    assert b.degree() <= n * q
    assert not a.as_expr().has(H) and not b.as_expr().has(H)
    g0, g1 = G.get(0, 0), G.get(1, 0)
    assert sp.diff(g0, H, 2) == 0 and sp.diff(g1, H, 2) == 0
    A0, A1 = sp.diff(g0, H), sp.diff(g1, H)
    expected = (-1) ** (q - 1) * (J + 1) * ((J + 2) * d1 + ((n - 2) * J - 2) * coefficients[1])
    assert sp.expand(A0 - expected) == 0
    assert not sp.expand(A0 * g1 - A1 * g0).has(H)
    # This algebraic boundary can be checked without asserting that every
    # arbitrary choice of these jets satisfies an ODE recurrence.
    zero_A0 = {coefficients[1]: 0, d1: 0}
    assert A0.subs(zero_A0) == 0
    assert sp.expand(A1.subs(zero_A0)) != 0
    assert not sp.expand(g0.subs(zero_A0)).has(H)
    zeta = sp.Symbol("zeta")
    assert sp.rem(zeta**n - zeta, zeta ** (n - 1) - 1, zeta) == 0
    return {"n": n, "q": q, "degree_a": a.degree(), "degree_bound_b": n * q, "A0": str(sp.factor(A0)), "principal_and_constant_checks": True, "phase_check": True, "arbitrary_jet_A0_zero_A1_nonzero_boundary": True}


def arbitrary_affine_audit(n: int, q: int):
    p, J = (n - 1) * q, (n + 1) * q
    assert p % 2 == 0 and p > 2 * q + 1
    coefficients = [sp.Integer(1), *sp.symbols(f"b1:{J}"), H]
    E = laurent(convolution(coefficients, coefficients, J), -2 * q)
    F = laurent(seed_coefficients(coefficients, 2, q, J), -2 * q - 2)
    exponent = p - 2 * q - 1
    a, b, G = bracket(E, F, exponent)
    assert a.degree() == 2 * q - 1 and a.LC() == -1 / sp.factorial(2 * q - 1)
    assert b.degree() <= 2 * q
    assert not a.as_expr().has(H) and not b.as_expr().has(H)
    assert not sp.sympify(G.get(0, 0)).has(H)
    gamma = -J * (J + 1) * falling(p - 2, 2 * q - 1) / sp.factorial(2 * q - 1)
    detector = sp.expand(G.get(exponent, 0))
    assert sp.diff(detector, H) == gamma != 0
    assert sp.diff(detector, H, 2) == 0
    return {"n": n, "q": q, "p": p, "detector_power": exponent, "detector_slope": str(gamma), "fixed_constant_term": True, "principal_cancellation": True}


def optional_slope_identity():
    """Independent formal variation and recurrence check of A1=(nq+2)b1*A0."""
    n, q = sp.symbols("n q", integer=True, positive=True)
    z, b1, b2, d1, d2 = sp.symbols("z b1 b2 d1 d2")
    m, N, J = n - 1, n * q, (n + 1) * q

    # Write u=z^(-q)(F+h*z^J*L+...), retaining two subsequent powers.
    # Differentiate both the fixed part and its variation directly.
    F, L = 1 + b1 * z + b2 * z**2, 1 + d1 * z + d2 * z**2
    U1 = -q * F + z * sp.diff(F, z)
    U2 = (-q - 1) * U1 + z * sp.diff(U1, z)
    V1 = (J - q) * L + z * sp.diff(L, z)
    V2 = (J - q - 1) * V1 + z * sp.diff(V1, z)
    H0 = sp.expand(F * U2 - (q + 1) * U1**2 / q)
    delta_H = sp.expand(L * U2 + F * V2 - 2 * (q + 1) * U1 * V1 / q)

    def truncated_F_power(exponent):
        return 1 + exponent * b1 * z + (exponent * b2 + exponent * (exponent - 1) * b1**2 / 2) * z**2

    delta_S = sp.expand(truncated_F_power(n - 2) * delta_H + (n - 2) * truncated_F_power(n - 3) * L * H0)
    L0, L1, L2 = [sp.factor(delta_S.coeff(z, j)) for j in range(3)]
    s2 = sp.factor(sp.expand(truncated_F_power(n - 2) * H0).coeff(z, 2))
    assert sp.expand(L0 - J * (J + 1)) == 0
    assert sp.expand(s2 - (6 * b2 + (2 * n - 3 - 1 / q) * b1**2)) == 0

    # All statements here are divided by the common sign (-1)^(q-1).
    A0 = L1 - b1 * L0 + 2 * b1
    A1 = q * L2 - (q - 1) * b1 * L1 + (q - 2) * b2 * L0 + 2 * (N + 1) * b1 * d1 - N * s2
    Dm, Dp = (N - 2) * (N - 1), (q + 1) * (q + 2)
    alpha = -m * (J - 1) * b1
    beta = (n * Dm - (q - 2) * (q - 1)) * b2 + (n * m * Dm / 2 - m * (J - 1) * (q - 1)) * b1**2
    d1_value = (n + 2) * b1
    d2_value = sp.cancel((m * Dp * (b2 + b1 * d1 + (n - 2) * b1**2 / 2) - alpha * (N + 1) * d1 - beta) / (m * q * (J + 3)))

    # Independently check the four recurrence equations used for these
    # substitutions after dividing their indicial coefficients by K.
    assert sp.cancel(((q - 1) / (N - 1) - n) * b1 - alpha / (N - 1)) == 0
    assert sp.cancel(((q - 2) * (q - 1) / Dm - n) * b2 - (n * m * b1**2 / 2 + alpha * (q - 1) * b1 / Dm - beta / Dm)) == 0
    assert sp.cancel((n * (N + 1) / (q + 1) - n) * d1_value - (n * m * b1 - alpha * n / (q + 1))) == 0
    assert sp.cancel((n * (N + 1) * (N + 2) / Dp - n) * d2_value - (n * m * (b2 + b1 * d1 + (n - 2) * b1**2 / 2) - alpha * n * (N + 1) * d1 / Dp - beta * n / Dp)) == 0

    residual = sp.factor((A1 - (N + 2) * b1 * A0).subs(d2, d2_value).subs(d1, d1_value))
    assert residual == 0
    A1_reduced = sp.factor(A1.subs(d2, d2_value).subs(d1, d1_value))
    assert sp.expand(A1_reduced - 2 * b1**2 * (n + 1) * (N + 1) * (N + 2) * (J + 1)) == 0
    return {
        "method": "direct formal variation of S, followed by four checked recurrence equations",
        "A1_without_parity_sign": str(A1_reduced),
        "identity": "A1=(n*q+2)*b1*A0",
        "identity_residual": str(residual),
        "optional_uniform_completion": "G'-(n*q+2)*b1*G",
        "main_imports_retain_two_case_proof": True,
    }


def recurrence(n: int, q: int, kind: str):
    p, J = (n - 1) * q, (n + 1) * q
    maximum = 2 * J
    x = sp.Symbol("x")
    if kind == "mixed":
        expression = sp.prod(x - p - (n - 1) * i for i in range(p))
    elif kind == "even":
        expression = x**p + 2 * x ** (p - 2) + 3
    elif kind == "pure":
        expression = x**p
    else:
        raise ValueError(kind)
    P = sp.Poly(sp.expand(expression), x)
    K = sp.rf(q, p)
    b = [sp.Integer(1)] + [sp.Integer(0)] * maximum
    for j in range(1, maximum + 1):
        lower = sum(P.nth(r) * falling(j - p + r - q, r) * b[j - p + r] for r in range(p) if j - p + r >= 0)
        rhs = sp.expand(K * power(b[: j + 1], n, j)[j] - lower)
        indicial = falling(j - q, p) - n * K
        if j == J:
            assert indicial == 0 and rhs == 0
            b[j] = H
        else:
            assert indicial != 0
            b[j] = sp.expand(rhs / indicial)
            assert sp.Poly(b[j], H).degree() <= j // J
    assert b[1] == -P.nth(p - 1) / ((n - 1) * (J - 1))
    assert b[J + 1].coeff(H) == (n + 2) * b[1]
    return P, b


def exact_front_parameter(n: int, q: int):
    """Normalized actual front u=(m/(1-exp(-m*z)))^q near a pole."""
    m, J = n - 1, (n + 1) * q
    base = [sp.Integer(1), sp.Rational(m, 2)]
    base.extend(sp.bernoulli(j) * m**j / sp.factorial(j) if j % 2 == 0 else sp.Integer(0) for j in range(2, J + 1))
    return power(base, q, J)[J]


def full_universal_audit(n: int, q: int, kind: str):
    p, J = (n - 1) * q, (n + 1) * q
    P, coefficients = recurrence(n, q, kind)
    F = laurent(seed_coefficients(coefficients, n, q, 2 * J), -n * q - 2)
    a, b, G = bracket(laurent(coefficients, -q), F, J - 1)
    R = sp.rf((n + 2) * q + 1, p) / sp.rf(q, p)
    c2 = sp.binomial(n, 2) / (R - n)
    assert R > n and coefficients[2 * J].coeff(H, 2) == c2
    kappa = sp.cancel(J * (2 * (2 * J + 1) * c2 + (n - 2) * J - 2))
    gamma = sp.cancel(a.LC() * kappa * falling((n + 2) * q - 2, q - 1))
    for e in range(J - 1):
        assert sp.Poly(G.get(e, 0), H).degree() <= 1
    assert sp.expand(G.get(J - 1, 0)).coeff(H, 2) == gamma
    g0, g1 = sp.sympify(G.get(0, 0)), sp.sympify(G.get(1, 0))
    A0, A1 = sp.diff(g0, H), sp.diff(g1, H)
    expected_A0 = (-1) ** (q - 1) * 2 * (n + 1) * (n * q + 1) * (J + 1) * coefficients[1]
    assert A0 == expected_A0
    assert A1 == (n * q + 2) * coefficients[1] * A0
    valid = (n, q) != (2, 2)
    result = {"n": n, "q": q, "operator_kind": kind, "operator": str(P.as_expr()), "A0": str(A0), "A1": str(A1), "b_is_zero": bool(b.is_zero), "kappa": str(kappa), "gamma": str(gamma), "included_in_theorem": valid}
    if valid:
        assert gamma != 0
        if A0 == 0:
            target = J - 1
            detector = sp.expand(G.get(target, 0))
            expected_leading = gamma
            assert not g0.has(H)
            result["constant_completion"] = "G"
        else:
            target = J - 2
            detector = sp.expand(A0 * (target + 1) * G.get(target + 1, 0) - A1 * G.get(target, 0))
            expected_leading = A0 * (J - 1) * gamma
            assert not sp.expand(A0 * g1 - A1 * g0).has(H)
            result["constant_completion"] = "A0*G'-A1*G"
        assert sp.Poly(detector, H).degree() == 2
        assert detector.coeff(H, 2) == expected_leading != 0
        result["detector_power"] = target
        result["monic_detector"] = str(sp.expand(detector / expected_leading))
        if kind == "mixed":
            actual_h = exact_front_parameter(n, q)
            assert sp.expand(detector.subs(H, actual_h)) == 0
            assert all(sp.expand(value.subs(H, actual_h)) == 0 for e, value in G.items() if e > 0)
            result["actual_front_h"] = str(actual_h)
            result["actual_front_all_checked_positive_G_coefficients_vanish"] = True
    else:
        assert gamma == 0 and kappa == 0
    return result


def main():
    report = {"uniform": uniform_checks(), "optional_slope_identity": optional_slope_identity(), "arbitrary_universal": [], "arbitrary_affine": [], "full_recurrence": []}
    print("PASS transform, symbolic seed/degeneracy identities, first-occurrence orders, and sign certificates.", flush=True)
    for n, q in ((2, 2), (2, 4), (3, 2), (3, 3), (4, 2)):
        report["arbitrary_universal"].append(arbitrary_universal_audit(n, q))
        print(f"PASS arbitrary-jet universal bracket n={n}, q={q}.", flush=True)
    for n, q in ((4, 2), (5, 1), (5, 2)):
        report["arbitrary_affine"].append(arbitrary_affine_audit(n, q))
        print(f"PASS arbitrary-jet affine bracket n={n}, q={q}.", flush=True)
    jobs = ((2, 2, "mixed"), (2, 4, "mixed"), (3, 2, "mixed"), (3, 3, "mixed"), (4, 2, "mixed"), (2, 4, "even"), (3, 2, "even"), (3, 2, "pure"))
    for n, q, kind in jobs:
        report["full_recurrence"].append(full_universal_audit(n, q, kind))
        print(f"PASS full recurrence and bracket n={n}, q={q}, operator={kind}.", flush=True)
    # q=1's universal principal transform is h-dependent; the affine proof
    # was already checked at q=1 above and does not share this failure.
    n, q, J = 5, 1, 6
    _, coefficients = recurrence(n, q, "mixed")
    S = laurent(seed_coefficients(coefficients, n, q, 2 * J), -n * q - 2)
    assert transform(S).as_expr().coeff(H) == J * (J + 1)
    report["q1_universal_obstruction"] = {"n": n, "q": q, "h_dependent_residue": str(J * (J + 1)) + "*h", "affine_bracket_covers_this_case": True}
    assert any(item["A0"] == "0" for item in report["full_recurrence"])
    assert any(item["A0"] != "0" for item in report["full_recurrence"])
    assert any(item["A0"] == "0" and item["A1"] == "0" and item["b_is_zero"] for item in report["full_recurrence"])
    print("PASS both constant-slope cases, zero b, and the q=1 / second-order exclusions.", flush=True)
    target = Path(__file__).with_name("check_bracket_certificate.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {target.name}.", flush=True)


if __name__ == "__main__":
    main()
