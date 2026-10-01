#!/usr/bin/env python3
"""Exact controls for example formulas (Swift--Hohenberg: swift_hohenberg_control).

Every existence check substitutes a rational profile into its full ODE.
The two nodal controls retain free parameters; the Swift--Hohenberg
controls recover the necessary-and-sufficient parameter factors from the
actual residual. No numerical point sampling or frozen-control rerun is
used. This script writes no files.
"""

from __future__ import annotations

import json

import sympy as s


t, Y, b, e, a, xi, beta = s.symbols("t Y b e a xi beta")
ZERO, ONE = s.S.Zero, s.S.One


def zero(value):
    reduced = s.cancel(value)
    assert reduced == 0, s.factor(reduced)


def derivatives(profile, coordinate, velocity, maximum=4):
    result = [profile]
    for _ in range(maximum):
        result.append(s.cancel(velocity * s.diff(result[-1], coordinate)))
    return result


def ode_residual(profile, operator, nonlinear, coordinate, velocity):
    jet = derivatives(profile, coordinate, velocity)
    return s.factor(sum(operator.get(j, ZERO) * jet[j] for j in range(5))
                    - nonlinear * profile**3)


def monic(poly, variable):
    return s.Poly(poly, variable).monic().as_expr()


def plus_nodal_control():
    # t=exp(k*z), k^2=12e. This is the entire plus nodal normalization,
    # including both equal-nonzero-end components and the zero-end one.
    profile = b + e + 12 * e * t / (t - 1)**2
    logarithmic_jet = derivatives(profile, t, t)
    gamma = s.Rational(3, 5) * e**2
    residual = s.factor(
        (12 * e)**2 * logarithmic_jet[4]
        + 60 * b * 12 * e * logarithmic_jet[2]
        + 360 * (b**2 + gamma) * profile - 120 * profile**3
    )
    parameter_equation = (b + e) * (5 * b**2 - 5 * b * e + 2 * e**2)
    numerator, denominator = s.fraction(s.cancel(residual))
    # Vanishing in the reducible parameter ring, not at sampled points.
    assert s.rem(numerator, parameter_equation, b) == 0
    assert not denominator.has(b, e)
    # Each component is checked separately as well.
    zero(residual.subs(b, -e))
    assert s.rem(numerator, 5 * b**2 - 5 * b * e + 2 * e**2, b) == 0

    X = profile - b
    dX_squared = 12 * e * (t * s.diff(X, t))**2
    zero(dX_squared - (4 * X**3 - 12 * e**2 * X + 8 * e**3))
    # The residual parameter equation is exactly the operator's nodal
    # invariant condition. This also identifies the double root e.
    zero(20 * b * (b**2 - gamma) + 8 * e**3 - 4 * parameter_equation)

    # Laurent coefficients follow from the actual cylinder generating
    # function k/(exp(kz)-1), via its Bernoulli expansion.
    h_from_profile = -7 * s.bernoulli(8) * (12 * e)**4 / s.factorial(8)
    zero(h_from_profile - gamma**2 / 3)
    return {
        "coordinate": "t=exp(k*z), k^2=12e, e!=0",
        "profile": str(profile),
        "parameter_equation": str(parameter_equation),
        "full_ODE_residual_before_parameter_reduction": str(residual),
        "full_ODE_residual_on_each_component": 0,
        "nodal_cubic_identity": 0,
        "h": str(s.factor(h_from_profile)),
    }


def minus_nodal_control():
    # t=exp(a*z), a^2=6b; the manuscript's primitive pole rate is k=2a.
    profile = 2 * a**2 * t * (t**2 + 1) / (t**2 - 1)**2
    operator = {4: ONE, 2: 10 * a**2, 0: -11 * a**4}
    residual = ode_residual(profile, operator, 120, t, a * t)
    zero(residual)
    zero(profile.subs(t, -t) + profile)  # exact character under the pole period
    parameter_b = a**2 / 6
    gamma = -s.Rational(7, 120) * a**4
    zero(60 * parameter_b - operator[2])
    zero(360 * (gamma + parameter_b**2) - operator[0])
    zero(21 * parameter_b**2 + 10 * gamma)
    h_from_profile = -7 * s.bernoulli(8, s.Rational(1, 2)) * (2 * a)**8 / s.factorial(8)
    h_minus = (-parameter_b**4 / 6 + s.Rational(5, 9) * parameter_b**2 * gamma
               - s.Rational(7, 54) * gamma**2)
    zero(h_from_profile - h_minus)
    return {
        "coordinate": "t=exp(a*z), a^2=6b, a!=0; primitive pole rate k=2a",
        "profile": str(profile),
        "operator": "D^4+10*a^2*D^2-11*a^4",
        "gamma": str(gamma),
        "full_ODE_residual": str(residual),
        "character_under_t_to_minus_t": -1,
        "h": str(s.factor(h_from_profile)),
    }


def swift_hohenberg_control():
    # The first profile is rational in t=exp(z), and its nonlinear
    # coefficient is -480 in the displayed manuscript normalization.
    generator = t / (1 + t**2)
    w1 = s.cancel(t * s.diff(generator, t) - xi * generator)
    operator1 = {
        4: ONE,
        3: -14 * xi,
        2: 46 * xi**2 + 10,
        1: -2 * xi * (7 * xi**2 + 10),
        0: -14 * xi**4 - 80 * xi**2 - 11,
    }
    residual1 = ode_residual(w1, operator1, -480, t, t)
    numerator1, _ = s.fraction(s.cancel(residual1))
    coefficients1 = s.Poly(numerator1, t).all_coeffs()
    residual_gcd1 = s.gcd_list(coefficients1)
    condition1 = xi * (xi**2 - 1) * (7 * xi**2 + 17)
    zero(monic(residual_gcd1, xi) - monic(condition1, xi))
    assert all(s.rem(coefficient, condition1, xi) == 0 for coefficient in coefficients1)

    # Y=(1+exp(2z))^-1, DY=-2Y(1-Y). This directly represents the
    # new midpoint-background profile, not its old coefficient formula.
    w2 = beta + 2 * Y * (1 - Y) - xi * (1 - 2 * Y) / 2
    operator2 = {
        4: ONE,
        3: -14 * xi,
        2: 46 * xi**2 - 120 * beta - 20,
        1: -2 * xi * (7 * xi**2 - 300 * beta - 20),
        0: 1440 * beta**2 - 120 * beta * xi**2 + 480 * beta
           - 14 * xi**4 + 160 * xi**2 + 64,
    }
    residual2 = s.expand(ode_residual(w2, operator2, 480, Y, -2 * Y * (1 - Y)))
    assert s.degree(residual2, Y) <= 1
    endpoint_minus = s.expand(w2.subs(Y, 1))
    endpoint_plus = s.expand(w2.subs(Y, 0))
    end_equation = s.factor(endpoint_minus * endpoint_plus
                            * (endpoint_plus**2 - endpoint_minus**2))
    zero(end_equation + 2 * beta * xi * (beta**2 - xi**2 / 4))

    rows = [
        ("beta=-xi/2", {beta: -xi / 2}, xi,
         xi * (xi - 4) * (xi**2 - 4) * (7 * xi - 2)),
        ("beta=xi/2", {beta: xi / 2}, xi,
         xi * (xi + 4) * (xi**2 - 4) * (7 * xi + 2)),
        ("xi=0", {xi: ZERO}, beta, beta * (30 * beta**2 + 15 * beta + 2)),
        ("beta=0", {beta: ZERO}, xi, xi * (xi**2 - 4) * (7 * xi**2 + 8)),
    ]
    checked_rows = []
    sigma2 = set()
    for name, substitution, parameter, condition in rows:
        row_coefficients = [s.factor(residual2.coeff(Y, j).subs(substitution)) for j in (0, 1)]
        actual_gcd = monic(s.gcd(*row_coefficients), parameter)
        zero(actual_gcd - monic(condition, parameter))
        assert all(s.rem(coefficient, condition, parameter) == 0 for coefficient in row_coefficients)
        checked_rows.append({"line": name, "necessary_and_sufficient_factor": str(s.factor(actual_gcd))})
        # Verify the displayed dispersion list from the full operator map.
        for root in s.roots(condition, parameter):
            a3 = s.simplify(operator2[3].subs(substitution).subs(parameter, root))
            a2 = s.simplify(operator2[2].subs(substitution).subs(parameter, root))
            assert a2 != 0
            value = s.simplify(2 * a3**2 / a2)
            assert value.is_Rational
            sigma2.add(value)
    expected_sigma2 = {ZERO, s.Rational(392, 71), s.Rational(1568, 239),
                       s.Rational(392, 11), s.Rational(392, 41), s.Rational(784, 127)}
    assert sigma2 == expected_sigma2
    sigma1 = {
        s.simplify((2 * operator1[3]**2 / operator1[2]).subs(xi, root))
        for root in s.roots(condition1, xi)
    }
    assert sigma1 == {ZERO, s.Integer(7), s.Rational(833, 89)}

    # The local normalization in the text follows from the actual pole
    # coefficients at t=i (z=pi*i/2), with t-i ~ i(z-z_*).
    w2_in_t = s.cancel(w2.subs(Y, 1 / (1 + t**2)))
    lead1 = s.simplify(s.limit((t - s.I)**2 * w1, t, s.I) / s.I**2)
    lead2 = s.simplify(s.limit((t - s.I)**2 * w2_in_t, t, s.I) / s.I**2)
    zero(lead1 - s.I / 2)
    zero(lead2 + s.Rational(1, 2))
    zero(-480 * lead1**2 - 120)
    zero(480 * lead2**2 - 120)
    zero(w1.subs(t, -t) + w1)
    zero(w2_in_t.subs(t, -t) - w2_in_t)
    return {
        "twisted_full_ODE_residual": str(residual1),
        "twisted_necessary_and_sufficient_factor": str(s.factor(residual_gcd1)),
        "untwisted_operator_vector": {"a" + str(j): str(operator2[j]) for j in range(4)},
        "untwisted_residual_degree_in_Y": int(s.degree(residual2, Y)),
        "endpoint_equation": str(end_equation),
        "untwisted_rows": checked_rows,
        "leading_pole_coefficients": [str(lead1), str(lead2)],
        "normalized_nonlinear_coefficient": 120,
        "primitive_pole_rate_at_displayed_scale": 2,
        "sigma_squared_sector_1": sorted(map(str, sigma1)),
        "sigma_squared_sector_2": sorted(map(str, sigma2)),
    }


def main():
    result = {
        "plus_nodal": plus_nodal_control(),
        "minus_nodal": minus_nodal_control(),
        "Swift_Hohenberg": swift_hohenberg_control(),
        "status": "all exact rational-coordinate ODE controls passed",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
