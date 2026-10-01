"""Explicit pairs for pole order q = 2 in every simply periodic sector (zero background):  v = sigma A(D) G_r,  G_r = e^{rx}/(1+e^{mx}),
A = rho - rho0.  For each pair: rho0, P (factored), label rho0 mod l, reciprocity check, pulse/front."""
import sympy as sp, sys
from sectors import count, rho
def table(n):
    q = 2; m = n - 1; l = n*q - 1
    print(f"=== n = {n}, q = 2, p = {m*q}, l = {l} ===")
    for r in range(1, m + 1):
        R = count(n, q, r); u = R['us'][0]
        elim = sp.Poly(R['eqs'][0], u)
        for e in R['eqs'][1:]: elim = sp.gcd(elim, sp.Poly(e, u))
        print(f"sector r = {r}{' (untwisted)' if r == m else ''}: eliminant {sp.factor(elim.as_expr())}")
        for fac, mult in sp.factor_list(elim.as_expr())[1]:
            if sp.Poly(fac, u).degree() > 2:
                print(f"    irreducible factor of degree {sp.Poly(fac, u).degree()}: {fac}  (roots not listed)")
                continue
            for root in sp.roots(sp.Poly(fac, u)):
                A = sp.expand(R['A'].subs(u, root)); P = sp.factor(sp.expand(R['P'].subs(u, root)))
                rho0 = sp.solve(A, rho)[0]
                lab = None
                if rho0.is_rational:
                    lab = (sp.Rational(rho0).p*pow(int(sp.Rational(rho0).q), -1, l)) % l
                print(f"    rho0 = {rho0},  label = {lab},  P = {P}")
for n in ([int(a) for a in sys.argv[1:]] or [3, 4]):
    table(n)
