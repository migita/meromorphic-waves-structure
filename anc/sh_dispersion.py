"""Swift-Hohenberg equation with dispersion,  u_t + 2 u_xx - sigma u_xxx + u_xxxx = alpha u - u^3  (pure cubic case of Kudryashov-Sinelshchikov 2012).
Travelling waves:  P(D) w + (cubic) = 0  with  P = rho^4 - sigma rho^3 + 2 rho^2 - C0 rho - alpha.
A normalized pair with operator P* = rho^4 + a3 rho^3 + a2 rho^2 + a1 rho + a0 and rate kappa gives  P(rho) = kappa^4 P*(rho/kappa):
      kappa^2 a2 = 2,   sigma = -kappa a3,   so   sigma^2 = 2 a3^2 / a2   (and alpha = -4 a0/a2^2,  C0 = -a1 kappa^3).
List of sigma^2 for all twenty simply periodic pairs at (n,q) = (3,2).
Kudryashov-Sinelshchikov 2012 found sigma = +-sqrt(7) and sigma = +-7 sqrt(1513)/89 for the solutions with two types of poles."""
import sympy as sp
rho = sp.Symbol('rho')
I14, I15, I119 = sp.sqrt(-14), sp.sqrt(-15), sp.sqrt(-119)
pairs = [
 ("sector 1 (twisted)", "{0}", (rho**2 - 1)*(rho**2 + 11)),
 ("sector 1 (twisted)", "{1}", (rho + 1)*(rho - 3)*(rho - 5)*(rho - 7)),
 ("sector 1 (twisted)", "{4}", (rho - 1)*(rho + 3)*(rho + 5)*(rho + 7)),
 ("sector 1 (twisted)", "{2},{3}", (rho**2 - 1)*(rho**2 - 2*I119*rho - sp.Rational(705, 7))),
 ("sector 2, 0 -> 0", "{0}", (rho**2 - 4)*(rho**2 - 16)),
 ("sector 2, 0 -> e", "{1}", (rho - 2)*(rho - 4)*(rho**2 + 2*rho + sp.Rational(240, 49))),
 ("sector 2, 0 -> e", "{2}", (rho - 4)*(rho - 6)*(rho - 8)*(rho - 10)),
 ("sector 2, 0 -> e", "{3}", (rho - 2)*(rho - 4)*(rho + 10)*(rho + 24)),
 ("sector 2, 0 -> e", "{4}", (rho - 2)*(rho - 8)*(rho - 16)*(rho - 30)),
 ("sector 2, e -> -e", "{2},{3}", rho**4 - 28*rho**3 + 164*rho**2 - 32*rho + 480),
 ("sector 2, e -> -e", "{1},{4}", rho**4 - 4*I14*rho**3 - sp.Rational(508, 7)*rho**2 + 16*I14*rho - sp.Rational(960, 7)),
 ("sector 2, e -> e", "{0}", rho**4 + (10 - 2*I15)*rho**2 + 28 - 4*I15),
]
print("mirror images (rho -> -rho) give the same sigma^2 and are not listed; 'e -> 0' fronts are mirror images of '0 -> e'.")
for name, lab, P in pairs:
    a = sp.Poly(sp.expand(P), rho).all_coeffs()
    a3, a2, a1, a0 = a[1], a[2], a[3], a[4]
    s2 = sp.nsimplify(sp.simplify(2*a3**2/a2)) if a2 != 0 else None
    al = sp.nsimplify(sp.simplify(-4*a0/a2**2))
    print(f"{name:22s} label {lab:8s} sigma^2 = {str(s2):14s} ({sp.N(s2, 6) if s2 is not None else ''})   alpha = {al}   kappa^2 = {sp.nsimplify(sp.simplify(2/a2))}")
print("check: 7*sqrt(1513)/89 squared =", sp.nsimplify((7*sp.sqrt(1513)/89)**2))
