"""All 15 pairs of the untwisted sector at (n,q) = (3,2) with arbitrary backgrounds:  w = b + (D - rho0) G_2,  G_2 = 1 - Y,  P(D) w = 480 w^3."""
import sympy as sp
Y, rho, b, r0 = sp.symbols('Y rho b rho0')
n, q, m, p = 3, 2, 2, 4
a = sp.symbols('a0:4')
Dm = lambda f: sp.expand(sp.diff(f, Y)*(-m*Y*(1 - Y)))
G = 1 - Y
w = sp.expand(b + Dm(G) - r0*G)
dw = [w]
for _ in range(p): dw.append(Dm(dw[-1]))
kappa = 480
res = sp.expand(dw[p] + sum(a[j]*dw[j] for j in range(p)) - kappa*w**3)
co = sp.Poly(res, Y).all_coeffs()[::-1]
eqs = [sp.expand(c) for c in co[:6]]
assert len(co) == 6
Gb = sp.groebner(eqs, *a, r0, b, order='lex')
elim_b = [g for g in Gb.exprs if g.free_symbols <= {b}][0]
print("eliminant in b:", sp.factor(elim_b))
sols = sp.solve(eqs, list(a) + [r0, b], dict=True)
print(len(sols), "solutions")
rows = []
for s in sols:
    P = sp.factor(rho**4 + sum(s[a[j]]*rho**j for j in range(4)))
    wm = sp.nsimplify(s[b]); wp = sp.nsimplify(sp.simplify(s[b] + (-s[r0])))     # A(0) = -rho0
    typ = ("pulse at 0" if wm == 0 and wp == 0 else "front 0 -> e" if wm == 0 else "front e -> 0" if wp == 0 else
           "pulse at e" if sp.simplify(wm - wp) == 0 else "front e -> -e" if sp.simplify(wm + wp) == 0 else "?")
    rows.append((typ, s[r0], s[b], P))
for typ in ["pulse at 0", "front 0 -> e", "front e -> 0", "front e -> -e", "pulse at e"]:
    for t, r_, b_, P in rows:
        if t == typ: print(f"{typ:14s} rho0 = {str(r_):22s} b = {str(b_):28s} P = {P}")
