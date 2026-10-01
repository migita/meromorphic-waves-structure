"""Twisted sectors of one-pole waves for the pure power equation  P(D)v + v^n/n = 0,  pole order q = p/(n-1),  m = n-1.

Sector r in {1, ..., m}  (r = m is the untwisted sector, zero background):  v = e^{r x} f(Y),  Y = 1/(1 + e^{m x}),
f a polynomial of degree q without constant term.  Then
   D (e^{rx} Y^k) = e^{rx} [ (r - m k) Y^k + m k Y^{k+1} ],            (e^{rx} f)^n = e^{rx} ((1-Y)/Y)^r f^n .
In the expansion at x -> -infinity, v = sum_{k = r (mod m)} c_k e^{kx}; v(x + 2 pi i/m) = exp(2 pi i r/m) v(x); v has m poles of order q
per period 2 pi i, one on each Laurent branch.  For r = m the variable e^{mx} is the usual one and v = f(1 - Q(mx)) is a one-pole wave of rate m.

This script: builds the square matching system (unknowns: p coefficients of P, q-1 coefficients of f), counts its solutions exactly
(msolve, or SymPy when small), and at a prime index l = nq-1 checks the reciprocity  A(rho) P(rho) = rho^l - rho (mod l)  and the labels."""
import sympy as sp, subprocess, os, re, itertools, sys
Y, rho = sp.symbols('Y rho')
MSOLVE = os.environ.get('MSOLVE', '/tmp/msolve/msolve')

def D_sector(f, r, m):
    """derivation on e^{rx} f(Y):  returns g with D(e^{rx} f) = e^{rx} g"""
    # Y' = -m Y (1-Y)
    return sp.expand(r*f + sp.diff(f, Y)*(-m*Y*(1 - Y)))

def system(n, q, r):
    m = n - 1; p = m*q
    fs = sp.symbols('f1:%d' % q) if q > 1 else ()
    lead = sp.symbols('c')
    f = lead*Y**q + sum(fk*Y**(k + 1) for k, fk in enumerate(fs))
    ps = sp.symbols('p0:%d' % p)
    d = [f]
    for _ in range(p): d.append(D_sector(d[-1], r, m))
    res = d[p] + sum(ps[j]*d[j] for j in range(p))
    res = sp.expand(res + sp.cancel(((1 - Y)/Y)**r * f**n)/n)
    co = sp.Poly(res, Y).all_coeffs()[::-1]                      # co[k] = coefficient of Y^k
    assert all(sp.expand(c_) == 0 for c_ in co[:1]), "constant term must vanish"
    top = sp.factor(co[n*q])                                     # fixes c^(n-1)
    cpow = sp.solve(sp.cancel(top/lead), lead**(n - 1))
    return f, fs, ps, lead, co, top, cpow

def A_from_f(f, fs, lead, r, m, q):
    """polynomial A (not normalized) with e^{rx} f(Y) = A(D) G_r,  G_r = e^{rx} Y"""
    A = 0
    for k in range(1, q + 1):
        fk = sp.Poly(f, Y).coeff_monomial(Y**k)
        A += fk * sp.prod([(rho - r + m*i) for i in range(1, k)]) / (sp.Integer(m)**(k - 1) * sp.factorial(k - 1))
    return sp.expand(A)

def count(n, q, r, verbose=True):
    m = n - 1; p = m*q; l = n*q - 1
    f, fs, ps, lead, co, top, cpow = system(n, q, r)
    assert len(cpow) == 1
    cval = cpow[0]
    # eliminate the leading coefficient: put f_k = lead * u_k (weights), so that every equation is lead * (polynomial in lead^(n-1), u, p)
    us = sp.symbols('u1:%d' % q) if q > 1 else ()
    sub = {fk: lead*uk for fk, uk in zip(fs, us)}
    eqs = []
    for k in range(1, n*q):
        e = sp.expand(co[k].subs(sub))
        e = sp.expand(sp.cancel(e/lead))
        P_ = sp.Poly(e, lead)
        assert all(kk % (n - 1) == 0 for (kk,) in P_.monoms()), "unexpected power of the leading coefficient"
        e = sum(cc * cval**(kk//(n - 1)) for (kk,), cc in P_.terms())
        eqs.append(sp.expand(e))
    gens = list(ps) + list(us)
    # the operator coefficients enter linearly and triangularly: solve them from the top equations, leaving q-1 equations in u
    sol = {}
    rem = list(eqs)
    for j in range(p - 1, -1, -1):
        for e in rem:
            e1 = sp.expand(e.subs(sol))
            if e1.has(ps[j]) and sp.Poly(e1, ps[j]).degree() == 1 and not any(e1.has(ps[i]) for i in range(j)):
                lc = sp.Poly(e1, ps[j]).LC()
                if lc.free_symbols == set():
                    sol[ps[j]] = sp.expand(sp.solve(e1, ps[j])[0]); rem.remove(e); break
    rest = [sp.expand(e.subs(sol)) for e in rem]
    rest = [e for e in rest if e != 0]
    assert all(not any(e.has(pj) for pj in ps) for e in rest) and len(sol) == p
    P = rho**p + sum(sol[ps[j]]*rho**j for j in range(p))
    A = sp.expand(A_from_f(f.subs(sub), fs, lead, r, m, q)/lead)
    A = sp.expand(A / sp.Poly(A, rho).LC())
    return dict(n=n, q=q, r=r, l=l, P=P, A=A, us=us, eqs=rest, cval=cval)

def n_solutions(eqs, gens, tag):
    if not gens: return 1, 1
    if len(gens) == 1:
        g = sp.Poly(eqs[0], gens[0])
        for e in eqs[1:]: g = sp.gcd(g, sp.Poly(e, gens[0]))
        return g.degree(), sp.Poly(sp.cancel(g.as_expr()/sp.gcd(g, g.diff()).as_expr()), gens[0]).degree()
    fn = f'_sec_{tag}.ms'
    cl = [sp.Poly(e, *gens).clear_denoms()[1].as_expr() for e in eqs]
    open(fn, 'w').write(','.join(map(str, gens)) + '\n0\n' + ',\n'.join(str(e).replace('**', '^') for e in cl) + '\n')
    rr = subprocess.run([MSOLVE, '-v', '1', '-t', '4', '-f', fn, '-o', fn + '.out'], capture_output=True, text=True)
    log = rr.stdout + rr.stderr
    for f_ in (fn, fn + '.out'):
        if os.path.exists(f_): os.remove(f_)
    g = lambda pat: (int(re.findall(pat, log)[-1]) if re.findall(pat, log) else None)
    return g(r'degree of ideal\s+(\d+)'), g(r'deg\. sqfr\. elim\. pol\.\s+(\d+)')

if __name__ == '__main__':
    cases = [(3, 1, 1), (3, 1, 2), (3, 2, 1), (3, 2, 2), (3, 3, 1), (3, 3, 2), (3, 4, 1), (3, 4, 2), (4, 1, 1), (4, 1, 2), (4, 1, 3), (4, 2, 1), (4, 2, 2), (4, 2, 3), (4, 3, 1), (5, 2, 1), (5, 2, 2)]
    if len(sys.argv) > 1:
        cases = [tuple(int(t) for t in a.split(',')) for a in sys.argv[1:]]
    for n, q, r in cases:
        R = count(n, q, r)
        tot, dist = n_solutions(R['eqs'], list(R['us']), f"{n}{q}{r}")
        print(f"(n,q,r)=({n},{q},{r}){' untwisted' if r == n-1 else ' twisted  '}: leading coeff^(n-1) = {R['cval']};  solutions: {tot} with multiplicity, {dist} distinct;"
              f"  C(nq-1,q-1) = {sp.binomial(n*q-1, q-1)};  l = {R['l']}{' prime' if sp.isprime(R['l']) else ''}", flush=True)
