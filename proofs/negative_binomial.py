"""Negative Binomial: the observed information, in eta = log S1 and kappa = log k.

Each pattern's voxel totals are negative binomial with mean mu_v = S1 u_v and dispersion
k = theta S2 / S1^2. The moderators reach the likelihood only through eta = log S1 and
kappa = log k, so the proof closes in three pieces, each small enough for sympy:

  1. NiMARE's summand *is* lambda(eta, kappa, S) under eta = log S1, kappa = log(theta S2 / S1^2);
  2. lambda's partials in (eta, kappa) are what the implementation uses; S_v enters as eta does;
  3. eta and kappa's first and second derivatives in gamma are what the implementation uses.

The composite is then the multivariate chain rule applied to 2 and 3. Two identities the
implementation relies on for precision near theta = 0 are checked as well: the digamma terms of
the kappa partials equal finite sums at integer counts, and the power series that replaces
log(1 + x)/x - 1/(1 + x) for small x.
"""
import sympy as sp

from proofs.latex import Proof

N, Q, MAX_COUNT, SERIES_TERMS = 2, 2, 5, 10

PREAMBLE = r"""NiMARE's negative binomial uses shape $r=s_1^2/(\theta s_2)$ and success
probability $\pi_v=(1+s_1/(\theta u_v s_2))^{-1}$. Writing $\eta=\log s_1$ and
$\kappa=\log k$ with $k=\theta s_2/s_1^2$, the voxel mean is $\mu_v=e^\eta u_v$, the size is
$1/k$, and each voxel's summand depends on $(\eta,\kappa)$ and $S_v$ alone. Below,
$x_v=k\mu_v$."""


def run():
    proof = Proof("negative_binomial", "Negative Binomial", preamble=PREAMBLE)

    eta, kappa = sp.symbols("eta kappa", real=True)
    theta, S1, S2 = sp.symbols("theta S1 S2", positive=True)
    S, y = sp.symbols("S y", real=True)
    u, k = sp.exp(S), sp.exp(kappa)
    mu, size = sp.exp(eta) * u, 1 / k
    x = k * mu

    # --- 1. the summand in (eta, kappa) is NiMARE's ---------------------------------------------
    r_ = S1**2 / (theta * S2)
    p_ = 1 / (1 + S1 / (theta * u * S2))
    nimare = sp.loggamma(y + r_) - sp.loggamma(r_) + r_ * sp.log(1 - p_) + y * sp.log(p_)
    lam = (sp.loggamma(y + size) - sp.loggamma(size) + y * kappa + y * sp.log(mu)
           - (y + size) * sp.log(1 + x))
    proof.define(r"\lambda(\eta, \kappa, S_v)", lam,
                 r"One voxel's log-likelihood, up to $-\log\Gamma(Y_v+1)$.")
    substituted = lam.subs({eta: sp.log(S1), kappa: sp.log(theta * S2 / S1**2)})
    proof.claim("lambda(eta,kappa,S) equals NiMARE's summand",
                sp.expand_log(sp.simplify(sp.expand_log(sp.expand(substituted - nimare),
                                                        force=True)), force=True),
                r"\lambda\big|_{\eta,\kappa} - \big(\log\mathcal{L}_v + \log\Gamma(Y_v+1)\big)")

    # --- 2. partials of lambda ------------------------------------------------------------------
    gap = sp.log(1 + x) / x - 1 / (1 + x)
    log_step = sp.digamma(y + size) - sp.digamma(size)
    square_step = sp.polygamma(1, y + size) - sp.polygamma(1, size)
    count_k = y - size * log_step
    count_kk = size * log_step + size**2 * square_step
    proof.define("G_1", count_k,
                 r"The count term of $\lambda_\kappa$. $\operatorname{polygamma}(0, \cdot)$"
                 r" is the digamma function.")
    proof.define("G_2", count_kk, r"The count term of $\lambda_{\kappa\kappa}$.")
    x_v = sp.Symbol("x_v", positive=True)
    proof.define("g(x_v)", sp.log(1 + x_v) / x_v - 1 / (1 + x_v))
    proof.claim("d lambda/dS = d lambda/deta", sp.diff(lam, S) - sp.diff(lam, eta),
                r"\partial_{S_v}\lambda - \partial_\eta\lambda")
    for label, expected, wrt, shown in [
        ("lambda_eta", (y - mu) / (1 + x), (eta,), r"\lambda_\eta - \frac{Y_v-\mu_v}{1+x_v}"),
        ("lambda_eta_eta", -(mu + y * x) / (1 + x)**2, (eta, eta),
         r"\lambda_{\eta\eta} + \frac{\mu_v+Y_vx_v}{(1+x_v)^2}"),
        ("lambda_eta_kappa", -(y - mu) * x / (1 + x)**2, (eta, kappa),
         r"\lambda_{\eta\kappa} + \frac{(Y_v-\mu_v)x_v}{(1+x_v)^2}"),
        ("lambda_kappa", count_k + mu * gap - y * x / (1 + x), (kappa,),
         r"\lambda_\kappa\ \textrm{as implemented}"),
        ("lambda_kappa_kappa", count_kk + mu * (x / (1 + x)**2 - gap) - y * x / (1 + x)**2,
         (kappa, kappa), r"\lambda_{\kappa\kappa}\ \textrm{as implemented}"),
    ]:
        proof.claim(label, sp.simplify(sp.diff(lam, *wrt) - expected), shown)

    # At integer counts the digamma terms are finite sums with positive terms.
    for count in range(MAX_COUNT + 1):
        proof.claim(f"count sum in lambda_kappa, y={count}",
                    sp.simplify(sp.expand_func(count_k.subs(y, count))
                                - sum(j * k / (1 + j * k) for j in range(count))),
                    rf"G_1(Y_v={count}) - \sum_{{j<{count}}}\frac{{jk}}{{1+jk}}")
        proof.claim(f"count sum in lambda_kappa_kappa, y={count}",
                    sp.simplify(sp.expand_func(count_kk.subs(y, count))
                                - sum(j * k / (1 + j * k)**2 for j in range(count))),
                    rf"G_2(Y_v={count}) - \sum_{{j<{count}}}\frac{{jk}}{{(1+jk)^2}}")

    xs = sp.Symbol("x", positive=True)
    proof.claim(f"gap series to x^{SERIES_TERMS}",
                sp.series(sp.log(1 + xs) / xs - 1 / (1 + xs), xs, 0, SERIES_TERMS + 1).removeO()
                - sum((-1)**(n + 1) * n * xs**n / (n + 1) for n in range(1, SERIES_TERMS + 1)),
                rf"\Big[\frac{{\log(1+x)}}{{x}} - \frac{{1}}{{1+x}}\Big]_{{x^{SERIES_TERMS}}}"
                rf" - \sum_{{n=1}}^{{{SERIES_TERMS}}}\frac{{(-1)^{{n+1}}nx^n}}{{n+1}}")

    # --- 3. derivatives of eta and kappa in gamma -----------------------------------------------
    gamma = sp.symbols(f"gamma0:{Q}", real=True)
    G = sp.Matrix(N, Q, lambda i, q: sp.Symbol(f"G{i}{q}", real=True))
    e = [sp.exp(sum(G[i, q] * gamma[q] for q in range(Q))) for i in range(N)]
    s1, s2 = sum(e), sum(ei**2 for ei in e)
    eta_of = sp.log(s1)
    kappa_of = sp.log(theta) + sp.log(s2) - 2 * sp.log(s1)
    U = [sum(e[i] * G[i, q] for i in range(N)) for q in range(Q)]
    W = [sum(e[i]**2 * G[i, q] for i in range(N)) for q in range(Q)]
    U2 = [[sum(e[i] * G[i, q] * G[i, r] for i in range(N)) for r in range(Q)] for q in range(Q)]
    W2 = [[sum(e[i]**2 * G[i, q] * G[i, r] for i in range(N)) for r in range(Q)]
          for q in range(Q)]
    ell = [U[q] / s1 for q in range(Q)]
    h = [2 * W[q] / s2 for q in range(Q)]
    ell2 = [[U2[q][r] / s1 - U[q] * U[r] / s1**2 for r in range(Q)] for q in range(Q)]
    h2 = [[4 * W2[q][r] / s2 - 4 * W[q] * W[r] / s2**2 for r in range(Q)] for q in range(Q)]

    for q in range(Q):
        proof.claim(f"deta/dgamma{q}", sp.simplify(sp.diff(eta_of, gamma[q]) - ell[q]),
                    rf"\partial_{{\gamma_{q}}}\eta - \ell_{q}")
        proof.claim(f"dkappa/dgamma{q}",
                    sp.simplify(sp.diff(kappa_of, gamma[q]) - (h[q] - 2 * ell[q])),
                    rf"\partial_{{\gamma_{q}}}\kappa - (h_{q}-2\ell_{q})")
        for r in range(Q):
            proof.claim(f"d2eta/dgamma{q} dgamma{r}",
                        sp.simplify(sp.diff(eta_of, gamma[q], gamma[r]) - ell2[q][r]),
                        rf"\partial^2_{{\gamma_{q}\gamma_{r}}}\eta - \ell^{{(2)}}_{{{q}{r}}}")
            proof.claim(f"d2kappa/dgamma{q} dgamma{r}",
                        sp.simplify(sp.diff(kappa_of, gamma[q], gamma[r])
                                    - (h2[q][r] - 2 * ell2[q][r])),
                        rf"\partial^2_{{\gamma_{q}\gamma_{r}}}\kappa"
                        rf" - (h^{{(2)}}_{{{q}{r}}}-2\ell^{{(2)}}_{{{q}{r}}})")
    return proof
