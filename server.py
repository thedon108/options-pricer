"""
Backend server for Options Pricer basket-quote endpoint.
- Realized vol: 2yr daily log-returns annualised (×√252)
- Implied vol: two flavours, both interpolated in total-variance space (σ²T)
  to the requested tenor:
    * ATM IV        — strike closest to spot
    * Strike IV     — strike closest to k%×spot (uses OTM puts below spot,
                      calls above), i.e. the vol AT the deal's strike (skew)
- Correlation: 2yr daily log-returns

Usage:
    pip install flask flask-cors yfinance numpy scipy
    python server.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, least_squares
import datetime

_GL_X, _GL_W = np.polynomial.legendre.leggauss(128)

app = Flask(__name__)
CORS(app)


def bs_price(S, K, T, r, sigma, is_call):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def solve_iv(price, S, K, T, r, is_call):
    intrinsic = max(S - K * np.exp(-r * T), 0.0) if is_call else max(K * np.exp(-r * T) - S, 0.0)
    if price <= intrinsic + 1e-6:
        return None
    try:
        iv = brentq(lambda s: bs_price(S, K, T, r, s, is_call) - price,
                    1e-4, 10.0, xtol=1e-6, maxiter=200)
        return iv if 0.01 < iv < 5.0 else None
    except Exception:
        return None


def heston_call(S, K, T, r, kappa, theta, xi, rho, v0):
    """Semi-analytic Heston call ('little trap' formulation), Gauss-Legendre on (0, 200)."""
    umax = 200.0
    u = 0.5 * umax * (_GL_X + 1)
    w = 0.5 * umax * _GL_W
    lnS, lnK = np.log(S), np.log(K)
    P = []
    for j in (1, 2):
        if j == 1:
            uj, bj = 0.5, kappa - rho * xi
        else:
            uj, bj = -0.5, kappa
        a = kappa * theta
        iu = 1j * u
        d = np.sqrt((rho * xi * iu - bj)**2 - xi**2 * (2 * uj * iu - u**2))
        g2 = (bj - rho * xi * iu - d) / (bj - rho * xi * iu + d)
        edT = np.exp(-d * T)
        C = r * iu * T + a / xi**2 * ((bj - rho * xi * iu - d) * T
                                      - 2 * np.log((1 - g2 * edT) / (1 - g2)))
        D = (bj - rho * xi * iu - d) / xi**2 * ((1 - edT) / (1 - g2 * edT))
        f = np.exp(C + D * v0 + iu * lnS)
        integrand = np.real(np.exp(-iu * lnK) * f / iu)
        P.append(0.5 + (w * integrand).sum() / np.pi)
    return S * P[0] - K * np.exp(-r * T) * P[1]


def calibrate_heston(atm_iv, k_iv, k_pct, spot, T, rfr, hist_vol):
    """
    Fit (v=θ=v₀, ρ_sv, ξ) with κ=2 to the ATM and strike IV points.
    Falls back to realized vol level with typical equity skew if no IVs.
    Returns dict + 'source' tag.
    """
    kappa = 2.0
    if atm_iv is None and k_iv is None:
        v = hist_vol**2
        return dict(kappa=kappa, theta=v, xi=0.6, rhoSV=-0.6, v0=v, source='hist')

    level_iv = atm_iv if atm_iv is not None else k_iv
    targets = [(spot, level_iv)]
    K_abs = spot * k_pct / 100.0
    if k_iv is not None and abs(k_pct - 100) > 1:
        targets.append((K_abs, k_iv))

    def resid(p):
        v, rho, xi = p
        out = []
        for K, tiv in targets:
            c = heston_call(spot, K, T, rfr, kappa, v, xi, rho, v)
            iv = solve_iv(c, spot, K, T, rfr, True)
            out.append((iv - tiv) if iv is not None else 1.0)
        return out

    try:
        res = least_squares(resid, x0=[level_iv**2, -0.6, 0.8],
                            bounds=([1e-4, -0.999, 0.05], [4.0, 0.5, 3.0]),
                            xtol=1e-10, ftol=1e-10)
        v, rho, xi = res.x
        return dict(kappa=kappa, theta=float(v), xi=float(xi),
                    rhoSV=float(rho), v0=float(v), source='smile')
    except Exception:
        v = level_iv**2
        return dict(kappa=kappa, theta=v, xi=0.6, rhoSV=-0.6, v0=v, source='atm-fallback')


def iv_from_chain(chain, T, spot, K_target, rfr):
    """IV at the listed strike closest to K_target. OTM side for liquidity."""
    use_calls = K_target >= spot
    df = chain.calls if use_calls else chain.puts
    df = df[(df['bid'] > 0) & (df['ask'] > 0)]
    if df.empty:
        return None
    idx = (df['strike'] - K_target).abs().idxmin()
    row = df.loc[idx]
    # Reject if nearest listed strike is far from what we asked for
    if abs(row['strike'] - K_target) / spot > 0.10:
        return None
    mid = (row['bid'] + row['ask']) / 2.0
    return solve_iv(mid, spot, row['strike'], T, rfr, use_calls)


def collect_ivs(tk, spot, target_T, rfr, k_pct):
    """
    One pass over expiries; returns (atm_iv, strike_iv), each interpolated
    in total-variance space to target_T (or None).
    """
    K_strike = spot * k_pct / 100.0
    atm_pts, k_pts = [], []
    try:
        exps = tk.options
    except Exception:
        return None, None
    if not exps:
        return None, None

    today = datetime.date.today()
    for exp in exps:
        try:
            days = (datetime.date.fromisoformat(exp) - today).days
        except ValueError:
            continue
        if days < 7:
            continue
        T = days / 365.0
        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue
        iv_a = iv_from_chain(chain, T, spot, spot, rfr)
        iv_k = iv_from_chain(chain, T, spot, K_strike, rfr)
        if iv_a:
            atm_pts.append((T, iv_a))
        if iv_k:
            k_pts.append((T, iv_k))
        # Stop once both series bracket the target tenor
        done_a = len(atm_pts) >= 2 and atm_pts[-1][0] >= target_T
        done_k = len(k_pts) >= 2 and k_pts[-1][0] >= target_T
        if done_a and done_k:
            break

    def interp(points):
        if not points:
            return None
        points.sort(key=lambda x: x[0])
        Ts = np.array([p[0] for p in points])
        tv = np.array([p[1]**2 * p[0] for p in points])  # total variance
        if target_T <= Ts[0]:
            iv = np.sqrt(tv[0] / Ts[0])
        elif target_T >= Ts[-1]:
            iv = np.sqrt(tv[-1] / Ts[-1])
        else:
            iv = np.sqrt(np.interp(target_T, Ts, tv) / target_T)
        iv = float(iv)
        return iv if 0.01 < iv < 5.0 else None

    return interp(atm_pts), interp(k_pts)


@app.route('/basket-quote')
def basket_quote():
    t1    = request.args.get('t1', '').upper().strip()
    t2    = request.args.get('t2', '').upper().strip()
    rfr   = float(request.args.get('rfr', 0.045))
    tenor = int(request.args.get('tenor', 365))     # calendar days
    k_pct = float(request.args.get('k', 100))       # strike as % of spot

    if not t1 or not t2:
        return jsonify(error='Missing tickers'), 400

    target_T = tenor / 365.0

    try:
        data = yf.download([t1, t2], period='2y', interval='1d',
                           auto_adjust=True, progress=False)['Close']
    except Exception as e:
        return jsonify(error=f'Download failed: {e}'), 500

    if data.empty or t1 not in data.columns or t2 not in data.columns:
        return jsonify(error='No data for one or both tickers'), 404

    s1 = data[t1].dropna()
    s2 = data[t2].dropna()
    if len(s1) < 20:
        return jsonify(error=f'Not enough price history for {t1}'), 404
    if len(s2) < 20:
        return jsonify(error=f'Not enough price history for {t2}'), 404

    # 2yr annualised realized vol + correlation
    r1 = np.log(s1 / s1.shift(1)).dropna()
    r2 = np.log(s2 / s2.shift(1)).dropna()
    sigma1_hist = float(r1.std() * np.sqrt(252))
    sigma2_hist = float(r2.std() * np.sqrt(252))
    common = r1.index.intersection(r2.index)
    corr = float(np.corrcoef(r1.loc[common], r2.loc[common])[0, 1]) if len(common) >= 20 else None

    spot1 = float(s1.iloc[-1])
    spot2 = float(s2.iloc[-1])

    tk1, tk2 = yf.Ticker(t1), yf.Ticker(t2)

    def safe_name(tk, fallback):
        try:
            info = tk.info
            return info.get('shortName') or info.get('longName') or fallback
        except Exception:
            return fallback

    name1 = safe_name(tk1, t1)
    name2 = safe_name(tk2, t2)

    def trailing_div_yield(tk, spot):
        """Dividends paid over the last 365 days / spot. Robust to yfinance quirks."""
        try:
            divs = tk.dividends
            if divs is None or divs.empty:
                return 0.0
            cutoff = divs.index.max() - np.timedelta64(365, 'D')
            paid = float(divs[divs.index > cutoff].sum())
            y = paid / spot
            return y if 0 <= y < 0.25 else 0.0
        except Exception:
            return 0.0

    q1 = trailing_div_yield(tk1, spot1)
    q2 = trailing_div_yield(tk2, spot2)

    iv1, kiv1 = collect_ivs(tk1, spot1, target_T, rfr, k_pct)
    iv2, kiv2 = collect_ivs(tk2, spot2, target_T, rfr, k_pct)

    # Auto-calibrate Heston to the smile points (level + skew)
    h1 = calibrate_heston(iv1, kiv1, k_pct, spot1, target_T, rfr, sigma1_hist)
    h2 = calibrate_heston(iv2, kiv2, k_pct, spot2, target_T, rfr, sigma2_hist)

    return jsonify(
        h1=h1, h2=h2,                             # auto-calibrated Heston params
        sigma1=sigma1_hist, sigma2=sigma2_hist,   # 2yr realized
        iv1=iv1, iv2=iv2,                         # ATM IV @ tenor
        kiv1=kiv1, kiv2=kiv2,                     # IV @ strike moneyness @ tenor
        corr=corr,
        q1=q1, q2=q2,                             # trailing 12m dividend yield
        spot1=spot1, spot2=spot2,
        name1=name1, name2=name2,
        tenor_days=tenor, k_pct=k_pct,
    )


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8900))
    print(f"Options Pricer backend running on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
