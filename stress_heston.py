"""
Heston Basket stress test — compares app vs Python benchmark across all cases.
"""
import asyncio, re, numpy as np
from playwright.async_api import async_playwright

FILE = "file:///Users/ishaanchaturvedi/Desktop/options%20pricer2.0/index.html"

def py_heston_basket(hp1, hp2, rho12, strike, tenor, rfr=0.045,
                     q1=0, q2=0, structure='wop', N=80_000, seed=42, label=None):
    T    = tenor / 365
    dt   = T / tenor
    sqDt = np.sqrt(dt)
    rng  = np.random.default_rng(seed)

    r1, r2 = hp1['rhoSV'], hp2['rhoSV']
    L = np.zeros((4,4))
    L[0,0] = 1.0
    L[1,0] = r1;  L[1,1] = np.sqrt(max(0, 1-r1**2))
    L[2,0] = rho12
    L[2,1] = (-rho12*r1/L[1,1]) if L[1,1]>1e-10 else 0
    L[2,2] = np.sqrt(max(0, 1 - L[2,0]**2 - L[2,1]**2))
    L[3,2] = (r2/L[2,2]) if L[2,2]>1e-10 else 0
    if abs(L[3,2]) > 0.9999: L[3,2] = 0.9999*np.sign(L[3,2])
    L[3,3] = np.sqrt(max(0, 1 - L[3,2]**2))

    # Generate all random numbers at once
    Z = rng.standard_normal((4, N, tenor))
    W = np.einsum('ij,jnk->ink', L, Z)
    wS1, wV1, wS2, wV2 = W[0], W[1], W[2], W[3]

    s1 = np.ones(N); s2 = np.ones(N)
    v1 = np.full(N, hp1['v0']); v2 = np.full(N, hp2['v0'])

    for t in range(tenor):
        sv1 = np.sqrt(np.maximum(0, v1)); sv2 = np.sqrt(np.maximum(0, v2))
        s1 *= np.exp((rfr-q1-0.5*v1)*dt + sv1*sqDt*wS1[:,t])
        s2 *= np.exp((rfr-q2-0.5*v2)*dt + sv2*sqDt*wS2[:,t])
        v1  = np.maximum(0, v1 + hp1['kappa']*(hp1['theta']-v1)*dt + hp1['xi']*sv1*sqDt*wV1[:,t])
        v2  = np.maximum(0, v2 + hp2['kappa']*(hp2['theta']-v2)*dt + hp2['xi']*sv2*sqDt*wV2[:,t])

    K = strike
    if   structure=='wop':      pay = np.maximum(K-np.minimum(s1,s2),0)
    elif structure=='woc':      pay = np.maximum(np.minimum(s1,s2)-K,0)
    elif structure=='bop':      pay = np.maximum(K-np.maximum(s1,s2),0)
    elif structure=='boc':      pay = np.maximum(np.maximum(s1,s2)-K,0)
    elif structure=='avg_put':  pay = np.maximum(K-(s1+s2)/2,0)
    elif structure=='avg_call': pay = np.maximum((s1+s2)/2-K,0)

    disc = np.exp(-rfr*T)
    pct  = pay.mean()*disc*100
    se   = pay.std()/np.sqrt(N)*disc*100
    return pct, se

HP1 = dict(kappa=3.0, theta=0.36,  xi=1.2, rhoSV=-0.65, v0=0.36)
HP2 = dict(kappa=5.0, theta=0.028, xi=0.4, rhoSV=-0.4,  v0=0.028)
RHO = 0.641

def mk(**kw): return {**dict(hp1=HP1,hp2=HP2,rho12=RHO,rfr=0.045,tenor=365), **kw}

TESTS = [
    mk(strike=0.70, structure='wop',      label='WoP  K=70%  base'),
    mk(strike=0.70, structure='woc',      label='WoC  K=70%  base'),
    mk(strike=0.70, structure='bop',      label='BoP  K=70%  base'),
    mk(strike=0.70, structure='boc',      label='BoC  K=70%  base'),
    mk(strike=0.70, structure='avg_put',  label='AvgPut  K=70%'),
    mk(strike=0.70, structure='avg_call', label='AvgCall K=70%'),
    mk(strike=0.50, structure='wop', label='WoP  K=50%'),
    mk(strike=0.90, structure='wop', label='WoP  K=90%'),
    mk(strike=1.00, structure='wop', label='WoP  K=100%'),
    mk(strike=1.10, structure='wop', label='WoP  K=110%'),
    mk(strike=0.70, structure='wop', tenor=90,  label='WoP  T=90d'),
    mk(strike=0.70, structure='wop', tenor=180, label='WoP  T=180d'),
    mk(strike=0.70, structure='wop', tenor=730, label='WoP  T=730d'),
    mk(strike=0.70, structure='wop', rho12=-0.5, label='WoP  rho=-0.5'),
    mk(strike=0.70, structure='wop', rho12=0.0,  label='WoP  rho=0.0'),
    mk(strike=0.70, structure='wop', rho12=0.9,  label='WoP  rho=0.9'),
    dict(hp1=dict(kappa=2,theta=0.64,xi=1.5,rhoSV=-0.7,v0=0.64),
         hp2=dict(kappa=3,theta=0.04,xi=0.3,rhoSV=-0.3,v0=0.04),
         rho12=0.3,strike=0.70,tenor=365,rfr=0.045,structure='wop',label='WoP  high-vol regime'),
    dict(hp1=dict(kappa=10,theta=0.001,xi=0.1,rhoSV=-0.1,v0=0.001),
         hp2=dict(kappa=10,theta=0.001,xi=0.1,rhoSV=-0.1,v0=0.001),
         rho12=0.5,strike=0.70,tenor=365,rfr=0.045,structure='wop',label='WoP  near-zero vol'),
    dict(hp1=dict(kappa=3,theta=0.36,xi=1.5,rhoSV=-0.9,v0=0.36),
         hp2=dict(kappa=5,theta=0.028,xi=0.4,rhoSV=-0.8,v0=0.028),
         rho12=0.641,strike=0.50,tenor=365,rfr=0.045,structure='wop',label='WoP  K=50% steep skew'),
    mk(strike=1.30, structure='woc', label='WoC  K=130%'),
    mk(strike=1.30, structure='boc', label='BoC  K=130%'),
]

async def run_app(page, t):
    await page.select_option('#bk-structure', t['structure'])
    await page.select_option('#bk-model', 'heston')
    hp1, hp2 = t['hp1'], t['hp2']
    for fid, val in [
        ('#bk-strike-pct', round(t['strike']*100,2)),
        ('#bk-tenor',      t['tenor']),
        ('#bk-sigma1',     round(np.sqrt(hp1['theta'])*100,2)),
        ('#bk-sigma2',     round(np.sqrt(hp2['theta'])*100,2)),
        ('#bk-q1', 0), ('#bk-q2', 0),
        ('#bk-corr',       t['rho12']),
        ('#bk-notional',   1000000),
        ('#bk-h1-kappa', hp1['kappa']), ('#bk-h1-theta', hp1['theta']),
        ('#bk-h1-xi',    hp1['xi']),    ('#bk-h1-rho',   hp1['rhoSV']),
        ('#bk-h1-v0',    hp1['v0']),
        ('#bk-h2-kappa', hp2['kappa']), ('#bk-h2-theta', hp2['theta']),
        ('#bk-h2-xi',    hp2['xi']),    ('#bk-h2-rho',   hp2['rhoSV']),
        ('#bk-h2-v0',    hp2['v0']),
    ]:
        await page.fill(fid, str(val))
    await page.select_option('#bk-sims', '200000')
    await page.click('#bk-run-btn')
    await page.wait_for_function(
        "document.getElementById('bk-cards').style.display !== 'none'", timeout=120000)
    await asyncio.sleep(0.3)
    txt = await page.inner_text('#bk-cards')
    m = re.search(r'OF NOTIONAL\s*\n([\d.]+)%', txt, re.I)
    return float(m.group(1)) if m else None

async def main():
    print("Pre-computing Python benchmarks (80k paths each)…")
    py_results = []
    for i, t in enumerate(TESTS):
        p, se = py_heston_basket(**{k:v for k,v in t.items() if k!='label'})
        py_results.append((p, se))
        print(f"  [{i+1}/{len(TESTS)}] {t['label']:<35} {p:.4f}% ±{1.96*se:.4f}%")

    print(f"\nRunning app tests…\n")
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page    = await browser.new_page()
        await page.goto(FILE)
        await page.wait_for_load_state('networkidle')
        await page.click("button.win-launcher[onclick*='basket']")
        await asyncio.sleep(0.3)

        passes = fails = 0
        print(f"{'Test':<35} {'App':>8} {'Python':>8} {'±2σ':>7} {'Diff':>8}  Status")
        print('─'*80)

        for t, (py, se) in zip(TESTS, py_results):
            app = await run_app(page, t)
            if app is None:
                print(f"{t['label']:<35} {'N/A':>8} {py:>7.4f}%  ⚠️  could not read")
                continue
            diff = app - py
            tol  = max(0.20, 2*se)
            ok   = abs(diff) < tol
            passes += ok; fails += (not ok)
            print(f"{t['label']:<35} {app:>7.4f}% {py:>7.4f}% ±{1.96*se:.3f}% {diff:>+7.4f}pp  {'✅' if ok else '❌'}")

        await browser.close()

    print('─'*80)
    print(f"\n  {passes}/{passes+fails} PASSED  {'🎉 ALL GOOD' if not fails else f'⚠️  {fails} FAILURES'}")

asyncio.run(main())
