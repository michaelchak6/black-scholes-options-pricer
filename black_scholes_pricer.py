"""
Black-Scholes Options Pricer
=============================
Analytical pricing of European options using the Black-Scholes model.

Features:
- European call/put pricing
- All five Greeks (Δ, Γ, ν, θ, ρ)
- Implied volatility via Brent's root-finding method
- Put-call parity verification
- Break-even analysis
- P&L heatmap
- 8-panel sensitivity visualisations
- Interactive CLI mode

Author: Michael Chak
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import norm
from scipy.optimize import brentq

# ── Styling ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "lines.linewidth":  1.8,
    "font.family":      "monospace",
})

BLUE   = "#58a6ff"
GREEN  = "#3fb950"
ORANGE = "#f78166"
PURPLE = "#bc8cff"
YELLOW = "#e3b341"
RED    = "#ff7b72"


# ── Core Black-Scholes ─────────────────────────────────────────────────────────
def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Analytical Black-Scholes price for a European option.

    Parameters
    ----------
    S           : float  – spot price
    K           : float  – strike price
    T           : float  – time to expiry (years)
    r           : float  – risk-free rate (annual)
    sigma       : float  – volatility (annual)
    option_type : str    – 'call' or 'put'
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ── Greeks ─────────────────────────────────────────────────────────────────────
def greeks(S, K, T, r, sigma, option_type="call"):
    """Returns all five Greeks as a dict."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)

    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega  = S * pdf_d1 * np.sqrt(T) / 100
    theta_call = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta = theta_call if option_type == "call" else (
        theta_call + r * K * np.exp(-r * T) / 365)
    rho = (K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == "call"
           else -K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}


# ── Implied Volatility ─────────────────────────────────────────────────────────
def implied_volatility(market_price, S, K, T, r, option_type="call"):
    """Solves for implied volatility using Brent's root-finding method."""
    objective = lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - market_price
    try:
        return brentq(objective, 1e-6, 10.0, xtol=1e-8)
    except ValueError:
        return np.nan


# ── Put-Call Parity ────────────────────────────────────────────────────────────
def put_call_parity_check(S, K, T, r, sigma):
    """
    Verifies put-call parity: C - P = S - K * e^(-rT)
    Returns call price, put price, LHS, RHS, and the difference.
    """
    call  = black_scholes(S, K, T, r, sigma, "call")
    put   = black_scholes(S, K, T, r, sigma, "put")
    lhs   = call - put
    rhs   = S - K * np.exp(-r * T)
    return call, put, lhs, rhs, abs(lhs - rhs)


# ── Break-Even Analysis ────────────────────────────────────────────────────────
def break_even(S, K, T, r, sigma, option_type="call"):
    """
    Returns the break-even spot price at expiry (premium-adjusted).
    Call break-even: K + premium
    Put break-even:  K - premium
    """
    premium = black_scholes(S, K, T, r, sigma, option_type)
    if option_type == "call":
        return K + premium, premium
    else:
        return K - premium, premium


# ── Visualisation ──────────────────────────────────────────────────────────────
def plot_results(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call"):
    """
    8-panel figure:
      1. Option price vs spot price (call & put)
      2. All Greeks vs spot price
      3. Delta vs spot (call & put)
      4. Vega & Gamma vs volatility
      5. Theta decay vs time to expiry
      6. P&L heatmap (spot × volatility)
      7. Break-even diagram
      8. Implied volatility smile
    """
    price  = black_scholes(S, K, T, r, sigma, option_type)
    g      = greeks(S, K, T, r, sigma, option_type)
    be, prem = break_even(S, K, T, r, sigma, option_type)

    spots  = np.linspace(60, 150, 200)
    sigmas = np.linspace(0.05, 0.80, 200)
    times  = np.linspace(0.01, 2.0, 200)

    fig = plt.figure(figsize=(20, 22))
    fig.suptitle("Black-Scholes Options Pricer  ·  European Options",
                 fontsize=16, fontweight="bold", color="#e6edf3", y=0.98)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    # ── Panel 1: Price vs spot ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    calls = [black_scholes(s, K, T, r, sigma, "call") for s in spots]
    puts  = [black_scholes(s, K, T, r, sigma, "put")  for s in spots]
    ax1.plot(spots, calls, color=BLUE,   lw=2, label="Call")
    ax1.plot(spots, puts,  color=PURPLE, lw=2, label="Put")
    ax1.axvline(K, color="#8b949e", lw=1, linestyle=":", label=f"Strike K={K}")
    ax1.axvline(S, color=GREEN,     lw=1, linestyle="--", label=f"Spot S={S}")
    ax1.set_title("Option Price vs Spot Price", color="#e6edf3")
    ax1.set_xlabel("Spot Price  S")
    ax1.set_ylabel("Option Price")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    # ── Panel 2: All Greeks vs spot ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    greek_series = {k: [] for k in ["delta", "gamma", "vega", "theta"]}
    for s in spots:
        g_ = greeks(s, K, T, r, sigma, option_type)
        for key in greek_series:
            greek_series[key].append(g_[key])
    colours = [BLUE, GREEN, ORANGE, PURPLE]
    for (name, vals), col in zip(greek_series.items(), colours):
        ax2.plot(spots, vals, color=col, lw=1.8, label=name.capitalize())
    ax2.axvline(K, color="#8b949e", lw=1, linestyle=":")
    ax2.axvline(S, color=GREEN,     lw=1, linestyle="--")
    ax2.axhline(0, color="#8b949e", lw=0.8)
    ax2.set_title("Greeks vs Spot Price", color="#e6edf3")
    ax2.set_xlabel("Spot Price  S")
    ax2.set_ylabel("Greek Value")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    # ── Panel 3: Delta comparison call vs put ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    delta_call = [greeks(s, K, T, r, sigma, "call")["delta"] for s in spots]
    delta_put  = [greeks(s, K, T, r, sigma, "put") ["delta"] for s in spots]
    ax3.plot(spots, delta_call, color=BLUE,   lw=2, label="Call Delta")
    ax3.plot(spots, delta_put,  color=PURPLE, lw=2, label="Put Delta")
    ax3.axhline(0,   color="#8b949e", lw=0.8)
    ax3.axhline(0.5, color=BLUE,   lw=0.8, linestyle=":", alpha=0.5)
    ax3.axhline(-0.5,color=PURPLE, lw=0.8, linestyle=":", alpha=0.5)
    ax3.axvline(K,   color="#8b949e", lw=1, linestyle=":")
    ax3.set_title("Delta: Call vs Put", color="#e6edf3")
    ax3.set_xlabel("Spot Price  S")
    ax3.set_ylabel("Delta")
    ax3.legend(fontsize=8)
    ax3.grid(True)

    # ── Panel 4: Vega & Gamma vs volatility ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    vega_v  = [greeks(S, K, T, r, s, option_type)["vega"]  for s in sigmas]
    gamma_v = [greeks(S, K, T, r, s, option_type)["gamma"] for s in sigmas]
    ax4b = ax4.twinx()
    ax4.plot(sigmas * 100, vega_v,  color=ORANGE, lw=2, label="Vega")
    ax4b.plot(sigmas * 100, gamma_v, color=GREEN,  lw=2, linestyle="--", label="Gamma")
    ax4.axvline(sigma * 100, color="#8b949e", lw=1, linestyle=":")
    ax4.set_title("Vega & Gamma vs Volatility", color="#e6edf3")
    ax4.set_xlabel("Volatility  σ (%)")
    ax4.set_ylabel("Vega", color=ORANGE)
    ax4b.set_ylabel("Gamma", color=GREEN)
    ax4b.tick_params(colors="#8b949e")
    ax4b.spines["right"].set_color("#30363d")
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax4.grid(True)

    # ── Panel 5: Theta decay vs time ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    price_t_call = [black_scholes(S, K, t, r, sigma, "call") for t in times]
    price_t_put  = [black_scholes(S, K, t, r, sigma, "put")  for t in times]
    ax5.plot(times, price_t_call, color=BLUE,   lw=2, label="Call")
    ax5.plot(times, price_t_put,  color=PURPLE, lw=2, label="Put")
    ax5.axvline(T, color="#8b949e", lw=1, linestyle=":", label=f"T={T}yr")
    ax5.set_title("Theta Decay: Price vs Time to Expiry", color="#e6edf3")
    ax5.set_xlabel("Time to Expiry  T (years)")
    ax5.set_ylabel("Option Price")
    ax5.legend(fontsize=8)
    ax5.grid(True)

    # ── Panel 6: P&L heatmap ───────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    spot_range = np.linspace(70, 140, 60)
    vol_range  = np.linspace(0.05, 0.60, 60)
    SS, VV     = np.meshgrid(spot_range, vol_range)
    PnL = np.vectorize(
        lambda s, v: black_scholes(s, K, T, r, v, option_type) - prem
    )(SS, VV)
    norm_pnl = TwoSlopeNorm(vmin=PnL.min(), vcenter=0, vmax=PnL.max())
    im = ax6.contourf(SS, VV * 100, PnL, levels=40, cmap="RdYlGn", norm=norm_pnl)
    plt.colorbar(im, ax=ax6, label="P&L (£)")
    ax6.axvline(S,         color="white", lw=1, linestyle="--", label=f"Spot={S}")
    ax6.axvline(K,         color=YELLOW,  lw=1, linestyle=":",  label=f"Strike={K}")
    ax6.axhline(sigma*100, color="white", lw=1, linestyle=":",  label=f"σ={sigma*100:.0f}%")
    ax6.set_title(f"P&L Heatmap  ({option_type.capitalize()})", color="#e6edf3")
    ax6.set_xlabel("Spot Price  S")
    ax6.set_ylabel("Volatility  σ (%)")
    ax6.legend(fontsize=7)

    # ── Panel 7: Break-even diagram ────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    expiry_spots = np.linspace(60, 150, 300)
    if option_type == "call":
        payoff = np.maximum(expiry_spots - K, 0) - prem
    else:
        payoff = np.maximum(K - expiry_spots, 0) - prem
    ax7.plot(expiry_spots, payoff, color=BLUE, lw=2, label="P&L at Expiry")
    ax7.axhline(0,   color="#8b949e", lw=1)
    ax7.axvline(K,   color=ORANGE, lw=1, linestyle=":", label=f"Strike K={K}")
    ax7.axvline(be,  color=GREEN,  lw=1.5, linestyle="--",
                label=f"Break-even={be:.2f}")
    ax7.axvline(S,   color=YELLOW, lw=1, linestyle=":", label=f"Spot S={S}")
    ax7.fill_between(expiry_spots, payoff, 0,
                     where=(payoff > 0), alpha=0.2, color=GREEN)
    ax7.fill_between(expiry_spots, payoff, 0,
                     where=(payoff < 0), alpha=0.2, color=RED)
    ax7.set_title(f"Break-Even Analysis  ({option_type.capitalize()})", color="#e6edf3")
    ax7.set_xlabel("Spot Price at Expiry")
    ax7.set_ylabel("Profit / Loss")
    ax7.legend(fontsize=8)
    ax7.grid(True)

    # ── Panel 8: Implied volatility smile ─────────────────────────────────────
    ax8     = fig.add_subplot(gs[3, 1])
    strikes = np.linspace(70, 135, 40)
    smile_vol   = 0.2 + 0.3 * ((strikes - S) / S) ** 2 - 0.05 * (strikes - S) / S
    mkt_prices  = [black_scholes(S, k, T, r, v, "call")
                   for k, v in zip(strikes, smile_vol)]
    implied_vols = [implied_volatility(p, S, k, T, r, "call") * 100
                    for p, k in zip(mkt_prices, strikes)]
    ax8.plot(strikes, implied_vols, color=YELLOW, lw=2, label="Implied Vol")
    ax8.axhline(sigma * 100, color="#8b949e", lw=1, linestyle="--",
                label=f"Flat σ={sigma*100:.0f}%")
    ax8.axvline(S, color=GREEN,  lw=1, linestyle=":", label=f"ATM S={S}")
    ax8.axvline(K, color=ORANGE, lw=1, linestyle=":", label=f"Strike K={K}")
    ax8.set_title("Implied Volatility Smile", color="#e6edf3")
    ax8.set_xlabel("Strike Price  K")
    ax8.set_ylabel("Implied Volatility (%)")
    ax8.legend(fontsize=8)
    ax8.grid(True)

    plt.savefig("black_scholes_output.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("Saved → black_scholes_output.png")


# ── Summary Print ──────────────────────────────────────────────────────────────
def print_summary(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call"):
    price        = black_scholes(S, K, T, r, sigma, option_type)
    g            = greeks(S, K, T, r, sigma, option_type)
    be, prem     = break_even(S, K, T, r, sigma, option_type)
    call, put, lhs, rhs, parity_err = put_call_parity_check(S, K, T, r, sigma)
    moneyness    = "ATM" if S == K else ("ITM" if (
        (option_type == "call" and S > K) or
        (option_type == "put"  and S < K)) else "OTM")

    print("\n" + "═" * 56)
    print(f"  BLACK-SCHOLES PRICER  ·  European {option_type.upper()}")
    print("═" * 56)
    print(f"  Spot S={S}  |  Strike K={K}  |  T={T}yr")
    print(f"  r={r*100:.1f}%  |  σ={sigma*100:.0f}%  |  {moneyness}")
    print("─" * 56)
    print(f"  Price          : {price:.4f}")
    print(f"  Break-even     : {be:.4f}  (premium = {prem:.4f})")
    print("─" * 56)
    print(f"  Δ Delta        : {g['delta']:+.4f}")
    print(f"  Γ Gamma        : {g['gamma']:+.4f}")
    print(f"  ν Vega         : {g['vega']:+.4f}  (per 1% Δσ)")
    print(f"  θ Theta        : {g['theta']:+.4f}  (per day)")
    print(f"  ρ Rho          : {g['rho']:+.4f}  (per 1% Δr)")
    print("─" * 56)
    print(f"  Put-Call Parity:")
    print(f"    Call={call:.4f}  Put={put:.4f}")
    print(f"    C - P = {lhs:.4f}  |  S - Ke^(-rT) = {rhs:.4f}")
    print(f"    Parity error  : {parity_err:.2e}  ✓" if parity_err < 1e-6
          else f"    Parity error  : {parity_err:.6f}  ✗")
    print("═" * 56 + "\n")

    plot_results(S, K, T, r, sigma, option_type)


# ── Interactive CLI ────────────────────────────────────────────────────────────
def interactive_mode():
    print("\n" + "═" * 56)
    print("  BLACK-SCHOLES PRICER  ·  Interactive Mode")
    print("  Press Enter to use default values shown in [ ]")
    print("═" * 56)

    def get(prompt, default, cast):
        val = input(f"  {prompt} [{default}]: ").strip()
        return cast(val) if val else default

    S    = get("Spot price          S", 100,    float)
    K    = get("Strike price        K", 100,    float)
    T    = get("Time to expiry (yr) T", 1.0,    float)
    r    = get("Risk-free rate      r", 0.05,   float)
    sigma= get("Volatility          σ", 0.20,   float)
    opt  = get("Option type (call/put)", "call", str).lower()

    print_summary(S, K, T, r, sigma, opt)


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        print_summary(
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            sigma=0.20,
            option_type="call",
        )
