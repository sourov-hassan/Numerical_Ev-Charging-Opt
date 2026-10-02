"""
EV Charging Cost Optimization — Final Publication Version
=========================================================
Fixes applied:
  - SOC clipped at SOC_max = 0.8 pu
  - RK4 stages fully derived; CC-CV taper makes Euler/RK4 diverge
  - Grid-side objective: P_grid = max(0, P_EV - P_REN)
  - Consistent 0.2 -> 0.8 SOC window -> 4.51 h charge
  - Off-peak window = 23:00-05:00 everywhere
  - Wind speed v(t) fed through cubic model
  - Flat baseline 9.6 BDT/kWh derived from BERC
  - Realistic inverter cap of 6 kW on combined renewables
  - Euler vs RK4 shown at coarse dt where divergence is visible
  - Ablation study: TOU-only vs renewables-only vs both
  - Sensitivity: battery capacity, charger power, tariff, renewables

Produces:
  fig_soc.png          SOC trajectory (clipped)
  fig_soc_zoom.png     Zoom showing Euler vs RK4 divergence
  fig_power.png        PV, Wind, Net demand
  fig_tariff.png       TOU tariff
  fig_cost.png         Cost vs start time
  fig_ablation.png     Bar chart of scenario comparison
  fig_sensitivity.png  4-panel sensitivity
  results_table.csv
  soc_error_table.csv
  ablation_table.csv
  sensitivity_table.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 1. SYSTEM PARAMETERS
# ============================================================
SOC_0    = 0.2
SOC_max  = 0.8
SOC_min  = 0.2
E_batt   = 50.0            # kWh
eta      = 0.95
P_ev     = 7.0             # kW charger rating

E_usable = (SOC_max - SOC_0) * E_batt     # 30.00 kWh
E_grid   = E_usable / eta                  # 31.58 kWh
T_charge = E_grid / P_ev                   # 4.51 h

FLAT_TARIFF          = 9.60    # BDT/kWh (BERC 0.0906 USD/kWh x 106 BDT/USD)
P_REN_INVERTER_LIMIT = 6.0     # kW combined PV+wind delivery cap
PV_RATED             = 6.0     # kW
WIND_RATED           = 5.0     # kW

print(f"Usable energy : {E_usable:.2f} kWh")
print(f"Grid energy   : {E_grid:.2f} kWh")
print(f"Charging time : {T_charge:.2f} h")

# ============================================================
# 2. MODELS
# ============================================================
def tariff(t, scale=1.0):
    """TOU tariff (BDT/kWh). Off-peak 23-05, Mid 05-17, Peak 17-23."""
    h = np.asarray(t) % 24
    base = np.where((h >= 23) | (h < 5), 5.0,
           np.where((h >= 5) & (h < 17), 8.0, 12.0))
    return base * scale


def pv_power(t, P_rated=PV_RATED, G_std=1000.0, G_peak=850.0, scale=1.0):
    """Bell-shaped PV output with irradiance profile G(t)."""
    t = np.asarray(t, dtype=float)
    G = np.where((t >= 6) & (t <= 18),
                 G_peak * np.sin(np.pi * (t - 6) / 12), 0.0)
    return P_rated * G / G_std * scale


def wind_speed(t, v_mean=8.0, v_amp=3.0, phase=4.0):
    """Diurnal wind speed in m/s."""
    return v_mean + v_amp * np.sin(2 * np.pi * (np.asarray(t, float) - phase) / 24.0)


def wind_power(t, v_ci=3.0, v_r=12.0, v_co=25.0, P_rated=WIND_RATED, scale=1.0):
    """Piecewise cubic wind power per Eq. (6)."""
    v = wind_speed(t)
    P = np.zeros_like(v, dtype=float)
    m = (v >= v_ci) & (v < v_r)
    P[m] = P_rated * (v[m]**3 - v_ci**3) / (v_r**3 - v_ci**3)
    m = (v >= v_r) & (v < v_co)
    P[m] = P_rated
    return P * scale


def renewable_power(t, ren_scale=1.0):
    """Combined PV + wind, capped by inverter rating."""
    total = pv_power(t, scale=ren_scale) + wind_power(t, scale=ren_scale)
    return np.minimum(total, P_REN_INVERTER_LIMIT)


# ============================================================
# 3. BATTERY — Euler vs RK4 with CC-CV taper
# ============================================================
def soc_rhs(soc, P_max=P_ev, soc_knee=0.70, soc_max=0.8):
    """CC-CV charging: full power below knee, linear taper above."""
    if soc <= soc_knee:
        P = P_max
    elif soc < soc_max:
        P = P_max * (soc_max - soc) / (soc_max - soc_knee)
    else:
        P = 0.0
    return eta * P / E_batt


def integrate_soc(soc0, t_arr, method="rk4", P_max=P_ev):
    soc = np.zeros_like(t_arr, dtype=float)
    soc[0] = soc0
    for n in range(len(t_arr) - 1):
        if soc[n] >= SOC_max:
            soc[n + 1] = soc[n]
            continue
        dt = t_arr[n + 1] - t_arr[n]
        if method == "euler":
            soc_next = soc[n] + dt * soc_rhs(soc[n], P_max)
        else:
            k1 = soc_rhs(soc[n], P_max)
            k2 = soc_rhs(soc[n] + dt * k1 / 2, P_max)
            k3 = soc_rhs(soc[n] + dt * k2 / 2, P_max)
            k4 = soc_rhs(soc[n] + dt * k3, P_max)
            soc_next = soc[n] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        soc[n + 1] = min(soc_next, SOC_max)
    return soc


# ============================================================
# 4. COST — grid-side objective with renewables
# ============================================================
def charging_cost(t_s, use_ren=True, T=None, N=800,
                  tariff_scale=1.0, ren_scale=1.0, P_charger=P_ev):
    """Grid-side charging cost over [t_s, t_s + T]."""
    if T is None:
        T = (SOC_max - SOC_0) * E_batt / (eta * P_charger)
    t = np.linspace(t_s, t_s + T, N)
    if use_ren:
        P_grid = np.clip(P_charger - renewable_power(t, ren_scale), 0, None)
    else:
        P_grid = np.full_like(t, P_charger)
    return np.trapezoid(P_grid * tariff(t, tariff_scale), t), t, P_grid


# ============================================================
# 5. BASELINE
# ============================================================
C_base = E_grid * FLAT_TARIFF
print(f"\nFlat baseline cost: {C_base:.2f} BDT")

# ============================================================
# 6. OPTIMIZATION — sweep start time
# ============================================================
t_starts    = np.arange(0, 24, 0.25)
costs_ren   = np.array([charging_cost(ts, use_ren=True)[0]  for ts in t_starts])
costs_noren = np.array([charging_cost(ts, use_ren=False)[0] for ts in t_starts])

i_ren   = int(np.argmin(costs_ren))
i_noren = int(np.argmin(costs_noren))
C_ren   = costs_ren[i_ren]
C_noren = costs_noren[i_noren]

print(f"\nOptimal start (with REN)   : {t_starts[i_ren]:.2f} h  -> {C_ren:.2f} BDT")
print(f"Optimal start (no REN)     : {t_starts[i_noren]:.2f} h  -> {C_noren:.2f} BDT")
print(f"Reduction vs flat (with REN): {100*(C_base-C_ren)/C_base:.2f}%")
print(f"Reduction vs flat (no REN)  : {100*(C_base-C_noren)/C_base:.2f}%")

C_unopt_TOU = charging_cost(17.0, use_ren=False)[0]
print(f"Unoptimized TOU (17:00)    : {C_unopt_TOU:.2f} BDT")

# ============================================================
# 7. FIGURES — SOC
# ============================================================
plt.rcParams.update({"font.size": 9})

# --- Fig SOC full trajectory ---
fig, ax = plt.subplots(figsize=(3.5, 3))
t_soc = np.arange(0, 24.01, 0.05)
soc_e = integrate_soc(SOC_0, t_soc, method="euler")
soc_r = integrate_soc(SOC_0, t_soc, method="rk4")
ax.plot(t_soc, soc_e, 'r--', lw=1.0, label='Euler')
ax.plot(t_soc, soc_r, 'b-',  lw=2.0, alpha=0.7, label='RK4')
ax.axhline(SOC_max, color='k', ls=':', lw=0.8)
ax.axhline(SOC_min, color='k', ls=':', lw=0.8)
ax.set_xlabel('Time (h)'); ax.set_ylabel('SOC (pu)')
ax.set_title('SOC: Euler vs RK4 (clipped at 0.8)')
ax.set_ylim(0, 1.0); ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_soc.png', dpi=300); plt.close()

# --- Fig SOC zoom: use dt = 0.5 h so Euler/RK4 visibly diverge ---
fig, ax = plt.subplots(figsize=(3.5, 3))
t_zoom = np.arange(3.5, 6.01, 0.01)
soc_ref_zoom = integrate_soc(SOC_0, t_zoom, method="rk4")

t_coarse = np.arange(3.5, 6.01, 0.5)
soc_e_c  = integrate_soc(SOC_0, t_coarse, method="euler")
soc_r_c  = integrate_soc(SOC_0, t_coarse, method="rk4")

ax.plot(t_zoom, soc_ref_zoom, 'k-', lw=1.0, label='Reference (dt=0.01h)')
ax.plot(t_coarse, soc_e_c, 'r--o', lw=1.3, ms=4, label='Euler (dt=0.5h)')
ax.plot(t_coarse, soc_r_c, 'b-s',  lw=1.3, ms=4, label='RK4 (dt=0.5h)')
ax.set_xlabel('Time (h)'); ax.set_ylabel('SOC (pu)')
ax.set_title('Zoom: SOC near CC-CV taper (dt = 0.5 h)')
ax.legend(loc='lower right', fontsize=7)
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_soc_zoom.png', dpi=300); plt.close()

# --- Fig power profiles ---
fig, ax = plt.subplots(figsize=(3.5, 3))
t = np.linspace(0, 24, 800)
ax.plot(t, pv_power(t),   label='PV')
ax.plot(t, wind_power(t), label='Wind')
ax.plot(t, P_ev - renewable_power(t), label='Net demand')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Time (h)'); ax.set_ylabel('Power (kW)')
ax.set_title('PV, Wind, and Net Power')
ax.legend(loc='upper right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_power.png', dpi=300); plt.close()

# --- Fig TOU tariff ---
fig, ax = plt.subplots(figsize=(3.5, 3))
t = np.linspace(0, 24, 2000)
ax.plot(t, tariff(t), 'b-', lw=1.5)
ax.set_xlabel('Time (h)'); ax.set_ylabel('Tariff (BDT/kWh)')
ax.set_title('Time-of-Use Tariff')
ax.set_ylim(4, 13); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_tariff.png', dpi=300); plt.close()

# --- Fig cost vs start time ---
fig, ax = plt.subplots(figsize=(3.5, 3))
ax.plot(t_starts, costs_ren,   label='With REN', lw=1.5)
ax.plot(t_starts, costs_noren, '--', label='No REN', lw=1.5)
ax.axhline(C_base, color='r', ls=':', label='Flat baseline')
ax.scatter(t_starts[i_ren],   C_ren,   color='k', zorder=5, s=30)
ax.scatter(t_starts[i_noren], C_noren, color='g', zorder=5, s=30)
ax.set_xlabel('Charging start time (h)')
ax.set_ylabel('Cost (BDT)')
ax.set_title('Cost vs Start Time')
ax.legend(loc='upper center', fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_cost.png', dpi=300); plt.close()

# ============================================================
# 8. SOC ERROR TABLE
# ============================================================
t_ref  = np.arange(0, 6.01, 0.001)
soc_ref = integrate_soc(SOC_0, t_ref, method="rk4")

err_rows = []
for dt in [0.05, 0.10, 0.25, 0.50]:
    t_c = np.arange(0, 6.0 + dt, dt)
    soc_e_c = integrate_soc(SOC_0, t_c, method="euler")
    soc_r_c = integrate_soc(SOC_0, t_c, method="rk4")
    soc_ref_c = np.interp(t_c, t_ref, soc_ref)
    err_rows.append({
        "dt (h)": dt,
        "Euler max |error| (pu)": np.max(np.abs(soc_e_c - soc_ref_c)),
        "RK4 max |error| (pu)":   np.max(np.abs(soc_r_c - soc_ref_c)),
    })
df_err = pd.DataFrame(err_rows)
print("\n--- Euler vs RK4 accuracy ---")
print(df_err.to_string(index=False, float_format="%.5f"))
df_err.to_csv("soc_error_table.csv", index=False)

# ============================================================
# 9. ABLATION STUDY
# ============================================================
# Scenarios:
#   S1: Flat tariff, no renewables, no optimization
#   S2: TOU tariff, no renewables, optimized start
#   S3: TOU tariff, renewables, but random (17:00) start
#   S4: TOU tariff, renewables, optimized start (full framework)

C_S1 = C_base
C_S2 = C_noren
C_S3 = charging_cost(17.0, use_ren=True)[0]       # renewables but bad schedule
C_S4 = C_ren

ablation_rows = [
    ["S1: Flat tariff, no REN, no opt.",      f"{C_S1:.2f}", "—"],
    ["S2: TOU, no REN, optimized",             f"{C_S2:.2f}",
     f"{100*(C_S1-C_S2)/C_S1:.2f}%"],
    ["S3: TOU + REN, unoptimized (17:00)",     f"{C_S3:.2f}",
     f"{100*(C_S1-C_S3)/C_S1:.2f}%"],
    ["S4: TOU + REN, optimized (full)",        f"{C_S4:.2f}",
     f"{100*(C_S1-C_S4)/C_S1:.2f}%"],
]
df_abl = pd.DataFrame(ablation_rows,
                      columns=["Scenario", "Cost (BDT)", "Reduction vs S1"])
print("\n--- Ablation study ---")
print(df_abl.to_string(index=False))
df_abl.to_csv("ablation_table.csv", index=False)

# Ablation bar chart
fig, ax = plt.subplots(figsize=(4.0, 3))
labels = ["S1\nFlat, no REN", "S2\nTOU only", "S3\nREN only", "S4\nFull framework"]
values = [C_S1, C_S2, C_S3, C_S4]
colors = ['gray', 'steelblue', 'orange', 'green']
bars = ax.bar(labels, values, color=colors, alpha=0.85)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v + 8, f"{v:.0f}",
            ha='center', fontsize=8)
ax.set_ylabel('Cost (BDT)')
ax.set_title('Ablation: Contribution of TOU and REN')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig('fig_ablation.png', dpi=300); plt.close()

# ============================================================
# 10. SENSITIVITY ANALYSIS
# ============================================================
def best_cost(E_batt_, P_charger_, tariff_scale_, ren_scale_):
    """Return optimized cost under modified parameters."""
    T_ch = (SOC_max - SOC_0) * E_batt_ / (eta * P_charger_)
    E_grid_ = T_ch * P_charger_
    C_base_ = E_grid_ * FLAT_TARIFF * tariff_scale_
    t_s_ = np.arange(0, 24, 0.25)
    costs_ = np.array([
        charging_cost(ts, use_ren=True, T=T_ch, N=400,
                      tariff_scale=tariff_scale_,
                      ren_scale=ren_scale_,
                      P_charger=P_charger_)[0]
        for ts in t_s_
    ])
    return C_base_, costs_.min()

# Battery capacity sweep
E_batt_list = [30, 40, 50, 60, 80]
results_batt = [best_cost(E, P_ev, 1.0, 1.0) for E in E_batt_list]

# Charger power sweep
P_list = [3.3, 7.0, 11.0, 22.0]
results_P = [best_cost(50, P, 1.0, 1.0) for P in P_list]

# Tariff scale sweep
tariff_list = [0.7, 0.85, 1.0, 1.15, 1.3]
results_tar = [best_cost(50, P_ev, s, 1.0) for s in tariff_list]

# Renewable scale sweep (0 = no renewables)
ren_list = [0.0, 0.5, 0.75, 1.0, 1.25]
results_ren = [best_cost(50, P_ev, 1.0, s) for s in ren_list]

# Plot 4-panel sensitivity
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

# Battery
base_b = [r[0] for r in results_batt]
opt_b  = [r[1] for r in results_batt]
axes[0,0].plot(E_batt_list, base_b, 'r--o', label='Baseline')
axes[0,0].plot(E_batt_list, opt_b,  'b-o',  label='Optimized')
axes[0,0].set_xlabel('Battery capacity (kWh)')
axes[0,0].set_ylabel('Cost (BDT)')
axes[0,0].set_title('Sensitivity: Battery size')
axes[0,0].legend(fontsize=7); axes[0,0].grid(alpha=0.3)

# Charger power
base_p = [r[0] for r in results_P]
opt_p  = [r[1] for r in results_P]
axes[0,1].plot(P_list, base_p, 'r--o', label='Baseline')
axes[0,1].plot(P_list, opt_p,  'b-o',  label='Optimized')
axes[0,1].set_xlabel('Charger power (kW)')
axes[0,1].set_ylabel('Cost (BDT)')
axes[0,1].set_title('Sensitivity: Charger power')
axes[0,1].legend(fontsize=7); axes[0,1].grid(alpha=0.3)

# Tariff scale
base_t = [r[0] for r in results_tar]
opt_t  = [r[1] for r in results_tar]
axes[1,0].plot(tariff_list, base_t, 'r--o', label='Baseline')
axes[1,0].plot(tariff_list, opt_t,  'b-o',  label='Optimized')
axes[1,0].set_xlabel('Tariff scale factor')
axes[1,0].set_ylabel('Cost (BDT)')
axes[1,0].set_title('Sensitivity: Tariff level')
axes[1,0].legend(fontsize=7); axes[1,0].grid(alpha=0.3)

# Renewable scale
base_r = [r[0] for r in results_ren]
opt_r  = [r[1] for r in results_ren]
axes[1,1].plot(ren_list, base_r, 'r--o', label='Baseline')
axes[1,1].plot(ren_list, opt_r,  'b-o',  label='Optimized')
axes[1,1].set_xlabel('Renewable output scale')
axes[1,1].set_ylabel('Cost (BDT)')
axes[1,1].set_title('Sensitivity: Renewable availability')
axes[1,1].legend(fontsize=7); axes[1,1].grid(alpha=0.3)

plt.tight_layout(); plt.savefig('fig_sensitivity.png', dpi=300); plt.close()

# Save sensitivity table
sens_rows = []
for label, xs, res in [
    ("Battery (kWh)",       E_batt_list, results_batt),
    ("Charger power (kW)",  P_list,      results_P),
    ("Tariff scale",        tariff_list, results_tar),
    ("Renewable scale",     ren_list,    results_ren),
]:
    for x, (cb, co) in zip(xs, res):
        sens_rows.append({
            "Parameter": label, "Value": x,
            "Baseline (BDT)": round(cb, 2),
            "Optimized (BDT)": round(co, 2),
            "Reduction (%)": round(100*(cb-co)/cb, 2) if cb > 0 else 0.0,
        })
df_sens = pd.DataFrame(sens_rows)
df_sens.to_csv("sensitivity_table.csv", index=False)

# ============================================================
# 11. FINAL RESULTS TABLE
# ============================================================
rows = [
    ["Baseline (flat 9.6 BDT/kWh)",        f"{C_base:.2f}",     "—"],
    ["Unoptimized TOU (17:00 start)",       f"{C_unopt_TOU:.2f}",
     f"{100*(C_base-C_unopt_TOU)/C_base:.2f}%"],
    ["Optimized TOU, no REN",               f"{C_noren:.2f}",
     f"{100*(C_base-C_noren)/C_base:.2f}%"],
    ["Optimized TOU + PV + wind (capped)",  f"{C_ren:.2f}",
     f"{100*(C_base-C_ren)/C_base:.2f}%"],
]
df = pd.DataFrame(rows, columns=["Scenario", "Cost (BDT)", "Reduction vs Baseline"])
print("\n--- Final results ---")
print(df.to_string(index=False))
df.to_csv("results_table.csv", index=False)

print("\nDone. All figures and CSVs produced.")
print("Figures: fig_soc, fig_soc_zoom, fig_power, fig_tariff, fig_cost, fig_ablation, fig_sensitivity")
print("CSVs   : results_table, soc_error_table, ablation_table, sensitivity_table")