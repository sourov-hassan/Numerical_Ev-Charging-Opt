# Numerical Optimization of EV Charging Cost Using TOU Tariffs and Renewable Energy Integration

Reference implementation for the paper:

> **Numerical Optimization of Electric Vehicle Charging Cost Using Time-of-Use Tariffs and Renewable Energy Integration**  
> N. H. Sourov, M. M. H. Zim, S. M. Kabir, M. S. Hossain, T. Islam  
> Department of Electrical and Electronic Engineering, Green University of Bangladesh

## Summary

This repository provides the complete Python implementation and simulation
results for a numerical optimization framework that minimizes EV charging
cost in a hybrid PV-wind-grid charging station under a three-tier time-of-use
(TOU) tariff.

Key results:
- Charging duration: 4.51 h for a 0.2–0.8 SOC window on a 50 kWh battery
- Cost reduction with TOU scheduling only: **47.92%**
- Cost reduction with TOU + PV + wind (6 kW inverter cap): **88.10%**
- RK4 is ~14× more accurate than Euler at Δt = 0.5 h

## Requirements

- Python 3.11 or newer
- See `requirements.txt` for dependencies

```bash
pip install -r requirements.txt
