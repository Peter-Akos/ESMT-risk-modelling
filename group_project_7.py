import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_integrated_flight(eco_booked, bus_booked, n_sims=500000):
    eco_rates = np.random.triangular(0.03, 0.05, 0.08, n_sims)
    bus_rates = np.random.triangular(0.15, 0.20, 0.30, n_sims)
    
    eco_shows = np.round(eco_booked * (1 - eco_rates))
    bus_shows = np.round(bus_booked * (1 - bus_rates))
    
    bus_cap, eco_cap = 100, 400
    
    bus_bumped = np.maximum(0, bus_shows - bus_cap)
    bus_remaining_seats = np.maximum(0, bus_cap - bus_shows)
    
    eco_excess = np.maximum(0, eco_shows - eco_cap)
    pax_to_upgrade = np.minimum(eco_excess, bus_remaining_seats)
    eco_bumped = eco_excess - pax_to_upgrade
    
    final_bus_empty = bus_remaining_seats - pax_to_upgrade
    final_eco_empty = np.maximum(0, eco_cap - eco_shows)
    
    # Costs: Vacant seats vs. Bumping penalties
    costs = (final_eco_empty * 450) + (final_bus_empty * 1450) + \
            (eco_bumped * 150) + (bus_bumped * 300)
    
    return np.mean(costs), np.std(costs)

# --- Configuration ---
RISK_AVERSION = 0.5  # Higher = more weight on avoiding high SD (Standard Deviation)
eco_range = range(390, 440) 
bus_range = range(110, 140)

# --- Execution ---
results = []
for eb in eco_range:
    for bb in bus_range:
        m, s = simulate_integrated_flight(eb, bb)
        # Calculate Risk-Adjusted Cost
        utility = m + (RISK_AVERSION * s)
        results.append({'eco': eb, 'bus': bb, 'mean': m, 'std': s, 'utility': utility})

df = pd.DataFrame(results)

# --- Identify Efficient Frontier ---
frontier = []
for i, row in df.iterrows():
    if not ((df['mean'] < row['mean']) & (df['std'] <= row['std']) | 
            (df['mean'] <= row['mean']) & (df['std'] < row['std'])).any():
        frontier.append(row)
frontier_df = pd.DataFrame(frontier).sort_values('mean')

# --- Select Best Option Based on Risk Aversion ---
best_strategy = df.loc[df['utility'].idxmin()]

# --- Plotting ---
plt.figure(figsize=(12, 7))

# Plot all tested points
plt.scatter(df['mean'], df['std'], c=df['utility'], cmap='viridis', alpha=0.3, label='All Strategies')
plt.colorbar(label=f'Risk-Adjusted Cost (Mean + {RISK_AVERSION}*SD)')

# Plot Frontier Line
plt.plot(frontier_df['mean'], frontier_df['std'], 'r--', linewidth=1.5, label='Efficient Frontier')

# Highlight only the "Best" risk-adjusted choice
plt.scatter(best_strategy['mean'], best_strategy['std'], color='red', s=150, edgecolors='black', zorder=5)
plt.annotate(f"BEST (Risk Aversion={RISK_AVERSION})\nE:{int(best_strategy.eco)}, B:{int(best_strategy.bus)}", 
             (best_strategy['mean'], best_strategy['std']),
             xytext=(15, -15), textcoords='offset points', fontweight='bold', color='red')

plt.title("Booking Strategy: Risk vs. Cost", fontsize=14)
plt.xlabel("Mean Total Cost (£)")
plt.ylabel("Risk (Standard Deviation)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Risk-Averse Recommendation ({RISK_AVERSION}): {int(best_strategy.eco)} Eco / {int(best_strategy.bus)} Bus")
print(f"Expected Cost: £{best_strategy['mean']:.2f} | Std Dev: £{best_strategy['std']:.2f}")


# --- Final Analysis for the Chosen Combination ---
# Parameters for the "Best" strategy found in previous step
best_eco = int(best_strategy.eco)
best_bus = int(best_strategy.bus)

# Re-run simulation with higher fidelity for the chosen point
n_final = 200000000
eco_rates = np.random.triangular(0.03, 0.05, 0.08, n_final)
bus_rates = np.random.triangular(0.15, 0.20, 0.30, n_final)

# Logic execution
eco_shows = np.round(best_eco * (1 - eco_rates))
bus_shows = np.round(best_bus * (1 - bus_rates))

bus_bumped = np.maximum(0, bus_shows - 100)
bus_rem = np.maximum(0, 100 - bus_shows)
eco_excess = np.maximum(0, eco_shows - 400)
pax_upgraded = np.minimum(eco_excess, bus_rem)
eco_bumped = eco_excess - pax_upgraded

final_costs = ((400 - eco_shows).clip(min=0) * 450) + \
              ((bus_rem - pax_upgraded) * 1450) + \
              (eco_bumped * 150) + (bus_bumped * 300)

# --- Metrics Calculation ---
mean_cost = np.mean(final_costs)
std_cost = np.std(final_costs)

# 90% Confidence Interval for the Mean
# Formula: Mean +/- 1.645 * (StdDev / sqrt(N))
z_score = 1.645
margin_of_error = z_score * (std_cost / np.sqrt(n_final))
ci_90 = (mean_cost - margin_of_error, mean_cost + margin_of_error)

# P(Cost > £10,000)
prob_over_10k = np.mean(final_costs > 10000)

print(f"--- Strategy: {best_eco} Eco / {best_bus} Bus ---")
print(f"Mean Cost:            £{mean_cost:,.2f}")
print(f"90% CI (Mean):        £{ci_90[0]:,.2f} to £{ci_90[1]:,.2f}")
print(f"P(Cost > £10,000):    {prob_over_10k:.2%}")