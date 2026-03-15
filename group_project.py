import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Logic Function for Grid Search ---
def simulate_deterministic_costs(capacity, demand_range, no_show_dist, empty_cost, bump_cost, n_sims=10000):
    results = []
    for booked in demand_range:
        no_show_rates = np.random.triangular(no_show_dist[0], no_show_dist[1], no_show_dist[2], n_sims)
        shows = np.round(booked * (1 - no_show_rates))
        
        empty_seats = np.maximum(0, capacity - shows)
        bumped_pax = np.maximum(0, shows - capacity)
        
        total_costs = (empty_seats * empty_cost) + (bumped_pax * bump_cost)
        
        results.append({
            'booked': booked,
            'mean_cost': np.mean(total_costs),
            'std_cost': np.std(total_costs),
            'p05_cost': np.percentile(total_costs, 5),
            'p96_cost': np.percentile(total_costs, 96), # Updated to 96th Pctile
            'prob_gt_10k': np.mean(total_costs > 10000)
        })
    return pd.DataFrame(results)

# --- Configuration ---
n_simulations = 100000
opt_eco_val, opt_bus_val = 426, 133

# Run Simulations for plotting
eco_df = simulate_deterministic_costs(400, range(400, 451), (0.03, 0.05, 0.08), 450, 150, n_simulations)
bus_df = simulate_deterministic_costs(100, range(100, 151), (0.15, 0.20, 0.30), 1450, 300, n_simulations)


# --- Calculate COMBINED Total Cost for Optimal Configuration ---
# Simulate Economy Costs
eco_rates = np.random.triangular(0.03, 0.05, 0.08, n_simulations)
eco_shows = np.round(opt_eco_val * (1 - eco_rates))
eco_empty = np.maximum(0, 400 - eco_shows)
eco_bumped = np.maximum(0, eco_shows - 400)
opt_eco_costs = (eco_empty * 450) + (eco_bumped * 150)

# Simulate Business Costs
bus_rates = np.random.triangular(0.15, 0.20, 0.30, n_simulations)
bus_shows = np.round(opt_bus_val * (1 - bus_rates))
bus_empty = np.maximum(0, 100 - bus_shows)
bus_bumped = np.maximum(0, bus_shows - 100)
opt_bus_costs = (bus_empty * 1450) + (bus_bumped * 300)

# Add them together element-wise
total_combined_costs = opt_eco_costs + opt_bus_costs

print("="*50)
print(f"TOTAL COMBINED COST METRICS (Eco: {opt_eco_val}, Bus: {opt_bus_val})")
print("="*50)
print(f"Mean Total Cost   : £{np.mean(total_combined_costs):.2f}")
print(f"5th Pctile Cost   : £{np.percentile(total_combined_costs, 5):.2f}")
print(f"96th Pctile Cost  : £{np.percentile(total_combined_costs, 96):.2f}")
print(f"P(Total > £10k)   : {np.mean(total_combined_costs > 10000):.2%}\n")


# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

def annotate_plot(ax, df, opt_val, step=5):
    for i, row in df.iterrows():
        if row['booked'] % step == 0 or row['booked'] == opt_val:
            ax.annotate(f" {int(row['booked'])}", 
                        (row['mean_cost'], row['std_cost']),
                        textcoords="offset points", 
                        xytext=(5,5), 
                        ha='left', fontsize=9, fontweight='bold' if row['booked'] == opt_val else 'normal')

# Economy Plot
ax1.plot(eco_df['mean_cost'], eco_df['std_cost'], 'o-', color='skyblue', alpha=0.3)
opt_eco_stats = eco_df[eco_df['booked'] == opt_eco_val].iloc[0]
ax1.scatter(opt_eco_stats['mean_cost'], opt_eco_stats['std_cost'], color='red', marker='*', s=300, zorder=5, label=f'Optimal: {opt_eco_val}')
annotate_plot(ax1, eco_df, opt_eco_val)

ax1.set_title('Economy: Mean Cost vs Risk', fontsize=14)
ax1.set_xlabel('Mean Cost (£)', fontsize=12)
ax1.set_ylabel('Risk (Std Dev of Cost)', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# Business Plot
ax2.plot(bus_df['mean_cost'], bus_df['std_cost'], 'o-', color='lightgreen', alpha=0.3)
opt_bus_stats = bus_df[bus_df['booked'] == opt_bus_val].iloc[0]
ax2.scatter(opt_bus_stats['mean_cost'], opt_bus_stats['std_cost'], color='red', marker='*', s=300, zorder=5, label=f'Optimal: {opt_bus_val}')
annotate_plot(ax2, bus_df, opt_bus_val)

ax2.set_title('Business: Mean Cost vs Risk', fontsize=14)
ax2.set_xlabel('Mean Cost (£)', fontsize=12)
ax2.set_ylabel('Risk (Std Dev of Cost)', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.show()