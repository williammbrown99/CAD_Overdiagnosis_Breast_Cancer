import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set publication-style formatting
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

# 1. LOAD THE MASTER DATA
filename = "Master_Optimization_Data_All_Years.csv"
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"[!] Error: Could not find {filename}. Make sure it is in the same directory.")
    exit()

years = [5, 6, 7, 8, 9, 10]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Clinical Trade-off: Overdiagnosed Cases Avoided vs. Lethal Errors (FP) Introduced", 
             fontsize=18, fontweight='bold', y=0.98)
axes = axes.flatten()

print("="*90)
print(" ADVISOR TRADE-OFF TABLES (ALL THRESHOLDS)")
print("="*90)

for i, yrs in enumerate(years):
    df_yr = df[df['Threshold_Years'] == yrs].copy()
    
    # The baseline for "Max OD" is the FN count when FP = 0 (Max Clinical Safety)
    # This represents the absolute maximum number of overdiagnosed cases the model can flag
    max_fn = df_yr[df_yr['FP'] == 0]['FN'].max()
    
    # Group by unique outcomes to find the ranges of FP_Cost that produce them
    grouped = df_yr.groupby(['FP', 'FN']).agg(
        Max_Cost=('FP_Cost', 'max'),
        Min_Cost=('FP_Cost', 'min')
    ).reset_index()
    
    # Sort descending by cost (to match the advisor's table flow)
    grouped = grouped.sort_values('Max_Cost', ascending=False)
    
    # Calculate Advisor's Metrics
    grouped['Cost_Range'] = grouped.apply(lambda row: f"{int(row['Max_Cost'])} to {int(row['Min_Cost'])}", axis=1)
    grouped['OD_Avoided'] = max_fn - grouped['FN']
    grouped['Ratio'] = np.where(grouped['FP'] > 0, grouped['OD_Avoided'] / grouped['FP'], np.nan)
    
    # Prepare table for terminal output
    print(f"\n--- TABLE FOR {yrs} YEARS (Max Baseline OD cases = {max_fn}) ---")
    print(f"{'FP Cost Range':<15} | {'FP (Missed Agg.)':<18} | {'FN (Missed OD)':<15} | {'OD Avoided':<12} | {'Trade-off Ratio':<15}")
    print("-" * 85)
    
    for _, row in grouped.iterrows():
        ratio_str = f"{row['Ratio']:.2f}" if pd.notnull(row['Ratio']) else "N/A"
        print(f"{row['Cost_Range']:<15} | {int(row['FP']):<18} | {int(row['FN']):<15} | {int(row['OD_Avoided']):<12} | {ratio_str:<15}")

    # ==========================================
    # PLOTTING THE MARGINAL BENEFIT CURVE
    # ==========================================
    ax = axes[i]
    
    # Filter out the FP=0 row for plotting the ratio, as it is undefined (division by zero)
    plot_data = grouped[grouped['FP'] > 0].copy()
    
    # Plot FP vs Ratio
    ax.plot(plot_data['FP'], plot_data['Ratio'], marker='o', color='#d62728', linewidth=2.5, markersize=6)
    
    # Highlight the "sweet spot" (the first point of compromise)
    if not plot_data.empty:
        first_point = plot_data.iloc[0]
        
        ax.plot(first_point['FP'], first_point['Ratio'], marker='*', color='#1f77b4', markersize=12, zorder=5,
                label=f'{first_point["Ratio"]:.1f} OD avoided per 1 FP')
        ax.legend(loc='upper right')

    ax.set_title(f"{yrs}-Year Horizon", fontweight='bold')
    ax.set_xlabel("Lethal Errors Permitted (FP)")
    ax.set_ylabel("OD Cases Avoided per FP")
    
    # Add a baseline at Ratio = 1.0 (where the trade-off becomes 1 life for 1 OD case)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='1:1 Trade-off Boundary')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\n" + "="*90)
print(" ANALYSIS COMPLETE. Plot generated successfully.")
print("="*90)