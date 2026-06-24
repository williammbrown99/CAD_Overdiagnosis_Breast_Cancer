import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-style formatting
sns.set_theme(style="white")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

# ==========================================
# 1. HARDCODED DATA (Updated from Final Metrics)
# ==========================================
years = ['5 Yr', '6 Yr', '7 Yr', '8 Yr', '9 Yr', '10 Yr']
x = np.arange(len(years))  # Label locations
width = 0.2                # Width of the bars

# --- EXPERIMENT 1: Max Clinical Safety ---
safe_p   = [100, 100, 100, 100, 100, 100]
safe_r   = [32, 31, 30, 28, 33, 32]
safe_f1  = [48, 48, 46, 44, 50, 49]
safe_acc = [61, 62, 64, 67, 71, 73] # Calculated as (TP+TN)/Total

# --- EXPERIMENT 2: 1-to-1 Balanced Cost ---
bal_p    = [83, 80, 85, 86, 84, 84]
bal_r    = [90, 87, 77, 78, 75, 78]
bal_f1   = [86, 83, 81, 81, 79, 81]
bal_acc  = [83, 81, 82, 83, 83, 85]

# --- EXPERIMENT 3: Max Indolence (Bounded) ---
maxod_p  = [75, 71, 68, 63, 61, 66]
maxod_r  = [95, 96, 94, 96, 97, 92]
maxod_f1 = [84, 82, 79, 76, 75, 77]
maxod_acc= [79, 76, 74, 72, 72, 78]

# ==========================================
# 2. GENERATE 1x3 GROUPED BAR CHARTS
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
fig.suptitle("Performance Metrics by Cost Optimization (Target: Indolent Cohort)", 
             fontsize=18, fontweight='bold', y=0.98)

# Colors matching standard ML metric visual styles (Blue, Orange, Green, Red)
c_prec = '#1f77b4'
c_rec  = '#ff7f0e'
c_f1   = '#2ca02c'
c_acc  = '#d62728'

def plot_four_bars(ax, prec, rec, f1, acc, title):
    # Plotting the 4 bars per year group
    ax.bar(x - 1.5*width, prec, width, label='Precision', color=c_prec, edgecolor='white')
    ax.bar(x - 0.5*width, rec,  width, label='Recall/Sens',    color=c_rec,  edgecolor='white')
    ax.bar(x + 0.5*width, f1,   width, label='F1 Score',  color=c_f1,   edgecolor='white')
    ax.bar(x + 1.5*width, acc,  width, label='Accuracy',  color=c_acc,  edgecolor='white')

    # Formatting axes
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Score (%)", fontweight='bold')
    ax.set_ylim(0, 105)
    
    # Clean gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', visible=False)
    
    # Despine for clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Plot the three separate graphs
plot_four_bars(axes[0], safe_p, safe_r, safe_f1, safe_acc, "1. Max Clinical Safety")
plot_four_bars(axes[1], bal_p, bal_r, bal_f1, bal_acc, "2. 1-to-1 Balanced Cost")
plot_four_bars(axes[2], maxod_p, maxod_r, maxod_f1, maxod_acc, "3. Max Indolence")

# Add a single master legend to the right of the entire figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5), fontsize=12, frameon=True)

plt.tight_layout(rect=[0, 0.02, 0.92, 0.92]) # Leave room on the right for the master legend
plt.show()
